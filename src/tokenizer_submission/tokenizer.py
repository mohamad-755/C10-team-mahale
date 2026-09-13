"""
SuperBPE-style tokenizer for the ASCII-normalized African-language tokenization
challenge. Pure Python, standard library only -- no dependency on the
`tokenizers` or `transformers` packages, since neither is guaranteed to be
present in the competition's runtime environment.

Design summary:
  - Input text arrives already preprocessed to pure ASCII by the evaluator.
    Non-ASCII source characters appear as explicit markers, e.g.
    "[U+00E9 LATIN SMALL LETTER E WITH ACUTE]".
  - Each recognized marker is collapsed into a single internal placeholder
    character (Unicode Supplementary Private Use Area-A, 0xF0000+) before
    tokenization, so that pretokenization/merging is not disrupted by the
    marker's own internal spaces and punctuation. Markers that cannot be
    safely round-tripped (never seen during training, or a reconstruction
    mismatch) are left as literal text instead -- still lossless, just
    less compressed.
  - Stage 1 is ordinary BPE, trained with word-boundary-respecting
    (Metaspace-style) pretokenization.
  - Stage 2 is trained on top of Stage 1's *output*, with each Stage 1
    token id mapped to its own placeholder character (Plane 16, 0x100000+)
    and no word-boundary restriction at all -- this is what allows merges
    to span whitespace ("superwords").
  - Both stages use a priority-queue-based BPE merge (O(n log n)) rather
    than the naive rescan-everything approach (O(n^2)), which matters a
    lot for Stage 2 since it processes whole documents with no word
    splitting.
"""

import heapq
import json
import os
import re
import unicodedata

PLACEHOLDER_BASE = 0xF0000          # marker-collapse placeholders
STAGE2_PLACEHOLDER_BASE = 0x100000  # stage-1-token placeholders (kept in a
                                     # separate Unicode range so the two
                                     # schemes can never collide)
STAGE1_REPLACEMENT = "\u2581"       # Metaspace's real word-boundary marker

MARKER_RE = re.compile(r"\[U\+([0-9A-Fa-f]+) ([^\]]*)\]")

_DATA_FILENAME = "tokenizer_data.json"


def _bpe_merge_efficient(symbols, rank_table):
    """
    Standard BPE merge, applied greedily in learned-merge-priority order,
    using a min-heap + doubly linked list so each merge only touches its
    immediate neighbors instead of rescanning the whole sequence.
    """
    symbols = list(symbols)
    n = len(symbols)
    if n <= 1:
        return symbols

    prev = list(range(-1, n - 1))
    next_ = list(range(1, n + 1))
    next_[-1] = -1
    active = [True] * n
    heap = []

    def maybe_push(i):
        j = next_[i]
        if j == -1:
            return
        pair = (symbols[i], symbols[j])
        rank = rank_table.get(pair)
        if rank is not None:
            heapq.heappush(heap, (rank, i, j, symbols[i], symbols[j]))

    for i in range(n - 1):
        maybe_push(i)

    while heap:
        rank, i, j, left_val, right_val = heapq.heappop(heap)
        if not active[i] or not active[j]:
            continue
        if symbols[i] != left_val or symbols[j] != right_val:
            continue
        if next_[i] != j:
            continue  # stale candidate: no longer adjacent

        symbols[i] = left_val + right_val
        active[j] = False
        nj = next_[j]
        next_[i] = nj
        if nj != -1:
            prev[nj] = i

        pi = prev[i]
        if pi != -1:
            maybe_push(pi)
        maybe_push(i)

    result = []
    i = 0
    while i != -1:
        if active[i]:
            result.append(symbols[i])
        i = next_[i]
    return result


class Tokenizer:
    def __init__(self):
        data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), _DATA_FILENAME
        )
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._s1_vocab = data["stage1_vocab"]
        self._s1_merges = data["stage1_merges"]
        self._s1_rank = {
            (left, right): i for i, (left, right) in enumerate(self._s1_merges)
        }
        self._s1_unk_id = self._s1_vocab["[UNK]"]
        self._s1_id_to_tok = {v: k for k, v in self._s1_vocab.items()}

        self._s2_vocab = data["stage2_vocab"]
        self._s2_merges = data["stage2_merges"]
        self._s2_rank = {
            (left, right): i for i, (left, right) in enumerate(self._s2_merges)
        }
        self._s2_unk_id = self._s2_vocab["[UNK]"]
        self._s2_id_to_tok = {v: k for k, v in self._s2_vocab.items()}

        # markers only get collapsed to a placeholder if that exact placeholder
        # was actually part of Stage 1's learned vocabulary -- otherwise the
        # marker is left as literal ASCII text (always encodable, just less
        # compressed), which is what keeps this lossless for genuinely novel
        # foreign characters that never appeared during training.
        self._known_safe_placeholders = {
            ch
            for ch in self._s1_vocab.keys()
            if len(ch) == 1 and ord(ch) >= PLACEHOLDER_BASE
        }

        self._id_to_stage2_placeholder = {
            tok_id: chr(STAGE2_PLACEHOLDER_BASE + tok_id)
            for tok_id in self._s1_vocab.values()
        }
        self._stage2_placeholder_to_id = {
            v: k for k, v in self._id_to_stage2_placeholder.items()
        }

    # ---- marker collapse / expand ----

    def _collapse_markers(self, text):
        def replace(m):
            codepoint = int(m.group(1), 16)
            expected = (
                f"[U+{codepoint:04X} "
                f"{unicodedata.name(chr(codepoint), 'UNKNOWN CHARACTER')}]"
            )
            if expected != m.group(0):
                return m.group(0)  # can't reconstruct exactly -> leave as literal text
            placeholder = chr(PLACEHOLDER_BASE + codepoint)
            return (
                placeholder
                if placeholder in self._known_safe_placeholders
                else m.group(0)  # never seen in training -> leave as literal text
            )

        return MARKER_RE.sub(replace, text)

    def _expand_markers(self, text):
        out = []
        for ch in text:
            cp = ord(ch)
            if cp >= PLACEHOLDER_BASE:
                real_cp = cp - PLACEHOLDER_BASE
                name = unicodedata.name(chr(real_cp), "UNKNOWN CHARACTER")
                out.append(f"[U+{real_cp:04X} {name}]")
            else:
                out.append(ch)
        return "".join(out)

    # ---- stage 1: word-boundary-respecting BPE ----

    def _stage1_encode(self, text):
        ids = []
        for word in text.split(" "):
            chunk = STAGE1_REPLACEMENT + word
            symbols = _bpe_merge_efficient(list(chunk), self._s1_rank)
            for sym in symbols:
                ids.append(self._s1_vocab.get(sym, self._s1_unk_id))
        return ids

    def _stage1_decode(self, ids):
        joined = "".join(self._s1_id_to_tok.get(i, "[UNK]") for i in ids)
        result = joined.replace(STAGE1_REPLACEMENT, " ")
        if result.startswith(" "):
            result = result[1:]  # the very first word's marker isn't a real space
        return result

    # ---- stage 2: unrestricted ("super") BPE across stage-1 tokens ----

    def _stage2_encode(self, text):
        symbols = _bpe_merge_efficient(list(text), self._s2_rank)
        return [self._s2_vocab.get(s, self._s2_unk_id) for s in symbols]

    def _stage2_decode(self, ids):
        return "".join(self._s2_id_to_tok.get(i, "[UNK]") for i in ids)

    # ---- full pipeline ----

    def _encode_one(self, text):
        collapsed = self._collapse_markers(text)
        stage1_ids = self._stage1_encode(collapsed)
        placeholder_str = "".join(
            self._id_to_stage2_placeholder[i] for i in stage1_ids
        )
        return self._stage2_encode(placeholder_str)

    def _decode_one(self, ids):
        placeholder_str = self._stage2_decode(ids)
        stage1_ids = [
            self._stage2_placeholder_to_id[ch]
            for ch in placeholder_str
            if ch in self._stage2_placeholder_to_id
        ]
        collapsed = self._stage1_decode(stage1_ids)
        return self._expand_markers(collapsed)

    # ---- public interface ----

    def encode(self, texts):
        """Encode each input string as a list of integer token IDs."""
        return [self._encode_one(t) for t in texts]

    def decode(self, encoded_texts):
        """Reconstruct every original input string exactly."""
        return [self._decode_one(ids) for ids in encoded_texts]
