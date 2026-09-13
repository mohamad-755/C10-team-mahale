"""
Step: Convert MasakhaNER CoNLL-format data into plain-text corpus lines.

MasakhaNER data is one token per line, tagged with an NER label (O, B-PER,
I-PER, B-LOC, etc.), separated by a space, with blank lines marking sentence
boundaries:

    SAFIYETU B-PER
    BÉEY I-PER
    Céy O
    Koronaa O
    ! O

This script drops the NER tags, rejoins tokens into plain sentences, and
combines train/dev/test splits into a single corpus (the NER train/dev/test
split is irrelevant here - we only care about the sentences as raw text for
tokenizer training, not as labeled examples).

Applies two light sanity checks, consistent with 03_clean_corpus.py:
- drop sentences below --min-words (default 3, much lower than the 15 used
  for Wikipedia, since these are naturally short single sentences, not
  full articles - comparing them against an article-length threshold
  would drop nearly everything)
- exact deduplication, in case a sentence repeats across train/dev/test

Usage:
    python3 05_convert_masakhaner_to_corpus.py \\
        --input-dir ./masakhane-ner/data/wol \\
        --output ./data/wo_masakhaner/cleaned \\
        --min-words 3

Output:
    <output>/corpus.txt          - one reconstructed sentence per line
    <output>/conversion_report.md - stats on what was combined/dropped
"""

import argparse
import hashlib
from pathlib import Path


def parse_conll_file(path: Path) -> list:
    """Read a CoNLL file and return a list of plain-text sentences,
    reconstructed by rejoining tokens with spaces (NER tags discarded)."""
    sentences = []
    current_tokens = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if current_tokens:
                    sentences.append(" ".join(current_tokens))
                    current_tokens = []
                continue
            # Expected format: "TOKEN TAG" - split on the LAST space in case
            # a token itself happens to contain a space (defensive, though
            # not expected in this dataset)
            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                # Malformed line - skip rather than guess
                continue
            token, _tag = parts
            current_tokens.append(token)

    # Catch a final sentence if the file doesn't end with a blank line
    if current_tokens:
        sentences.append(" ".join(current_tokens))

    return sentences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True,
                         help="Directory containing train.txt, dev.txt, test.txt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-words", type=int, default=3,
                         help="Drop sentences with fewer words than this (default 3)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train.txt", "dev.txt", "test.txt"]
    all_sentences = []
    per_split_counts = {}

    for split in splits:
        split_path = input_dir / split
        if not split_path.exists():
            print(f"WARNING: {split_path} not found, skipping")
            per_split_counts[split] = 0
            continue
        sentences = parse_conll_file(split_path)
        per_split_counts[split] = len(sentences)
        all_sentences.extend(sentences)

    total_parsed = len(all_sentences)

    # Drop too-short sentences
    before = len(all_sentences)
    all_sentences = [s for s in all_sentences if len(s.split()) >= args.min_words]
    dropped_short = before - len(all_sentences)

    # Exact deduplication (a sentence could repeat across train/dev/test,
    # or within a split)
    seen_hashes = set()
    deduped = []
    dropped_duplicates = 0
    for s in all_sentences:
        h = hashlib.sha256(s.encode("utf-8")).hexdigest()
        if h in seen_hashes:
            dropped_duplicates += 1
            continue
        seen_hashes.add(h)
        deduped.append(s)
    all_sentences = deduped

    corpus_path = output_dir / "corpus.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for s in all_sentences:
            f.write(s + "\n")

    final_count = len(all_sentences)
    final_words = sum(len(s.split()) for s in all_sentences)

    report_lines = [
        "# MasakhaNER Conversion Report (Wolof)",
        "",
        f"**Input:** {input_dir}",
        f"**Output:** {corpus_path}",
        "",
        "## Per-split sentence counts (before filtering)",
        "",
        "| Split | Sentences |",
        "|---|---|",
    ]
    for split in splits:
        report_lines.append(f"| {split} | {per_split_counts.get(split, 0)} |")

    report_lines += [
        "",
        "## Pipeline results",
        "",
        "| Stage | Count | Notes |",
        "|---|---|---|",
        f"| Parsed (train+dev+test combined) | {total_parsed} | NER tags discarded, tokens rejoined into sentences |",
        f"| Dropped (< {args.min_words} words) | {dropped_short} | Below minimum sentence length |",
        f"| Dropped (exact duplicate) | {dropped_duplicates} | Same sentence text appeared more than once (e.g. across splits) |",
        f"| **Final sentence count** | **{final_count}** | |",
        f"| **Final total words** | **{final_words:,}** | |",
        "",
        "## For the Data Card",
        "",
        "- Source: MasakhaNER 1.0 (github.com/masakhane-io/masakhane-ner), Wolof (`wol`) split",
        "- Domain: local news articles, human-annotated by Masakhane community volunteers",
        "- License: CC-BY-4.0-NC (per MasakhaNER repo)",
        "- Collection method: NER tags discarded; original train/dev/test task split "
        "not preserved, since all three are treated as plain text here, not as "
        "labeled examples",
        f"- Cleaning applied: minimum sentence length ({args.min_words} words), exact deduplication",
        "- Known limitation: sentences are shorter and more fragmented than Wikipedia "
        "article text (news-sentence granularity rather than full articles), so "
        "this source changes the domain/register mix of the combined Wolof corpus "
        "rather than just adding more of the same kind of text",
        "- Known limitation: no wiki-markup-style leaks expected (not extracted from "
        "Wikipedia dumps), so the Wikipedia-specific stripping rules in "
        "03_clean_corpus.py were not applied here and are not needed",
    ]

    report_path = output_dir / "conversion_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Parsed {total_parsed} sentences from train+dev+test")
    print(f"Dropped {dropped_short} short, {dropped_duplicates} duplicate")
    print(f"Final: {final_count} sentences, {final_words:,} words")
    print(f"Corpus: {corpus_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
