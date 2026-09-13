"""
Step 3: Clean the extracted corpus.

Applies four cleaning rules, each justified by something we actually found
or that the SuperBPE paper itself flagged as a real risk:

1. MIN WORD COUNT FILTER
   Our 20-doc spot check found that 6/20 "articles" (30%) were actually
   empty stubs (date pages, infobox-only pages) with 0 extracted words.
   Anything below --min-words is dropped.

2. EXACT DEDUPLICATION
   Wikipedia dumps can contain redirect artifacts or duplicate snapshots.
   We hash each article's text and drop exact repeats.

3. LONG-DOCUMENT TRUNCATION
   The SuperBPE paper (Liu et al., 2025) found a single document duplicated
   2,224 times inside one file, which distorted their tokenizer training.
   As a safeguard, we truncate the longest 1% of documents to the 99th
   percentile length (their exact mitigation), rather than assuming our
   corpus doesn't have a similar issue.

4. BASIC ENCODING SANITY CHECK
   Drops any document where mojibake patterns are detected (same heuristic
   as the spot-check script), since these are unrecoverable during cleaning
   and better excluded than fed to a tokenizer.

Usage:
    python3 03_clean_corpus.py \\
        --input ./data/sw_wikipedia/extracted \\
        --output ./data/sw_wikipedia/cleaned \\
        --min-words 15

Output:
    <output>/corpus.txt         - one cleaned article per line, ready for
                                   tokenizer training
    <output>/cleaning_report.md - stats on what was dropped and why, to
                                   paste into your Data Card later
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

MOJIBAKE_PATTERNS = re.compile(r"(Ã.|â€.|Â.)")

# Found during the Zulu Wikipedia spot-check (articles "IsiLatviya" and
# "City of Cape Town") leaked raw revision metadata into the extracted
# text. Checking the raw JSON directly revealed the tags are HTML-escaped
# entities, not literal angle brackets - e.g. the real text contains
# "&lt;ns&gt;0&lt;/ns&gt; &lt;revision&gt; &lt;parentid&gt;41414&lt;/parentid&gt;..."
# not "<ns>0</ns> <revision>...". A first attempt at this regex used literal
# < > and matched nothing, silently leaving all 59 leaked instances in the
# cleaned corpus - a good reminder to check the raw data before assuming
# a fix worked, not just checking that the code exists.
LEAKED_XML_METADATA = re.compile(
    r"&lt;ns&gt;\d+&lt;/ns&gt;\s*&lt;revision&gt;.*?&lt;/format&gt;\s*", re.DOTALL
)


def strip_leaked_xml_metadata(text: str) -> str:
    return LEAKED_XML_METADATA.sub("", text)


# Found during the Igbo spot-check (article "Ahmed Saidu Baba" / "SystemSpecs"):
# leaked <templatestyles> markup, same HTML-entity encoding as the Zulu
# revision-metadata leak. First version of this regex only matched when
# nothing but whitespace sat between the open/close tags - that covered
# Igbo's case but missed a Yoruba example where literal text ("Infobox ")
# sat between the tags, e.g.:
#   &lt;templatestyles src="Module:Infobox/styles.css"&gt;Infobox &lt;/templatestyles&gt;
# Fixed to match ANY content between the tags (non-greedy), verified against
# both real examples before trusting it.
LEAKED_TEMPLATESTYLES = re.compile(
    r"&lt;templatestyles[^&]*?&gt;.*?&lt;/templatestyles&gt;", re.DOTALL
)


def strip_leaked_templatestyles(text: str) -> str:
    return LEAKED_TEMPLATESTYLES.sub("", text)


# A second, messier variant found later in Igbo: raw JSON "data-mw" template
# metadata leaking into text with no consistent tag structure, e.g.:
#   ":"templatestyles","attrs":{"src":"Fraction/styles.css"},"body":{...
#   &lt;data-mw='{"name":"templatestyles",...}]
#   templatestyles UK: /aı'zaı.ǝ/   (bare word, no JSON at all)
# Unlike the clean tag-pair case above, this debris has no reliable closing
# marker, and in at least one real example ran with NO space at all into
# following real Igbo words (e.g. ...\"ak\u1ee5k\u1ee5\":[{\"ihe, where "ihe"
# is real content stuck directly onto the JSON with no separator). There is
# no way to draw a fully safe boundary here.
#
# DELIBERATE, DOCUMENTED TRADEOFF: strip the entire non-whitespace token
# containing "templatestyles". This reliably removes the JSON debris (which
# is the more damaging problem - it's mostly non-linguistic content), at the
# cost of occasionally also removing a few real words directly adjacent to
# it with no space boundary. Affects an estimated ~5/57,457 Igbo documents
# (0.009%) - noted explicitly in the Data Card rather than silently accepted.
TEMPLATESTYLES_DEBRIS = re.compile(r"\S*templatestyles\S*")


def strip_templatestyles_debris(text: str) -> str:
    return TEMPLATESTYLES_DEBRIS.sub("", text)


# Found during the Igbo spot-check: 5 of 20 articles had leftover citation
# reference markers like [1], [2][3][4][5] where wikiextractor didn't fully
# strip Wikipedia's footnote superscripts. These are noise, not content -
# a tokenizer shouldn't learn "[1][2][3]" as a meaningful pattern.
CITATION_MARKERS = re.compile(r"\[\d{1,3}\]")


def strip_citation_markers(text: str) -> str:
    return CITATION_MARKERS.sub("", text)


# Found during the Yoruba spot-check: 10 of 20 sampled articles were
# near-identical bot-generated asteroid stubs, e.g.
#   "404 Arsinoë jẹ́ plánẹ́tì kékeré ní ibi ìgbàjá ástẹ́rọ́ìdì."
#   "10918 Kodaly jẹ́ plánẹ́tì kékeré ní ibi ìgbàjá ástẹ́rọ́ìdì."
# These are NOT exact duplicates (the name/number differs), so the exact-hash
# dedup above lets every one of them through as a "unique" document, even
# though they contribute almost no new language content. This is the same
# risk the SuperBPE paper found with one document duplicated 2,224 times,
# just spread across many near-duplicates instead of one exact repeat.
#
# Fix: build a "template skeleton" per document by masking out numbers and
# capitalized words (a rough, language-agnostic proxy for proper nouns/names,
# since we can't rely on a real named-entity tagger for 8 low-resource
# languages). Documents that reduce to the identical skeleton are almost
# certainly instances of the same template. We keep only the first
# --max-per-template examples of any skeleton and drop the rest.
NUMBER_PATTERN = re.compile(r"\d+")


def template_skeleton(text: str) -> str:
    words = text.split()
    masked = []
    for w in words:
        stripped = w.strip(".,!?()[]\"'")
        if NUMBER_PATTERN.search(stripped):
            masked.append("<NUM>")
        elif stripped[:1].isupper():
            masked.append("<NAME>")
        else:
            masked.append(stripped.lower())
    return " ".join(masked)


def load_articles(input_dir: Path):
    articles = []
    for fp in input_dir.rglob("wiki_*"):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    articles.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return articles


def clean_text(text: str) -> str:
    # Collapse whitespace/newlines left over from extraction into single spaces
    return " ".join(text.split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-words", type=int, default=15,
                         help="Drop documents with fewer words than this (default 15)")
    parser.add_argument("--truncate-percentile", type=float, default=99.0,
                         help="Truncate the longest documents to this percentile length")
    parser.add_argument("--max-per-template", type=int, default=3,
                         help="Keep at most this many documents that reduce to the same "
                              "template skeleton (e.g. bot-generated stub patterns). "
                              "Set to 0 to disable this check.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading articles from {input_dir} ...")
    articles = load_articles(input_dir)
    total_loaded = len(articles)
    print(f"Loaded {total_loaded} articles")

    # ---- Step A: strip leaked XML/template metadata, citation markers, clean whitespace, compute word counts ----
    stripped_xml_count = 0
    stripped_templatestyles_count = 0
    stripped_citations_count = 0
    for art in articles:
        raw = art.get("text", "")
        step1 = strip_leaked_xml_metadata(raw)
        if step1 != raw:
            stripped_xml_count += 1
        step2 = strip_leaked_templatestyles(step1)
        if step2 != step1:
            stripped_templatestyles_count += 1
        step2b = strip_templatestyles_debris(step2)
        if step2b != step2:
            stripped_templatestyles_count += 1
        step3 = strip_citation_markers(step2b)
        if step3 != step2:
            stripped_citations_count += 1
        art["clean_text"] = clean_text(step3)
        art["word_count"] = len(art["clean_text"].split())

    # ---- Step B: drop too-short documents ----
    before = len(articles)
    articles = [a for a in articles if a["word_count"] >= args.min_words]
    dropped_short = before - len(articles)

    # ---- Step C: drop mojibake / encoding-suspect documents ----
    before = len(articles)
    articles = [a for a in articles if not MOJIBAKE_PATTERNS.search(a["clean_text"])]
    dropped_encoding = before - len(articles)

    # ---- Step D: exact deduplication ----
    seen_hashes = set()
    deduped = []
    dropped_duplicates = 0
    for art in articles:
        h = hashlib.sha256(art["clean_text"].encode("utf-8")).hexdigest()
        if h in seen_hashes:
            dropped_duplicates += 1
            continue
        seen_hashes.add(h)
        deduped.append(art)
    articles = deduped

    # ---- Step D.5: cap near-duplicate templated documents ----
    # (found during Yoruba spot-check: bot-generated asteroid stubs)
    dropped_template = 0
    templates_found = 0
    if args.max_per_template > 0:
        template_counts = {}
        capped = []
        for art in articles:
            skel = template_skeleton(art["clean_text"])
            n = template_counts.get(skel, 0)
            if n == 0:
                templates_found += 1  # first time seeing this skeleton (not yet counted as "a template" unless it repeats, but we count all skeletons seen and only report ones that repeated below)
            if n < args.max_per_template:
                capped.append(art)
                template_counts[skel] = n + 1
            else:
                dropped_template += 1
        articles = capped
        # Recount: only skeletons that actually occurred more than once are "templates"
        templates_found = sum(1 for skel, n in template_counts.items() if n >= args.max_per_template)

    # ---- Step E: truncate outlier-long documents (99th percentile) ----
    word_counts = sorted(a["word_count"] for a in articles)
    if word_counts:
        idx = int(len(word_counts) * (args.truncate_percentile / 100.0))
        idx = min(idx, len(word_counts) - 1)
        cutoff = word_counts[idx]
    else:
        cutoff = 0

    truncated_count = 0
    for art in articles:
        words = art["clean_text"].split()
        if len(words) > cutoff:
            art["clean_text"] = " ".join(words[:cutoff])
            art["word_count"] = cutoff
            truncated_count += 1

    # ---- Write outputs ----
    corpus_path = output_dir / "corpus.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for art in articles:
            f.write(art["clean_text"] + "\n")

    final_count = len(articles)
    final_words = sum(a["word_count"] for a in articles)

    report_lines = [
        "# Cleaning Report",
        "",
        f"**Input:** {input_dir}",
        f"**Output:** {corpus_path}",
        "",
        "## Pipeline results",
        "",
        f"| Stage | Count | Notes |",
        f"|---|---|---|",
        f"| Loaded | {total_loaded} | Raw extracted articles |",
        f"| XML metadata stripped | {stripped_xml_count} | Leaked revision metadata removed (found during Zulu spot-check), text kept |",
        f"| Templatestyles markup stripped | {stripped_templatestyles_count} | Leaked infobox template tags removed (found during Igbo spot-check), text kept |",
        f"| Citation markers stripped | {stripped_citations_count} | Leftover [1][2][3] footnote references removed (found during Igbo spot-check), text kept |",
        f"| Dropped (< {args.min_words} words) | {dropped_short} | Empty/stub pages - matches what the 20-doc spot check found |",
        f"| Dropped (encoding issue) | {dropped_encoding} | Mojibake pattern detected |",
        f"| Dropped (exact duplicate) | {dropped_duplicates} | Same content hash as an earlier document |",
        f"| Dropped (over template cap) | {dropped_template} | Reduced to a skeleton (numbers/names masked) that already reached {args.max_per_template} kept examples - found via Yoruba's asteroid-stub pattern |",
        f"| Truncated (outlier length) | {truncated_count} | Cut to {cutoff}-word cap ({args.truncate_percentile}th percentile), per SuperBPE paper's mitigation |",
        f"| **Final document count** | **{final_count}** | |",
        f"| **Final total words** | **{final_words:,}** | |",
        "",
        f"**Retention rate:** {final_count}/{total_loaded} = {100*final_count/total_loaded:.1f}% of raw articles kept",
        f"**Distinct repeated templates detected:** {templates_found} (each capped at {args.max_per_template} kept examples)",
        "",
        "## For the Data Card",
        "",
        f"- Source: Wikipedia (dump extracted via wikiextractor)",
        f"- Cleaning rules applied: minimum word count ({args.min_words}), "
        f"encoding sanity check, exact deduplication, near-duplicate template "
        f"capping (max {args.max_per_template} per skeleton), outlier-length truncation "
        f"(cap at {cutoff} words, {args.truncate_percentile}th percentile)",
        f"- Known limitation: stub/list/infobox-only pages are systematically "
        f"excluded, so this corpus skews toward articles with substantial prose",
        f"- Known limitation: bot-generated templated stub articles (e.g. one-per-catalog-entry "
        f"astronomy pages) are capped rather than removed entirely, to preserve some "
        f"representation of this real article type without letting it dominate the corpus",
        f"- Known limitation (tonal languages - Yoruba, Igbo): tone/diacritic marking is "
        f"inconsistently applied across articles in the same corpus (e.g. Igbo 'amụrụ' vs "
        f"'amuru' for the same word, confirmed in a 20-doc manual sample). This cannot be "
        f"corrected automatically without a real morphological analyzer or native-speaker "
        f"review, per the project's morphological rule toolkit (Section B.3 consultation protocol).",
        f"- Known limitation: a small number of documents (~0.01% in Igbo) contained raw "
        f"JSON template metadata with no reliable boundary from surrounding text; removing "
        f"it may have also removed a few adjacent real words with no whitespace separator. "
        f"Documented rather than left silently in the corpus.",
    ]

    report_path = output_dir / "cleaning_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\nDone. {final_count}/{total_loaded} articles kept ({100*final_count/total_loaded:.1f}%).")
    print(f"Cleaned corpus: {corpus_path}")
    print(f"Report:         {report_path}")


if __name__ == "__main__":
    main()
