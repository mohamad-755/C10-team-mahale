"""
Step 2: Randomly sample 20 extracted articles and run automated pre-checks.

This does NOT replace your manual judgment - it flags likely problems so
your 20-document manual review (language correctness, encoding integrity,
diacritic/tone survival) goes faster. You still read each one yourself.

Usage:
    python3 02_spot_check_sample.py --input ./data/sw_wikipedia/extracted --n 20

Output:
    spot_check_report.md - one section per sampled article, with:
      - the article title and a text preview
      - automated flags (too short, mostly non-target-language, repeated
        boilerplate, suspicious encoding artifacts)
      - a blank "Manual verdict" line for you to fill in by hand
"""

import argparse
import json
import random
import re
from pathlib import Path

# NOTE: this used to be a hardcoded list of Swahili-specific function words.
# That was a real bug: applying Swahili stopwords to Zulu, Yoruba, etc. would
# falsely flag genuinely correct text as "wrong language" just because those
# languages don't share Swahili's function words. Fixed below by learning
# each corpus's own most common words instead of assuming any one language.

# Rough signal for garbled encoding (mojibake): sequences that show up when
# UTF-8 text gets mis-decoded as Latin-1/Windows-1252 or similar.
MOJIBAKE_PATTERNS = re.compile(r"(Ã.|â€.|Â.)")


def build_corpus_fingerprint(articles, top_n=40):
    """Learn the N most common words across the WHOLE loaded corpus, from
    a sample of articles. This replaces the old hardcoded Swahili stopword
    list with something that works for any language automatically: if a
    document shares almost none of the corpus's own common words, that's
    a reasonable signal it might be mixed-language or misclassified,
    regardless of which of the 8 languages we're checking."""
    from collections import Counter
    counter = Counter()
    sample_for_fingerprint = articles[:2000] if len(articles) > 2000 else articles
    for art in sample_for_fingerprint:
        words = art.get("text", "").split()
        counter.update(w.strip(".,!?()[]\"'").lower() for w in words)
    common = [w for w, _ in counter.most_common(top_n * 2) if len(w) > 1 and not w.isdigit()]
    return set(common[:top_n])


def load_articles(input_dir: Path):
    """wikiextractor writes nested folders (AA/wiki_00, AA/wiki_01, ...),
    each file containing one JSON object per line."""
    articles = []
    json_files = list(input_dir.rglob("wiki_*"))
    if not json_files:
        raise FileNotFoundError(
            f"No extracted files found under {input_dir}. "
            "Did 01_download_and_extract.sh finish successfully?"
        )
    for fp in json_files:
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


def automated_flags(text: str, corpus_fingerprint: set) -> list:
    flags = []
    word_count = len(text.split())

    if word_count < 30:
        flags.append(f"SHORT: only {word_count} words - may be a stub, not real prose")

    if MOJIBAKE_PATTERNS.search(text):
        flags.append("ENCODING: possible mojibake pattern detected (Ã, â€, Â sequences)")

    words_lower = set(w.strip(".,!?()[]\"'").lower() for w in text.split())
    fingerprint_hits = len(words_lower & corpus_fingerprint)
    if word_count > 50 and fingerprint_hits < 3:
        flags.append(
            f"LANGUAGE: only {fingerprint_hits} of this corpus's common words found in "
            f"{word_count} words - check this isn't a different language or code-switched text"
        )

    if text.count("|") > word_count * 0.05 or text.count("{{") > 0:
        flags.append("MARKUP: leftover wiki markup detected - extraction may not have fully cleaned this doc")

    return flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to wikiextractor output directory")
    parser.add_argument("--n", type=int, default=20, help="Number of articles to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed, for a reproducible sample")
    parser.add_argument("--output", default="spot_check_report.md")
    args = parser.parse_args()

    input_dir = Path(args.input)
    articles = load_articles(input_dir)
    print(f"Loaded {len(articles)} total extracted articles from {input_dir}")

    if len(articles) < args.n:
        raise ValueError(f"Only {len(articles)} articles available, cannot sample {args.n}")

    print("Learning this corpus's own common words (language-agnostic fingerprint)...")
    corpus_fingerprint = build_corpus_fingerprint(articles)
    print(f"Top words found: {', '.join(list(corpus_fingerprint)[:10])}...")

    random.seed(args.seed)
    sample = random.sample(articles, args.n)

    lines = [
        f"# Spot Check Report",
        f"",
        f"Random sample of {args.n} articles (seed={args.seed}) from {len(articles)} total extracted.",
        f"For each: read the preview, note the automated flags (if any), then fill in your own manual verdict.",
        f"",
    ]

    for i, art in enumerate(sample, 1):
        title = art.get("title", "UNKNOWN TITLE")
        text = art.get("text", "")
        preview = " ".join(text.split())[:500]
        flags = automated_flags(text, corpus_fingerprint)

        lines.append(f"## {i}. {title}")
        lines.append("")
        lines.append(f"**Preview:** {preview}...")
        lines.append("")
        if flags:
            lines.append("**Automated flags:**")
            for f in flags:
                lines.append(f"- ⚠️ {f}")
        else:
            lines.append("**Automated flags:** none")
        lines.append("")
        lines.append("**Manual verdict** (fill in: PASS / FAIL + reason):")
        lines.append("- Language correct? ")
        lines.append("- Encoding clean? ")
        lines.append("- Diacritics/spelling intact? ")
        lines.append("")
        lines.append("---")
        lines.append("")

    Path(args.output).write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report to {args.output} - open it and fill in the manual verdict for each of the {args.n} articles.")


if __name__ == "__main__":
    main()
