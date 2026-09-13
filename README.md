# Bridging the Tokenization Gap in African LLMs via SuperBPE

Comparing **SuperBPE** (Liu et al., 2025) against a standard BPE baseline
across 9 African languages, to measure whether "superword" tokenization
reduces text fragmentation, and whether that translates into better model
performance.

**Program:** TRI AI Saturdays, Cohort 10

## Dataset

Nine African languages spanning three language families — chosen to test
whether results generalize rather than overfit to one morphological
pattern: Niger-Congo/Bantu (Swahili, Chichewa, Zulu, Kinyarwanda),
Afro-Asiatic (Hausa, Amharic), and Niger-Congo/Volta-Niger (Yoruba, Wolof,
plus Igbo as a documented bonus ninth language).

Sourced from Wikipedia (`dumps.wikimedia.org`, CC-BY-SA-4.0) for all nine
languages, with Wolof supplemented by MasakhaNER 1.0 (Adelani et al., 2021,
CC-BY-4.0-NC) after its Wikipedia-only corpus was flagged as insufficient.
Cleaning (8 rules: min-length filtering, encoding checks, dedup,
near-duplicate template capping, outlier truncation, and leaked-markup
removal) reduced 425,268 raw documents to 286,135 final documents.
Retention varies genuinely by language (27.0% Zulu to 93.4% Igbo) — a
property of each source Wikipedia, not inconsistent cleaning. Chichewa
(204,640 words) and Wolof (458,166 words, even combined) remain the two
most data-scarce languages in the set. Full sourcing, per-rule cleaning
statistics, and known limitations: [`docs/data_card.pdf`](./docs/data_card.pdf).

## Training Pipeline

**Collection & preprocessing:** `wikiextractor` on monthly Wikipedia dumps;
MasakhaNER's CoNLL-tagged Wolof split reconstructed as plain text with NER
tags discarded. Each language passed through an identical 8-rule cleaning
pipeline (re-applied retroactively whenever a new rule was added), then a
fixed-seed (42) 95/5 train/eval split with SHA-256 verification that eval
text is never touched during training.

**Corpus mixing:** corpus sizes vary over 200x across languages (Hausa vs.
Chichewa). We use temperature sampling — `p_i = n_i^0.3 / Sum(n_j^0.3)` — so
small languages get meaningfully more influence without full equalization.

**Model/design choices:** Baseline is standard byte-level BPE,
`vocab_size=24,000`, whitespace-restricted. SuperBPE trains in two matched
stages: Stage 1 learns ordinary subwords to `vocab_size=19,200` (80% of
budget, whitespace-restricted, same as baseline); Stage 2 re-encodes the
corpus with Stage 1, remaps each Stage 1 token to a placeholder character,
and trains a second BPE pass **without** whitespace restriction, filling
the remaining ~4,800 slots with cross-word "superword" merges. Both
tokenizers share the same corpus and total vocab budget, isolating the
algorithm's effect from data or size differences.

## Evaluation

Three independent metrics on held-out eval data:

**Fragmentation (tokens/word):** SuperBPE reduces fragmentation in **every**
language, 10.1% average. Best: Igbo 21.0%. Worst: Amharic 2.3%.

**Compression efficiency (bytes/token):** 11.8% average improvement,
identical per-language ranking to fragmentation — convergent validation
from two independently-computed metrics.

| Language | Baseline tok/word | SuperBPE tok/word | Baseline B/tok | SuperBPE B/tok |
|---|---|---|---|---|
| Igbo | 1.546 | 1.222 | 4.121 | 5.213 |
| Hausa | 1.434 | 1.179 | 4.012 | 4.878 |
| Yoruba | 1.816 | 1.502 | 3.810 | 4.605 |
| Swahili | 1.577 | 1.398 | 4.070 | 4.592 |
| Kinyarwanda | 1.934 | 1.771 | 3.782 | 4.130 |
| Chichewa | 1.769 | 1.682 | 4.048 | 4.257 |
| Wolof | 1.656 | 1.581 | 3.255 | 3.407 |
| Zulu | 2.489 | 2.407 | 3.759 | 3.887 |
| Amharic | 2.897 | 2.830 | 5.232 | 5.356 |

**Model comparison (bits/byte):** matched ~9.4M-param GPT-2-style models,
3 epochs, same raw text. Baseline: 1.8657. SuperBPE: 2.0125 (7.9% worse) —
a genuine negative result at this tiny scale, likely because SuperBPE's
coarser tokenization means fewer gradient updates per unit of text.

**Known caveat:** a data-hygiene issue (the corpus-build step didn't fully
exclude eval-holdout text from training) was found after these results were
produced. It affects both tokenizers identically, so the relative
comparison holds, but absolute figures should be read with that in mind.

## Reproduction

1. `src/data_pipeline/01_download_and_extract.sh` through
   `05_convert_masakhaner_to_corpus.py`, in order — rebuilds cleaned corpora
   and eval splits per language.
2. `notebooks/tokenizer_training.ipynb` — trains baseline BPE and
   two-stage SuperBPE, reproduces the fragmentation/compression numbers
   above. Exact figures also saved in `results/*.json`.
3. `src/tokenizer_submission/` — a separately-adapted, ASCII-only version
   of this tokenizer built for a related Codabench SuperBPE competition
   (3rd place), kept byte-identical to the code actually submitted there.

Large artifacts (combined corpus, trained tokenizer files, model weights)
are kept in Google Drive rather than git — see
[`drive_structure.md`](./drive_structure.md) for exact paths.

## Appendix

**Contributors:** Mohamad Ziadah - Zione Kamtambe - Amale Herbert - Thereza Pete - Fatu Kromah


## References

Liu, A., Hayase, J., Hofmann, V., Oh, S., Smith, N. A., & Choi, Y. (2025).
SuperBPE: Space travel for language models. *arXiv:2503.13423*.

Adelani, D. I., et al. (2021). MasakhaNER: Named Entity Recognition for
African Languages. `github.com/masakhane-io/masakhane-ner`.
