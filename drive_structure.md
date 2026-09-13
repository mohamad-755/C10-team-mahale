# Expected Google Drive structure

The scripts in `src/data_pipeline/` and notebooks in `notebooks/` expect the
following layout under `DATA_ROOT/` in Google Drive (this folder was
renamed from `TRI-Project/` after Phase 4). This data is not included in
the git repo (see `.gitignore`) — see `docs/data_card.pdf` for full sourcing
and cleaning documentation.

```
DATA_ROOT/
├── am_wikipedia/
│   ├── spot_check_report_am.md     # manual spot-check notes (src/data_pipeline/02_spot_check_sample.py)
│   ├── cleaned/
│   │   ├── corpus.txt              # 100% of cleaned data, BEFORE the eval-holdout split
│   │   └── cleaning_report.md      # output of src/data_pipeline/03_clean_corpus.py
│   └── train/
│       └── corpus.txt              # 95% split, holdout excluded -- use THIS, not
│                                    # cleaned/corpus.txt, for any future corpus-building
│
├── ha_wikipedia/            (same structure, spot_check_report_ha.md)
├── ig_wikipedia/            (same, ig)
├── ny_wikipedia/            (same, ny)
├── rw_wikipedia/            (same, rw)
├── sw_wikipedia/            (same, sw)
├── yo_wikipedia/            (same, yo)
├── zu_wikipedia/            (same, zu)
│
├── wo_wikipedia/            # Wolof, Wikipedia-only source (smaller; NOT used for training)
│   ├── spot_check_report_wo.md
│   ├── cleaned/
│   └── train/
├── wo_combined/             # Wolof, Wikipedia + MasakhaNER -- the source actually used
│   ├── spot_check_report_wo.md     # for training (see docs/data_card.pdf, Section 2.1)
│   ├── cleaned/
│   └── train/
│
├── eval_holdout/            # top-level folder, NOT nested inside each language folder
│   ├── am_wikipedia/
│   │   ├── eval.txt
│   │   └── eval_manifest.md
│   ├── ha_wikipedia/        (same: eval.txt + eval_manifest.md)
│   ├── ig_wikipedia/        (same)
│   ├── ny_wikipedia/        (same)
│   ├── rw_wikipedia/        (same)
│   ├── sw_wikipedia/        (same)
│   ├── wo_wikipedia/        (same -- Wolof's Wikipedia-only eval split)
│   ├── wo_combined/         (same -- Wolof's combined eval split, matches the training source)
│   ├── yo_wikipedia/        (same)
│   └── zu_wikipedia/        (same)
│
├── tri_ai_archive/                          # all pipeline artifacts from Phases 2-5
│   ├── combined_corpus.txt                  # Phase 2: temperature-sampled (α=0.3) combined training text
│   ├── stage2_meta_corpus.txt               # Phase 3: placeholder-character corpus for Stage 2 training
│   ├── baseline_bpe_tokenizer.json          # Phase 2 output, vocab_size=24000
│   ├── superbpe_stage1_tokenizer.json       # Phase 3 Stage 1 output, vocab_size=19200
│   ├── superbpe_stage2_raw.json             # Phase 3 Stage 2 output, vocab_size=24000
│   ├── model_train_text.txt                 # Phase 4: training subset (19,000 lines)
│   ├── model_eval_text.txt                  # Phase 4: held-out subset (1,000 lines)
│   ├── baseline_fragmentation_results.json  # Phase 2: fragmentation eval (tokens/word)
│   ├── superbpe_fragmentation_results.json  # Phase 3: fragmentation eval (tokens/word)
│   ├── model_comparison_results.json        # Phase 4: bits-per-byte comparison
│   ├── compression_efficiency_results.json  # Phase 5: bytes/token comparison
│   └── DATA_CARD.md                         # source of truth for docs/data_card.pdf
│
├── checkpoints_all_9langs/                      # Phase 4: single model trained on the combined 9-language corpus
│   └── *.pt at steps 1000, 2000, 3000, 4000, 4999, plus latest.pt
├── checkpoints_per_language/                    # Phase 4 variant: models trained per language
│   └── *.pt at the same steps, plus latest.pt
├── checkpoints_per_language_tuned_beta/         # same, with a tuned beta hyperparameter
│   └── *.pt at the same steps, plus latest.pt
└── checkpoints_per_language_tuned_beta_lowreg/  # same, tuned beta + lower regularization
    └── *.pt at the same steps, plus latest.pt
```

Note: the four `checkpoints_*/` folders' internal per-language subfolder
naming (if any) hasn't been independently confirmed against Drive for this
doc update -- verify before scripting against paths inside them.

## Wolof: two source folders, only one used for training

Wolof is the only language with two separate source folders, both at the
top level and under `eval_holdout/`:

- `wo_wikipedia/` — Wolof Wikipedia alone, which turned out too small to
  be usable on its own.
- `wo_combined/` — Wikipedia + MasakhaNER (see `docs/data_card.pdf` Section
  2.1). **This is the one actually used for training and for all
  fragmentation/compression/model results in this repo.**

Earlier in the project, `eval_holdout/` only had a `wo_wikipedia/` entry
while the training side used `wo_combined/` — a path-handling gotcha
flagged during Phase 2/3. That's since been resolved: both source folders
now have a matching `eval_holdout/` entry, so there's no more mismatch to
work around. The thing to still keep in mind when writing code that loops
over all 9 languages is simply to make sure you're pointing at
`wo_combined/`, not `wo_wikipedia/`.

## Reproducing from scratch

1. Run `src/data_pipeline/01_download_and_extract.sh` through
   `src/data_pipeline/05_convert_masakhaner_to_corpus.py` in order to rebuild the raw/cleaned
   corpora and train/eval splits for each language (Phase 1). Output lands
   under `DATA_ROOT/{code}_wikipedia/` (or `wo_wikipedia/` /
   `wo_combined/` for Wolof), with the held-out eval split written to the
   top-level `eval_holdout/` folder.
2. Run the Phase 2/3 notebook (`notebooks/tokenizer_training.ipynb`)
   to rebuild `tri_ai_archive/combined_corpus.txt` and both tokenizers --
   this reads each language's `train/corpus.txt`, not `cleaned/corpus.txt`.
3. Run the Phase 4 notebook to rebuild `tri_ai_archive/model_train_text.txt`
   / `model_eval_text.txt`, train the comparison models, and produce the
   `checkpoints_*/` folders.
4. Run the Phase 5 compression-efficiency step to rebuild
   `tri_ai_archive/compression_efficiency_results.json`.
