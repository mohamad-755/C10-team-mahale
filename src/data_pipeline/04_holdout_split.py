"""
Step 5: Hold out an evaluation split, BEFORE this data goes anywhere near
tokenizer/model training.

Per the project's collection plan: "Hold out the evaluation split before
anything else touches the data, and store it in a location the Model
Development and Deployment Lead cannot accidentally include in training."

This script:
1. Reads the final cleaned corpus.txt for a language
2. Randomly holds out a fixed percentage of documents (default 5%) as eval
3. Writes train.txt and eval.txt into SEPARATE top-level folders (not nested
   under the same "cleaned" folder as training data) specifically so eval
   data doesn't get accidentally swept up by a later "just grab everything
   in cleaned/" step
4. Records the exact document count and a content hash of the eval set, so
   later on anyone can verify the model was never trained on this exact split

Usage:
    python3 04_holdout_split.py --input ./data/sw_wikipedia/cleaned/corpus.txt \\
        --train-output ./data/sw_wikipedia/train/corpus.txt \\
        --eval-output ./eval_holdout/sw_wikipedia/eval.txt \\
        --eval-fraction 0.05
"""

import argparse
import hashlib
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the final cleaned corpus.txt")
    parser.add_argument("--train-output", required=True, help="Where to write the training split")
    parser.add_argument("--eval-output", required=True, help="Where to write the held-out eval split")
    parser.add_argument("--eval-fraction", type=float, default=0.05,
                         help="Fraction of documents to hold out for eval (default 5%%)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed, for reproducibility")
    args = parser.parse_args()

    input_path = Path(args.input)
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]

    total = len(lines)
    eval_count = max(1, int(total * args.eval_fraction))

    random.seed(args.seed)
    indices = list(range(total))
    random.shuffle(indices)
    eval_indices = set(indices[:eval_count])

    train_lines = [line for i, line in enumerate(lines) if i not in eval_indices]
    eval_lines = [line for i, line in enumerate(lines) if i in eval_indices]

    train_path = Path(args.train_output)
    eval_path = Path(args.eval_output)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    with open(train_path, "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line + "\n")

    with open(eval_path, "w", encoding="utf-8") as f:
        for line in eval_lines:
            f.write(line + "\n")

    # Content hash of the eval set - lets anyone later verify this exact
    # eval split was never touched during training, without re-reading the
    # whole file line by line
    eval_content = "\n".join(eval_lines).encode("utf-8")
    eval_hash = hashlib.sha256(eval_content).hexdigest()

    manifest_path = eval_path.parent / "eval_manifest.md"
    manifest_path.write_text(
        f"# Eval Split Manifest\n\n"
        f"- Source corpus: {input_path}\n"
        f"- Total documents in source: {total}\n"
        f"- Held-out eval documents: {eval_count} ({args.eval_fraction*100:.1f}%)\n"
        f"- Training documents remaining: {len(train_lines)}\n"
        f"- Random seed used: {args.seed}\n"
        f"- SHA-256 of eval set content: {eval_hash}\n\n"
        f"**This eval set must never be included in tokenizer or model training.** "
        f"Verify against the hash above if there is ever doubt about which split "
        f"a given corpus file represents.\n",
        encoding="utf-8",
    )

    print(f"Total documents: {total}")
    print(f"Train: {len(train_lines)} -> {train_path}")
    print(f"Eval:  {len(eval_lines)} -> {eval_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Eval content hash: {eval_hash}")


if __name__ == "__main__":
    main()
