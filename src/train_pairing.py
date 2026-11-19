"""Train the TF-IDF + PyTorch + pairing-table model on synthetic + tracker-style pairs and save it.

Reload with src.train_pytorch.predict_rows(directory, rows); no retraining needed.
Labels are rule-generated, not human reviews.
"""

import argparse
from pathlib import Path

from src.augment_pairs import augment_rows
from src.compare_models import load_partitions
from src.paths import REPO_ROOT
from src.ranking_classifier import LABELS
from src.train_pytorch import save_model, train

SYNTHETIC_TO_FIT = dict(zip(["unrelated", "ineligible", "match"], LABELS))


DEFAULT_DIR = REPO_ROOT / "models/pytorch_pairing_v1"


def build(output):
    rows = [{"split": split, "label": SYNTHETIC_TO_FIT[r["label"]], "profile_description": r["organization"],
             "program": r["grant"], "type": "", "core_requirements": "", "amount_benefit": ""}
            for split, part in load_partitions().items() for r in part]
    rows += augment_rows("train", 100) + augment_rows("validation", 150, 60, 60)
    checkpoint, features, report = train(rows, seed=42)
    save_model(output, checkpoint, features, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}; choose a new directory")
    build(args.output)
    print(f"Saved model, features and report to {args.output}")


if __name__ == "__main__":
    main()
