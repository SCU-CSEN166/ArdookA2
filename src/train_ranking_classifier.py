"""Validate human reviews, then optionally train an offline relevance classifier."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import sklearn

from src.paths import REPO_ROOT
from src.ranking_classifier import load_reviews, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=REPO_ROOT / "data/ranking_review/pairs.jsonl")
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "models/relevance_v1")
    args = parser.parse_args()
    try:
        rows, counts = load_reviews(args.pairs, args.reviews)
        print(json.dumps({"review_counts": counts, "trainable_pairs": len(rows)}, indent=2))
        if args.validate_only:
            return
        model, report = train(rows)
        report.update(review_counts=counts, sklearn_version=sklearn.__version__,
                      pairs_sha256=hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
                      reviews_sha256=hashlib.sha256(args.reviews.read_bytes()).hexdigest())
        args.output.mkdir(parents=True, exist_ok=False)
        joblib.dump(model, args.output / "model.joblib")
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved offline classifier and evaluation report to {args.output}")
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
