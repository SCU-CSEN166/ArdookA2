"""Freeze a reproducible, unlabeled review sample from the local tracker."""

import argparse
import hashlib
import json
import random
from datetime import date
from pathlib import Path

from src.ingest import load_grants
from src.lexical import BM25
from src.paths import DEFAULT_WORKBOOK, REPO_ROOT
from src.pipeline import text

REVIEW_DIR = REPO_ROOT / "data/ranking_review"


def build_pairs(profiles, grants, as_of, source_sha256):
    fields = ("program", "type", "core_requirements", "amount_benefit",
              "deadline", "current_status", "official_source")
    records = grants.to_dict("records")
    documents = [" ".join(text(g[f]) for f in ("program", "type", "core_requirements"))
                 for g in records]
    bm25 = BM25(documents)
    pairs = []
    for profile in profiles:
        scores = bm25.normalized(profile["description"])
        ranked = sorted(range(len(records)), key=lambda i: (-scores[i], records[i]["source_row"]))
        selected = ranked[:10]
        rng = random.Random(f"ardook-review-v1:{profile['profile_id']}")
        selected += rng.sample(ranked[10:], min(10, len(ranked[10:])))
        rng.shuffle(selected)  # Do not show reviewers ranking order or predicted labels.
        for index in selected:
            grant = records[index]
            pairs.append({
                "pair_id": f"{profile['profile_id']}-row{grant['source_row']}",
                "profile_id": profile["profile_id"], "split": profile["split"],
                "profile_description": profile["description"],
                "profile_attributes": profile["attributes"], "profile_origin": "fictional",
                "as_of": as_of.isoformat(), "source_workbook": "Ardook_data.xlsx",
                "source_sheet": "Grant Tracker", "source_row": int(grant["source_row"]),
                "source_sha256": source_sha256,
                **{f: text(grant[f]) for f in fields},
            })
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, default=REVIEW_DIR / "pairs.jsonl")
    args = parser.parse_args()
    profiles = json.loads((REVIEW_DIR / "profiles.json").read_text())
    pairs = build_pairs(profiles, load_grants(), args.as_of,
                        hashlib.sha256(DEFAULT_WORKBOOK.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # A frozen review set must not silently change underneath completed labels.
    with args.output.open("x") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Prepared {len(pairs)} unlabeled pairs in {args.output}")


if __name__ == "__main__":
    main()
