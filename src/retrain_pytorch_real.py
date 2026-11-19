import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

from src.compare_models import LABELS, SEED, embedding_features, load_partitions, train_mlp
from src.ingest import load_grants

ARDOOK_DATA = (
    "Ardook is an early-stage, woman-led U.S. startup building AI, physical-AI and data "
    "products with a social-impact mission.\nApplicant facts: "
    + json.dumps({"entity_type": "for-profit", "region": "US", "woman_led": "yes"}, sort_keys=True)
)
GRANT_FIELDS = ("program", "type", "core_requirements", "amount_benefit")
INELIGIBLE = re.compile(r"only if nonprofit|low practical fit", re.I)


def fit_to_label(fit: str) -> str:
    fit = str(fit)
    if INELIGIBLE.search(fit):
        return "ineligible"
    if re.match(r"(excellent|strong)", fit, re.I):
        return "match"
    return "unrelated"


def grant_text(row) -> str:
    return "\n".join("" if pd.isna(row[f]) else str(row[f]) for f in GRANT_FIELDS)


def evaluate(truth: dict, predicted: dict) -> dict:
    tp = sum(predicted[k] and truth[k] for k in truth)
    fp = sum(predicted[k] and not truth[k] for k in truth)
    fn = sum(not predicted[k] and truth[k] for k in truth)
    tn = len(truth) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": (tp + tn) / len(truth), "precision": precision, "recall": recall,
            "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True, help="TA labels: grant_id,label")
    parser.add_argument("--oversample", type=int, nargs="+", default=[1, 5, 20],
                        help="Repeat each real training pair N times (one run per value).")
    parser.add_argument("--output", type=Path, default=Path("results/pytorch_retrain_real"))
    args = parser.parse_args()
    torch.set_num_threads(4)

    synthetic = load_partitions()
    real = load_grants()
    real_rows = [{"organization": ARDOOK_DATA, "grant": grant_text(r), "label": fit_to_label(r["fit"]),
                  "program": r["program"]} for _, r in real.iterrows()]
    real_train, real_val = train_test_split(real_rows, test_size=0.25, random_state=SEED,
                                            stratify=[r["label"] == "match" for r in real_rows])

    hidden = pd.read_excel(args.hidden, keep_default_na=False)
    hidden.columns = hidden.columns.str.strip()
    hidden = hidden.rename(columns={"Program": "program", "Type": "type",
                                    "Core Requirements": "core_requirements",
                                    "Amount / Benefit": "amount_benefit"})
    hidden_pairs = [(ARDOOK_DATA, grant_text(r)) for _, r in hidden.iterrows()]
    truth = {int(r["grant_id"]): r["label"] in ("Relevant", "Good")
             for r in csv.DictReader(args.labels.open())}
    ids = hidden["Grant ID"].astype(int).tolist()

    def pairs(rows):
        return [(r["organization"], r["grant"]) for r in rows]

    def labels_of(rows):
        return np.array([LABELS.index(r["label"]) for r in rows])

    runs = {"synthetic only (Week 7 baseline)": (synthetic["train"], synthetic["validation"] + real_val)}
    for n in args.oversample:
        runs[f"synthetic + real (x{n})"] = (synthetic["train"] + real_train * n,
                                            synthetic["validation"] + real_val)
    results, predictions_out = [], {}
    for name, (train_rows, val_rows) in runs.items():
        print(f"Training: {name} ({len(train_rows)} training pairs)", flush=True)
        texts = {"train": pairs(train_rows), "validation": pairs(val_rows), "test": hidden_pairs}
        features = embedding_features(texts, "all-MiniLM-L6-v2")
        predictions, settings = train_mlp(features, {"train": labels_of(train_rows),
                                                      "validation": labels_of(val_rows)})
        val_classes = [LABELS[int(p)] for p in predictions["validation"][-len(real_val):]]
        real_val_metrics = evaluate({i: r["label"] == "match" for i, r in enumerate(real_val)},
                                    {i: c == "match" for i, c in enumerate(val_classes)})
        classes = [LABELS[int(p)] for p in predictions["test"]]
        predicted = {gid: c == "match" for gid, c in zip(ids, classes)}
        metrics = evaluate(truth, predicted)
        results.append({"system": name, "real_val_f1": round(real_val_metrics["f1"], 3),
                        "real_val_accuracy": round(real_val_metrics["accuracy"], 3), **{k: round(v, 3) if isinstance(v, float) else v
                                           for k, v in metrics.items()},
                        "best_epoch": settings["best_epoch"]})
        predictions_out[name] = classes

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with (args.output / "hidden_test_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["grant_id", "program", "ta_label", *predictions_out])
        for i, gid in enumerate(ids):
            writer.writerow([gid, hidden.loc[i, "program"], "Relevant" if truth[gid] else "Not relevant",
                             *(predictions_out[n][i] for n in predictions_out)])
    with (args.output / "real_training_labels.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["program", "fit", "label", "split"])
        fits = dict(zip(real["program"], real["fit"]))
        for split, rows in (("train", real_train), ("validation", real_val)):
            writer.writerows([r["program"], fits[r["program"]], r["label"], split] for r in rows)
    print(pd.DataFrame(results).to_string(index=False))
    print(f"\nWrote {args.output}/")


if __name__ == "__main__":
    main()
