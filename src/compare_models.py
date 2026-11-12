"""Compare classifiers on synthetic pairs, holding out both applicants and grants.

Run: python -m src.compare_models --output results/model_comparison_v1
This experiment does not change the recommendation pipeline or use human review labels.
"""

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.ardook_extra_synthetic_data import DB_PATH, load_grants, load_matches, load_profiles
from src.paths import REPO_ROOT
from src.ranking_classifier import PairFeatures

LABELS = ["unrelated", "ineligible", "match"]
MODELS = ["all-MiniLM-L6-v2", "all-MiniLM-L12-v2", "all-mpnet-base-v2"]
SEED = 42


def assign_splits(texts):
    # Group identical input text even when its records have different IDs.
    unique = sorted(set(texts))
    train, rest = train_test_split(unique, test_size=0.4, random_state=SEED)
    validation, test = train_test_split(rest, test_size=0.5, random_state=SEED)
    return {text: split for split, group in
            [("train", train), ("validation", validation), ("test", test)] for text in group}


def load_partitions():
    profiles = load_profiles().set_index("id")
    grants = load_grants().set_index("source_row")
    org_text = {int(i): row.description + "\nApplicant facts: " + json.dumps(row.attributes, sort_keys=True)
                for i, row in profiles.iterrows()}
    grant_text = {int(i): "\n".join(str(row[f]) for f in
                  ("program", "type", "core_requirements", "amount_benefit"))
                  for i, row in grants.iterrows()}
    org_split, grant_split = assign_splits(org_text.values()), assign_splits(grant_text.values())
    partitions = {split: [] for split in ("train", "validation", "test")}
    for pair in load_matches().itertuples(index=False):
        org, grant = org_text[int(pair.profile_id)], grant_text[int(pair.source_row)]
        split = org_split[org]
        if split == grant_split[grant]:
            partitions[split].append({"profile_id": int(pair.profile_id),
                                      "source_row": int(pair.source_row),
                                      "organization": org, "grant": grant, "label": pair.label})
    for split, rows in partitions.items():
        if set(r["label"] for r in rows) != set(LABELS):
            raise ValueError(f"{split} must contain all three classes")
    return partitions


def metrics(truth, predicted):
    return {"accuracy": float(accuracy_score(truth, predicted)),
            "macro_f1": float(f1_score(truth, predicted, labels=range(3), average="macro", zero_division=0)),
            "per_class": classification_report(truth, predicted, labels=range(3),
                                               target_names=LABELS, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(truth, predicted, labels=range(3)).tolist()}


def embedding_features(texts, name):
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(f"sentence-transformers/{name}", device="cpu", local_files_only=True)
    unique = sorted({text for pairs in texts.values() for pair in pairs for text in pair})
    vectors = encoder.encode(unique, batch_size=32, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False)
    lookup = dict(zip(unique, vectors))
    features = {}
    for split, pairs in texts.items():
        org = np.stack([lookup[p[0]] for p in pairs])
        grant = np.stack([lookup[p[1]] for p in pairs])
        features[split] = np.concatenate([org, grant, org * grant, abs(org - grant)], axis=1)
    scaler = StandardScaler().fit(features["train"])
    return {s: scaler.transform(x).astype(np.float32) for s, x in features.items()}


def train_mlp(features, labels):
    torch.manual_seed(SEED)
    model = nn.Sequential(nn.Linear(features["train"].shape[1], 64), nn.ReLU(),
                          nn.Dropout(0.2), nn.Linear(64, len(LABELS)))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    counts = np.bincount(labels["train"], minlength=3)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32))
    xs = {s: torch.from_numpy(x) for s, x in features.items()}
    loader = DataLoader(TensorDataset(xs["train"], torch.tensor(labels["train"])),
                        batch_size=128, shuffle=True)
    best_score, best_state, best_epoch = -1, None, None
    history = []
    for epoch in range(1, 21):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            predicted = model(xs["validation"]).argmax(1).numpy()
        score = metrics(labels["validation"], predicted)["macro_f1"]
        history.append({"epoch": epoch, "validation_macro_f1": score})
        if score > best_score:
            best_score, best_state, best_epoch = score, deepcopy(model.state_dict()), epoch
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = {s: model(xs[s]).argmax(1).numpy() for s in ("validation", "test")}
    return predictions, {"best_epoch": best_epoch, "history": history,
                         "hidden_units": 64, "dropout": 0.2, "batch_size": 128, "learning_rate": 0.001}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results/model_comparison_v1")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists; choose a new --output directory")
    torch.set_num_threads(4)
    partitions = load_partitions()
    texts = {s: [(r["organization"], r["grant"]) for r in rows] for s, rows in partitions.items()}
    labels = {s: np.array([LABELS.index(r["label"]) for r in rows]) for s, rows in partitions.items()}
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "dataset_sha256": hashlib.sha256(DB_PATH.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {p: importlib.metadata.version(p) for p in
                     ("torch", "scikit-learn", "sentence-transformers", "numpy")},
        "label_order": LABELS,
        "label_meaning": {"unrelated": "No shared generated focus, regardless of eligibility",
                          "ineligible": "Shared focus but a known generated rule is violated",
                          "match": "Shared focus and no known rule violation; missing facts are tolerated"},
        "limitations": ["Rule-generated synthetic labels, not human judgments or real-world accuracy.",
                        "Deadlines are ignored by the target labels; match does not establish eligibility.",
                        "Applicant and grant records are held out, but generator wording/templates are shared.",
                        "One seeded split; pair observations share records and are not independent.",
                        "Classification metrics are not recommendation Precision@5.",
                        "Frozen pretrained encoders; only classification heads are trained."],
        "splits": {s: {"pairs": len(rows), "class_counts": dict(Counter(r["label"] for r in rows)),
                       "profile_ids": sorted({r["profile_id"] for r in rows}),
                       "source_rows": sorted({r["source_row"] for r in rows})}
                   for s, rows in partitions.items()},
        "models": {},
    }
    report["excluded_cross_split_pairs"] = len(load_matches()) - sum(len(r) for r in partitions.values())
    predictions_by_model = {}

    def record(name, predictions, elapsed, settings):
        report["models"][name] = {s: metrics(labels[s], predictions[s]) for s in ("validation", "test")}
        report["models"][name].update(seconds=elapsed, settings=settings)
        predictions_by_model[name] = predictions["test"]
        print(f"{name}: validation macro-F1={report['models'][name]['validation']['macro_f1']:.3f}", flush=True)

    majority = int(np.bincount(labels["train"]).argmax())
    record("Majority class", {s: np.full(len(labels[s]), majority) for s in ("validation", "test")},
           0.0, {"predicted_label": LABELS[majority]})

    def logistic(name, features, start):
        model = LogisticRegression(C=1, class_weight="balanced", max_iter=2000, random_state=SEED)
        model.fit(features["train"], labels["train"])
        record(name, {s: model.predict(features[s]) for s in ("validation", "test")},
               time.perf_counter() - start, {"C": 1, "class_weight": "balanced", "max_iter": 2000})

    start = time.perf_counter()
    vectorizer = PairFeatures().fit(texts["train"])
    lexical = {s: vectorizer.transform(t) for s, t in texts.items()}
    logistic("TF-IDF + logistic regression", lexical, start)
    for name in MODELS:
        print(f"Encoding with {name} (local cache, CPU)...", flush=True)
        start = time.perf_counter()
        features = embedding_features(texts, name)
        encoding_seconds = time.perf_counter() - start
        logistic(f"{name} + logistic regression", features, start)
        if name == "all-MiniLM-L6-v2":
            start = time.perf_counter()
            predictions, settings = train_mlp(features, labels)
            record(f"{name} + PyTorch MLP", predictions,
                   encoding_seconds + time.perf_counter() - start, settings)

    # No test result is used to choose hyperparameters, epochs, or the winning model.
    best = max(report["models"], key=lambda name: report["models"][name]["validation"]["macro_f1"])
    report["selected_by_validation"] = best
    test_rows = [{**row, "predictions": {name: LABELS[int(pred[i])] for name, pred in predictions_by_model.items()}}
                 for i, row in enumerate(partitions["test"])]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "test_predictions.json").write_text(json.dumps(test_rows, indent=2) + "\n")
    print(f"Selected on validation: {best}\nSaved report and test predictions to {args.output}")


if __name__ == "__main__":
    main()
