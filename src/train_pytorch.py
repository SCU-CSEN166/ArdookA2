"""Train a small PyTorch relevance classifier on completed human reviews.

Uses the existing TF-IDF pair features so this starter needs no model downloads.
Eligibility remains the responsibility of the deterministic recommendation checks.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.paths import REPO_ROOT
from src.ranking_classifier import LABELS, PairFeatures, load_reviews, pair_texts


def build_model(input_dim, hidden_dim=64):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim, len(LABELS)),
    )


def train(rows, *, epochs=20, batch_size=16, learning_rate=1e-3, seed=42):
    if epochs < 1 or batch_size < 1 or not 0 < learning_rate < float("inf"):
        raise ValueError("epochs, batch_size and learning_rate must be positive and finite")
    torch.manual_seed(seed)
    partitions = {split: [r for r in rows if r["split"] == split]
                  for split in ("train", "validation", "test")}
    counts = np.array([sum(r["label"] == label for r in partitions["train"])
                       for label in LABELS])
    if counts.sum() < 30 or (counts < 5).any():
        raise ValueError("Need at least 30 human-labeled training pairs, including 5 per class")
    if any(len(partitions[split]) < 10 for split in ("validation", "test")):
        raise ValueError("Need at least 10 human-labeled pairs in each validation and test split")

    # Fit vocabulary on training text only. Dense tensors suit this small pilot.
    features = PairFeatures().fit(pair_texts(partitions["train"]))
    datasets = {}
    for split, subset in partitions.items():
        x = features.transform(pair_texts(subset)).toarray().astype(np.float32)
        y = [LABELS.index(r["label"]) for r in subset]
        datasets[split] = TensorDataset(torch.from_numpy(x), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
    input_dim = datasets["train"].tensors[0].shape[1]
    model = build_model(input_dim)
    weights = torch.tensor(counts.sum() / (len(LABELS) * counts), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_score, best_state, best_epoch = -1.0, None, 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            x, y = datasets["validation"].tensors
            predictions = model(x).argmax(dim=1).numpy()
        score = float(f1_score(y.numpy(), predictions, labels=range(len(LABELS)),
                               average="macro", zero_division=0))
        history.append({"epoch": epoch, "validation_macro_f1": score})
        print(f"Epoch {epoch:02d}/{epochs}: validation macro-F1={score:.3f}")
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = deepcopy(model.state_dict())

    # Select by validation performance, then evaluate the test set once.
    model.load_state_dict(best_state)
    model.eval()
    report = {"classes": LABELS, "best_epoch": best_epoch, "history": history,
              "seed": seed, "epochs": epochs, "batch_size": batch_size,
              "learning_rate": learning_rate, "torch_version": str(torch.__version__)}
    with torch.no_grad():
        for split in ("validation", "test"):
            x, y = datasets[split].tensors
            predictions = model(x).argmax(dim=1).numpy()
            report[split] = classification_report(
                y.numpy(), predictions, labels=range(len(LABELS)), target_names=LABELS,
                output_dict=True, zero_division=0,
            )
    checkpoint = {"state_dict": model.state_dict(), "input_dim": input_dim,
                  "hidden_dim": 64, "labels": LABELS}
    return checkpoint, features, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, default=REPO_ROOT / "data/ranking_review/pairs.jsonl")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "models/pytorch_v1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}; choose a new directory")
        rows, counts = load_reviews(args.pairs, args.reviews)
        checkpoint, features, report = train(
            rows, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
        )
        report["review_counts"] = counts
        args.output.mkdir(parents=True, exist_ok=False)
        torch.save(checkpoint, args.output / "model.pt")
        joblib.dump(features, args.output / "features.joblib")
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved checkpoint, text features and evaluation report to {args.output}")
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
