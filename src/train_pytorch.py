"""Train a small PyTorch relevance classifier on completed human reviews.

Uses the existing TF-IDF pair features so this starter needs no model downloads.
Eligibility remains the responsibility of the deterministic recommendation checks.
"""

import argparse
from copy import deepcopy
import json
import re
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.paths import REPO_ROOT
from src.ranking_classifier import LABELS, PairFeatures, load_reviews, pair_texts


# Keywords seen in tracker-style requirement text; lowercase substrings.
ENTITY_CLAUSES = {"nonprofit": ("nonprofit", "501(c)(3)"), "for-profit": ("for-profit",),
                  "university": ("university", "academic institution")}
REGION_CLAUSES = {"US": ("u.s.",), "EU": ("horizon europe",), "UK": ("uk-registered", "uk-based"),
                  "India": ("india",), "Africa": ("sub-saharan africa",), "Armenia": ("armenia",)}
WOMAN_CLAUSES = ("woman-led", "woman founder", "women-owned", "female entrepreneur")


def compat_features(org, grant):
    """Applicant facts, grant requirements, and their pairings. No rule says what counts as a mismatch.

    The model learns which pairings are compatible (e.g. nonprofit x "for-profit company").
    Missing facts give all zeros. shortcut: regex over this wording only, upgrade to the
    extractor's structured fields for real grants.
    """
    match = re.search(r"Applicant facts: (\{.*\})", org)
    facts = json.loads(match.group(1)) if match else {}
    text = grant.lower()
    applicant = ([facts.get("entity_type") == e for e in ENTITY_CLAUSES]
                 + [facts.get("region") == r for r in REGION_CLAUSES]
                 + [facts.get("woman_led") == "no"])
    required = ([any(k in text for k in ks) for ks in ENTITY_CLAUSES.values()]
                + [any(k in text for k in ks) for ks in REGION_CLAUSES.values()]
                + [any(k in text for k in WOMAN_CLAUSES)])
    limit = re.search(r"under \$(\d+)(k|m)", text)
    cap = int(limit[1]) * (1_000 if limit[2] == "k" else 1_000_000) if limit else 0
    numbers = [np.log10(facts.get("annual_revenue", 0) + 1) / 7, np.log10(cap + 1) / 7]
    return applicant + required + numbers + [a and r for a in applicant for r in required]


def build_model(input_dim, hidden_dim=64, dropout=0.2):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, len(LABELS)),
    )


class LoRALinear(nn.Module):
    """Frozen Linear plus a trainable low-rank update: y = Wx + (alpha/r) * B(Ax)."""

    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base, self.scale = base, alpha / rank
        self.lora_a = nn.Parameter(torch.randn(rank, base.in_features) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))  # zero init: starts equal to base

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.lora_a.T) @ self.lora_b.T

    def merge(self):
        with torch.no_grad():
            self.base.weight += self.scale * self.lora_b @ self.lora_a
        return self.base


def add_lora(model, rank, alpha):
    """Freeze every base weight and wrap each Linear layer with a LoRA adapter, in place."""
    for p in model.parameters():
        p.requires_grad = False
    for i, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            model[i] = LoRALinear(layer, rank, alpha)
    return model


def merge_lora(model):
    for i, layer in enumerate(model):
        if isinstance(layer, LoRALinear):
            model[i] = layer.merge()
    return model


def macro_f1(y, predictions):
    return float(f1_score(y, predictions, labels=range(len(LABELS)), average="macro", zero_division=0))


def tune_offsets(logits, y, grid=np.linspace(-2, 2, 17), rounds=2):
    """Per-class logit offsets maximizing macro-F1 (coordinate search). Fit on validation only.

    shortcut: coordinate search can overfit a small validation set, upgrade to CV if splits stay tiny.
    """
    offsets = np.zeros(len(LABELS))
    best = macro_f1(y, (logits + offsets).argmax(1))
    for _ in range(rounds):
        for c in range(len(LABELS)):
            for g in grid:
                trial = offsets.copy()
                trial[c] = g
                score = macro_f1(y, (logits + trial).argmax(1))
                if score > best + 1e-9:  # strict: ties keep the smaller change
                    best, offsets = score, trial
    return offsets, best


def fit(model, loader, val, loss_fn, *, epochs, learning_rate, label, weight_decay=1e-4, patience=8):
    """Train trainable params; return (best validation macro-F1, best state, best epoch, history)."""
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=learning_rate, weight_decay=weight_decay)
    best_score, best_state, best_epoch, history = -1.0, None, 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss_fn(model(x), y).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            predictions = model(val.tensors[0]).argmax(dim=1).numpy()
        score = macro_f1(val.tensors[1].numpy(), predictions)
        history.append({"epoch": epoch, "validation_macro_f1": score})
        print(f"[{label}] epoch {epoch:02d}/{epochs}: validation macro-F1={score:.3f}")
        if score > best_score:
            best_score, best_epoch, best_state = score, epoch, deepcopy(model.state_dict())
        elif epoch - best_epoch >= patience:
            break
    return best_score, best_state, best_epoch, history


def train(rows, *, epochs=20, batch_size=16, learning_rate=1e-3, seed=42,
          lora_rank=8, lora_alpha=16, lora_epochs=20, lora_learning_rate=1e-3,
          hidden_dim=128, dropout=0.3, weight_decay=1e-4):
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
        pairs = pair_texts(subset)
        x = np.hstack([features.transform(pairs).toarray(),
                       np.array([compat_features(*pair) for pair in pairs], dtype=float)]).astype(np.float32)
        y = [LABELS.index(r["label"]) for r in subset]
        datasets[split] = TensorDataset(torch.from_numpy(x), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True)
    input_dim = datasets["train"].tensors[0].shape[1]
    model = build_model(input_dim, hidden_dim, dropout)
    weights = torch.tensor(counts.sum() / (len(LABELS) * counts), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    val = datasets["validation"]

    # Stage 1: train the base MLP. Stage 2: freeze it and train only LoRA adapters on top.
    # Keep the adapter only if it beats the base on validation; the test set is never used to choose.
    base_score, base_state, base_epoch, history = fit(
        model, loader, val, loss_fn, epochs=epochs, learning_rate=learning_rate, label="base",
        weight_decay=weight_decay)
    model.load_state_dict(base_state)
    best_score, best_epoch, use_lora, lora_history, lora_state_dict = base_score, base_epoch, False, [], {}
    if lora_rank > 0:
        add_lora(model, lora_rank, lora_alpha)
        lora_score, lora_state, lora_epoch, lora_history = fit(
            model, loader, val, loss_fn, epochs=lora_epochs, learning_rate=lora_learning_rate, label="lora",
            weight_decay=weight_decay)
        if lora_score > base_score:
            model.load_state_dict(lora_state)
            best_score, best_epoch, use_lora = lora_score, lora_epoch, True
            lora_state_dict = {k: v for k, v in lora_state.items() if "lora_" in k}
            merge_lora(model)  # fold adapters into the weights: the checkpoint stays a plain MLP
        else:
            model = build_model(input_dim, hidden_dim, dropout)
            model.load_state_dict(base_state)
    model.eval()
    with torch.no_grad():
        val_logits = model(val.tensors[0]).numpy()
    offsets, tuned_score = tune_offsets(val_logits, val.tensors[1].numpy())
    report = {"classes": LABELS, "best_epoch": best_epoch, "history": history,
              "seed": seed, "epochs": epochs, "base_validation_macro_f1": base_score,
              "lora": {"used": use_lora, "rank": lora_rank, "alpha": lora_alpha, "history": lora_history},
              "best_validation_macro_f1": best_score, "batch_size": batch_size,
              "learning_rate": learning_rate, "compat_features": True, "hidden_dim": hidden_dim, "dropout": dropout,
              "weight_decay": weight_decay, "class_offsets": offsets.tolist(),
              "tuned_validation_macro_f1": tuned_score, "torch_version": str(torch.__version__)}
    with torch.no_grad():
        for split in ("validation", "test"):
            x, y = datasets[split].tensors
            predictions = (model(x).numpy() + offsets).argmax(1)
            report[split] = classification_report(
                y.numpy(), predictions, labels=range(len(LABELS)), target_names=LABELS,
                output_dict=True, zero_division=0,
            )
    checkpoint = {"state_dict": model.state_dict(), "input_dim": input_dim,
                  "hidden_dim": hidden_dim, "dropout": dropout, "labels": LABELS,
                  "class_offsets": offsets.tolist(),  # add to logits before argmax
                  "lora_state_dict": lora_state_dict}  # adapters are already merged into state_dict
    return checkpoint, features, report


def save_model(output, checkpoint, features, report):
    output.mkdir(parents=True, exist_ok=False)
    torch.save(checkpoint, output / "model.pt")
    joblib.dump(features, output / "features.joblib")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def load_model(directory):
    checkpoint = torch.load(Path(directory) / "model.pt", weights_only=False)
    features = joblib.load(Path(directory) / "features.joblib")
    model = build_model(checkpoint["input_dim"], checkpoint["hidden_dim"], checkpoint["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), features, np.array(checkpoint["class_offsets"])


def score_rows(directory, rows):
    """Offset-adjusted logits per row (profile_description + grant fields), no retraining."""
    model, features, offsets = load_model(directory)
    pairs = pair_texts(rows)
    x = np.hstack([features.transform(pairs).toarray(),
                   np.array([compat_features(*p) for p in pairs], dtype=float)]).astype(np.float32)
    with torch.no_grad():
        return model(torch.from_numpy(x)).numpy() + offsets


def predict_rows(directory, rows):
    return score_rows(directory, rows).argmax(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, default=REPO_ROOT / "data/ranking_review/pairs.jsonl")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "models/pytorch_v1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lora-rank", type=int, default=8, help="0 disables the LoRA stage")
    parser.add_argument("--lora-alpha", type=float, default=16)
    parser.add_argument("--lora-epochs", type=int, default=20)
    parser.add_argument("--lora-learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise FileExistsError(f"Output already exists: {args.output}; choose a new directory")
        rows, counts = load_reviews(args.pairs, args.reviews)
        checkpoint, features, report = train(
            rows, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
            lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_epochs=args.lora_epochs,
            lora_learning_rate=args.lora_learning_rate,
            hidden_dim=args.hidden_dim, dropout=args.dropout, weight_decay=args.weight_decay,
        )
        report["review_counts"] = counts
        save_model(args.output, checkpoint, features, report)
        print(f"Saved checkpoint, text features and evaluation report to {args.output}")
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
