"""Measure whether a LoRA stage improves the frozen-MiniLM PyTorch MLP from compare_models.

Run: python -m src.lora_experiment --output results/lora_mlp_v1
Same synthetic split, features, seed and MLP as compare_models; LoRA rank is chosen on validation only.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.compare_models import LABELS, SEED, embedding_features, load_partitions, metrics
from src.paths import REPO_ROOT
from src.train_pytorch import add_lora, fit, merge_lora

RANKS = [4, 8, 16, 32]


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return model(x).argmax(1).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results/lora_mlp_v1")
    parser.add_argument("--lora-epochs", type=int, default=30)
    parser.add_argument("--lora-learning-rate", type=float, default=3e-3)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists; choose a new --output directory")
    torch.set_num_threads(4)
    partitions = load_partitions()
    texts = {s: [(r["organization"], r["grant"]) for r in rows] for s, rows in partitions.items()}
    labels = {s: np.array([LABELS.index(r["label"]) for r in rows]) for s, rows in partitions.items()}
    features = embedding_features(texts, "all-MiniLM-L6-v2")
    xs = {s: torch.from_numpy(x) for s, x in features.items()}
    datasets = {s: TensorDataset(xs[s], torch.tensor(labels[s])) for s in xs}
    counts = np.bincount(labels["train"], minlength=3)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32))

    def new_loader():
        return DataLoader(datasets["train"], batch_size=128, shuffle=True)

    torch.manual_seed(SEED)
    base = nn.Sequential(nn.Linear(xs["train"].shape[1], 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 3))
    base_score, base_state, base_epoch, _ = fit(base, new_loader(), datasets["validation"], loss_fn,
                                                epochs=20, learning_rate=1e-3, label="base")
    base.load_state_dict(base_state)
    results = {"base": {s: metrics(labels[s], predict(base, xs[s])) for s in ("validation", "test")}}
    print(f"Base MLP: validation macro-F1={base_score:.3f}", flush=True)

    best = None
    for rank in RANKS:
        torch.manual_seed(SEED)
        model = nn.Sequential(*deepcopy(list(base.children())))
        add_lora(model, rank, 2 * rank)
        score, state, epoch, history = fit(model, new_loader(), datasets["validation"], loss_fn,
                                           epochs=args.lora_epochs, learning_rate=args.lora_learning_rate,
                                           label=f"lora r={rank}")
        model.load_state_dict(state)
        merge_lora(model)
        results[f"lora_r{rank}"] = {"best_epoch": epoch, "history": history,
                                    **{s: metrics(labels[s], predict(model, xs[s])) for s in ("validation", "test")}}
        print(f"LoRA r={rank}: validation macro-F1={score:.3f}", flush=True)
        if best is None or score > best[0]:
            best = (score, rank)

    # Choose by validation only; fall back to the base model if no adapter beats it.
    chosen = f"lora_r{best[1]}" if best[0] > base_score else "base"
    report = {"seed": SEED, "ranks": RANKS, "lora_epochs": args.lora_epochs,
              "lora_learning_rate": args.lora_learning_rate, "selected_by_validation": chosen,
              "results": results}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for name in ("base", chosen):
        r = results[name]
        print(f"{name}: validation macro-F1={r['validation']['macro_f1']:.3f} "
              f"test macro-F1={r['test']['macro_f1']:.3f} test accuracy={r['test']['accuracy']:.3f}")


if __name__ == "__main__":
    main()
