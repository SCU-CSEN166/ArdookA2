"""PyTorch MLP classifier whose input embeddings come from a LoRA-adapted MiniLM.

Run: python -m src.lora_mlp --output results/lora_mlp_encoder_v1
Stage 1 trains the usual MLP on frozen embeddings (same as compare_models). Stage 2 starts from those
exact MLP weights, adds LoRA to the encoder and trains both. Epoch 0 is the stage-1 model, so the
result can never fall below it on validation. Test is scored once after selection.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
from peft import LoraConfig, get_peft_model
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer

from src.compare_models import LABELS, SEED, load_partitions, metrics
from src.paths import REPO_ROOT
from src.train_pytorch import fit

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def encode(model, tokenizer, texts):
    enc = tokenizer(texts, truncation=True, max_length=128, padding=True, return_tensors="pt")
    hidden = model(**enc).last_hidden_state
    mask = enc["attention_mask"].unsqueeze(-1).float()
    return nn.functional.normalize((hidden * mask).sum(1) / mask.sum(1), dim=-1)  # mean pool, as sentence-transformers


def pair_features(org, grant):
    return torch.cat([org, grant, org * grant, (org - grant).abs()], dim=1)


class Classifier(nn.Module):
    def __init__(self, encoder, tokenizer, mean, scale, mlp):
        super().__init__()
        self.encoder, self.tokenizer, self.mlp = encoder, tokenizer, mlp
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)

    def forward(self, orgs, grants):
        # Encode each distinct text once per batch.
        texts = sorted(set(orgs) | set(grants))
        emb = encode(self.encoder, self.tokenizer, texts)
        index = {t: i for i, t in enumerate(texts)}
        o = emb[[index[t] for t in orgs]]
        g = emb[[index[t] for t in grants]]
        return self.mlp((pair_features(o, g) - self.mean) / self.scale)


def predict(clf, rows, size=128):
    clf.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), size):
            chunk = rows[i:i + size]
            out.append(clf([r["organization"] for r in chunk], [r["grant"] for r in chunk]).argmax(1).numpy())
    return np.concatenate(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results/lora_mlp_encoder_v1")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists; choose a new --output directory")
    torch.manual_seed(SEED)
    torch.set_num_threads(8)
    rng = np.random.default_rng(SEED)
    partitions = load_partitions()
    labels = {s: np.array([LABELS.index(r["label"]) for r in rows]) for s, rows in partitions.items()}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    encoder = AutoModel.from_pretrained(MODEL, local_files_only=True).eval()

    # Stage 1: frozen embeddings -> scaled features -> the existing MLP recipe.
    unique = sorted({t for rows in partitions.values() for r in rows for t in (r["organization"], r["grant"])})
    with torch.no_grad():
        lookup = {}
        for i in range(0, len(unique), 64):
            for t, v in zip(unique[i:i + 64], encode(encoder, tokenizer, unique[i:i + 64])):
                lookup[t] = v
    raw = {s: pair_features(torch.stack([lookup[r["organization"]] for r in rows]),
                            torch.stack([lookup[r["grant"]] for r in rows])).numpy()
           for s, rows in partitions.items()}
    scaler = StandardScaler().fit(raw["train"])
    xs = {s: torch.from_numpy(scaler.transform(x).astype(np.float32)) for s, x in raw.items()}
    datasets = {s: TensorDataset(xs[s], torch.tensor(labels[s])) for s in xs}
    counts = np.bincount(labels["train"], minlength=3)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32))
    mlp = nn.Sequential(nn.Linear(xs["train"].shape[1], 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 3))
    base_score, base_state, _, _ = fit(mlp, DataLoader(datasets["train"], batch_size=128, shuffle=True),
                                       datasets["validation"], loss_fn, epochs=20, learning_rate=1e-3, label="base")
    mlp.load_state_dict(base_state)
    print(f"Stage 1 MLP: validation macro-F1={base_score:.3f}", flush=True)

    # Stage 2: LoRA on the encoder; the MLP starts from its stage-1 weights and keeps training.
    peft_encoder = get_peft_model(encoder, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        target_modules=["query", "key", "value", "dense"]))
    peft_encoder.print_trainable_parameters()
    clf = Classifier(peft_encoder, tokenizer, torch.from_numpy(scaler.mean_.astype(np.float32)),
                     torch.from_numpy(scaler.scale_.astype(np.float32)), mlp)
    params = [p for p in clf.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)

    def trainable_state():
        return {k: v.detach().clone() for k, v in clf.state_dict().items() if "lora_" in k or k.startswith("mlp.")}

    # Epoch 0 is the stage-1 model: scored through the same pipeline (LoRA B starts at zero).
    base_val = metrics(labels["validation"], predict(clf, partitions["validation"]))["macro_f1"]
    best_score, best_epoch, best_state = base_val, 0, trainable_state()
    history = [{"epoch": 0, "validation_macro_f1": base_val}]
    print(f"Epoch 0 (stage-1 weights through the adapter pipeline): validation macro-F1={base_val:.3f}", flush=True)
    start = time.perf_counter()
    train_rows = partitions["train"]
    for epoch in range(1, args.epochs + 1):
        clf.train()
        order = rng.permutation(len(train_rows))
        for step, i in enumerate(range(0, len(order), 64)):
            chunk = [train_rows[j] for j in order[i:i + 64]]
            y = torch.tensor(labels["train"][order[i:i + 64]])
            optimizer.zero_grad()
            loss_fn(clf([r["organization"] for r in chunk], [r["grant"] for r in chunk]), y).backward()
            optimizer.step()
            if step % 40 == 0:
                print(f"epoch {epoch} step {step} {time.perf_counter() - start:.0f}s", flush=True)
        score = metrics(labels["validation"], predict(clf, partitions["validation"]))["macro_f1"]
        history.append({"epoch": epoch, "validation_macro_f1": score})
        print(f"epoch {epoch}: validation macro-F1={score:.3f}", flush=True)
        if score > best_score:
            best_score, best_epoch, best_state = score, epoch, trainable_state()
    clf.load_state_dict(best_state, strict=False)
    report = {"seed": SEED, "rank": args.rank, "learning_rate": args.learning_rate, "best_epoch": best_epoch,
              "stage1_validation_macro_f1": base_score, "history": history,
              "seconds": time.perf_counter() - start}
    for split in ("validation", "test"):
        report[split] = metrics(labels[split], predict(clf, partitions[split]))
    args.output.mkdir(parents=True, exist_ok=False)
    torch.save({"lora_and_mlp": best_state}, args.output / "model.pt")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"best epoch {best_epoch}: validation macro-F1={report['validation']['macro_f1']:.3f} "
          f"test macro-F1={report['test']['macro_f1']:.3f} test accuracy={report['test']['accuracy']:.3f}")


if __name__ == "__main__":
    main()
