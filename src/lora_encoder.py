"""Fine-tune all-MiniLM-L6-v2 with a LoRA adapter as a pair classifier (organization, grant) -> label.

Run: python -m src.lora_encoder --output results/lora_encoder_v1
Same synthetic split and seed as compare_models; the epoch is chosen on validation, test is scored once.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from peft import LoraConfig, get_peft_model
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.compare_models import LABELS, SEED, load_partitions, metrics
from src.paths import REPO_ROOT

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def batches(tokenizer, rows, labels, size, shuffle, rng):
    order = rng.permutation(len(rows)) if shuffle else np.arange(len(rows))
    for i in range(0, len(rows), size):
        idx = order[i:i + size]
        enc = tokenizer([rows[j]["organization"] for j in idx], [rows[j]["grant"] for j in idx],
                        truncation=True, max_length=192, padding=True, return_tensors="pt")
        yield enc, torch.tensor(labels[idx])


def predict(model, tokenizer, rows, labels):
    model.eval()
    out = []
    with torch.no_grad():
        for enc, _ in batches(tokenizer, rows, labels, 64, False, None):
            out.append(model(**enc).logits.argmax(1).numpy())
    return np.concatenate(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results/lora_encoder_v1")
    parser.add_argument("--epochs", type=int, default=3)
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
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=3, local_files_only=True)
    model = get_peft_model(model, LoraConfig(
        task_type="SEQ_CLS", r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        target_modules=["query", "key", "value", "dense"]))
    model.print_trainable_parameters()
    counts = np.bincount(labels["train"], minlength=3)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (3 * counts), dtype=torch.float32))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    best_score, best_state, best_epoch, history = -1.0, None, 0, []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for step, (enc, y) in enumerate(batches(tokenizer, partitions["train"], labels["train"], 32, True, rng)):
            optimizer.zero_grad()
            loss_fn(model(**enc).logits, y).backward()
            optimizer.step()
            if step % 50 == 0:
                print(f"epoch {epoch} step {step} {time.perf_counter() - start:.0f}s", flush=True)
        score = metrics(labels["validation"], predict(model, tokenizer, partitions["validation"],
                                                      labels["validation"]))["macro_f1"]
        history.append({"epoch": epoch, "validation_macro_f1": score})
        print(f"epoch {epoch}: validation macro-F1={score:.3f}", flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items() if v.requires_grad or "lora_" in k
                          or "classifier" in k}
    model.load_state_dict(best_state, strict=False)
    report = {"seed": SEED, "rank": args.rank, "learning_rate": args.learning_rate, "best_epoch": best_epoch,
              "history": history, "seconds": time.perf_counter() - start}
    for split in ("validation", "test"):
        report[split] = metrics(labels[split], predict(model, tokenizer, partitions[split], labels[split]))
    args.output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(args.output / "adapter")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"validation macro-F1={report['validation']['macro_f1']:.3f} "
          f"test macro-F1={report['test']['macro_f1']:.3f} test accuracy={report['test']['accuracy']:.3f}")


if __name__ == "__main__":
    main()
