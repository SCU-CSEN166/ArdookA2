
import argparse
import csv
from pathlib import Path

POSITIVE = {"relevant", "good"}

def read_labels(path: Path) -> dict[int, bool]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["grant_id"]): row["label"].strip().lower() in POSITIVE
                for row in csv.DictReader(handle)}

def score(truth: dict[int, bool], predicted: dict[int, bool]) -> list:
    if set(truth) != set(predicted):
        raise ValueError(f"grant_id mismatch: {sorted(set(truth) ^ set(predicted))}")
    tp = sum(truth[g] and predicted[g] for g in truth)
    fp = sum(predicted[g] and not truth[g] for g in truth)
    fn = sum(truth[g] and not predicted[g] for g in truth)
    tn = len(truth) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return [f"{(tp + tn) / len(truth):.2f}", f"{precision:.3f}", f"{recall:.3f}", f"{f1:.3f}",
            tp, fp, fn, tn]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", type=Path)
    parser.add_argument("systems", nargs="+", metavar="NAME=CSV")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    truth = read_labels(args.labels)
    lines = ["system,accuracy,precision,recall,f1,tp,fp,fn,tn"]
    for spec in args.systems:
        name, _, path = spec.partition("=")
        predicted = read_labels(Path(path))
        values = score(truth, predicted)
        lines.append(",".join([name, *map(str, values)]))
        print(f"{name:<20} acc={values[0]} prec={values[1]} rec={values[2]} f1={values[3]}  "
              f"wrong={[g for g in truth if truth[g] != predicted[g]]}")
    if args.output:
        args.output.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote {args.output}")

if __name__ == "__main__":
    main()
