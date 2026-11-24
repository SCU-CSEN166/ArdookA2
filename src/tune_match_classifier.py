"""Match classifier v2: regularization grid + requirement filter, evaluated on the v1 splits.

Run: HF_HUB_OFFLINE=1 python -m src.tune_match_classifier --output results/ardook_match_v2
Follows results/ardook_match_v2_protocol.md (committed before this was run). Selection uses
synthetic validation and out-of-fold real dev only; synthetic test, real held-out and the hidden
grants are scored after selection and are post-exposure regression checks.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.ardook_extra_synthetic_data import DB_PATH
from src.eligibility_filter import FILTER_VERSION, check_requirements, explain, overall
from src.match_classifier import (ARDOOK_PROFILE, LABELS, MatchClassifier, Pair, TwoStageClassifier,
                                  grant_family, make_pair, real_pairs, synthetic_pairs)
from src.paths import DEFAULT_WORKBOOK, REPO_ROOT
from src.predict_match import read_grants
from src.train_match_classifier import binary_metrics, choose_threshold

V1_DIR = REPO_ROOT / "results/ardook_match_v1"
HIDDEN_WORKBOOK = REPO_ROOT / "week8_hidden_test/Ardook_hidden_test_grants.xlsx"
HIDDEN_LABELS = REPO_ROOT / "week8_hidden_test/final/expected_labels.csv"
SPEC = "twostage:bge"
REFERENCE = (0.1, 1.0, 0.10)  # v1: related C, conflict C, real weight share
GRID = list(itertools.product((0.03, 0.1, 0.3), (0.1, 0.3, 1.0), (0.05, 0.10, 0.20)))
FOR_PROFIT = {**ARDOOK_PROFILE["attributes"], "entity_type": "for-profit"}  # sensitivity only


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def config_name(config) -> str:
    return "related_C={} conflict_C={} real_share={}".format(*config)


def load_data() -> dict:
    synthetic, _ = synthetic_pairs()
    real = real_pairs()
    data = {"syn_train": [p for p in synthetic if p.split == "train"],
            "syn_val": [p for p in synthetic if p.split == "validation"],
            "syn_test": [p for p in synthetic if p.split == "test"],
            "real_dev": [p for p in real if p.split == "dev"],
            "real_test": [p for p in real if p.split == "test"]}
    manifest = json.loads((V1_DIR / "split_manifest.json").read_text())
    for split, pairs in data.items():
        if sorted({p.key for p in pairs}) != manifest[split]:
            raise AssertionError(f"{split} differs from the v1 split manifest")
    data["real_all"] = real
    return data


def hidden_pairs(attributes: dict) -> tuple[list[Pair], list[str]]:
    grants = read_grants(HIDDEN_WORKBOOK)
    labels = pd.read_csv(HIDDEN_LABELS).set_index("grant_id")["label"]
    pairs = [make_pair(ARDOOK_PROFILE["description"], attributes, row, key=str(row["grant_id"]))
             for row in grants.to_dict("records")]
    truth = ["match" if labels[int(p.key)] == "Relevant" else "not_match" for p in pairs]
    return pairs, truth


def train(config, syn_train, real_train=()) -> TwoStageClassifier:
    related_c, conflict_c, share = config
    model = TwoStageClassifier(SPEC, "logreg")
    model.RELATED_C, model.CONFLICT_C = related_c, conflict_c  # instance values are pickled
    weights = np.concatenate([np.ones(len(syn_train)),
                              np.full(len(real_train), share * len(syn_train) / max(1, len(real_train)))])
    return model.fit(list(syn_train) + list(real_train), sample_weight=weights)


def evaluate(config, data) -> dict:
    dev = data["real_dev"]
    groups = [grant_family(p.grant["program"]) for p in dev]
    oof = np.zeros(len(dev))
    for train_index, held_index in GroupKFold(n_splits=5).split(dev, groups=groups):
        fold = train(config, data["syn_train"], [dev[i] for i in train_index])
        oof[held_index] = fold.predict_proba([dev[i] for i in held_index])[:, 1]
    model = train(config, data["syn_train"], dev)
    return {"model": model, "syn_val": model.predict_proba(data["syn_val"])[:, 1], "real_dev": oof}


def filter_results(pairs: list[Pair], attributes: dict | None = None) -> list[tuple[str, str]]:
    """(overall status, evidence for failed checks) per pair."""
    rows = []
    for p in pairs:
        checks = check_requirements(p.grant["core_requirements"], attributes or p.attributes)
        rows.append((overall(checks), explain(checks, "fail")))
    return rows


def decide(scores, threshold, filters=None) -> list[str]:
    predicted = [LABELS[int(s >= threshold)] for s in scores]
    if filters is None:
        return predicted
    return ["not_match" if f[0] == "fail" else p for p, f in zip(predicted, filters)]


def coverage(filters) -> dict:
    return dict(Counter(f[0] for f in filters))


def errors(pairs, truth, predicted) -> dict:
    return {"false_positives": [p.grant["program"] for p, t, y in zip(pairs, truth, predicted)
                                if t == "not_match" and y == "match"],
            "false_negatives": [p.grant["program"] for p, t, y in zip(pairs, truth, predicted)
                                if t == "match" and y == "not_match"]}


def block(pairs, truth, scores, threshold, filters=None, synthetic=False) -> dict:
    predicted = decide(scores, threshold, filters)
    result = binary_metrics(truth, predicted, [p.original_label for p in pairs] if synthetic else None)
    if not synthetic:
        result.update(errors(pairs, truth, predicted))
    if filters is not None:
        result["filter_coverage"] = coverage(filters)
        result["filter_flipped_to_not_match"] = sum(
            1 for s, f in zip(scores, filters) if s >= threshold and f[0] == "fail")
    return result


def selection_score(syn_truth, syn_pred, dev_truth, dev_pred) -> float:
    return 0.5 * binary_metrics(syn_truth, syn_pred)["macro_f1"] + 0.5 * binary_metrics(dev_truth, dev_pred)["macro_f1"]


def promoted(candidate: dict, reference: dict) -> tuple[bool, list[str]]:
    """The protocol's promotion rule; returns the decision and the reasons."""
    c, r = candidate, reference
    checks = {
        "selection score +0.02": c["selection_score"] >= r["selection_score"] + 0.02,
        "real-dev macro-F1 not lower": c["real_dev"]["macro_f1"] >= r["real_dev"]["macro_f1"],
        **{f"{split} match {m} drop <= 0.05": c[split][f"match_{m}"] >= r[split][f"match_{m}"] - 0.05
           for split in ("real_dev", "synthetic_validation") for m in ("precision", "recall")},
    }
    return all(checks.values()), [k for k, ok in checks.items() if not ok]


def v1_saved_oof(data) -> np.ndarray:
    frame = pd.read_csv(V1_DIR / "ardook_predictions.csv").set_index("source_row")
    return np.array([frame.loc[int(p.key.removeprefix("real-row")), "p_match"] for p in data["real_dev"]])


def run(output: Path) -> None:
    data = load_data()
    syn_truth = [p.label for p in data["syn_val"]]
    dev_truth = [p.label for p in data["real_dev"]]
    filters = {s: filter_results(data[s]) for s in ("syn_val", "syn_test", "real_dev", "real_test", "real_all")}
    hidden, hidden_truth = hidden_pairs(ARDOOK_PROFILE["attributes"])
    filters["hidden"] = filter_results(hidden)

    grid = []
    for config in GRID:
        result = evaluate(config, data)
        threshold, score = choose_threshold(syn_truth, result["syn_val"], dev_truth, result["real_dev"])
        grid.append({"config": config, "threshold": threshold, "selection_score": score, **result})
        print(f"{config_name(config)}: selection={score:.3f} threshold={threshold}", flush=True)
    reference = next(g for g in grid if g["config"] == REFERENCE)
    best = max(grid, key=lambda g: (g["selection_score"], g["config"] == REFERENCE))

    v1 = MatchClassifier.load(V1_DIR / "model.joblib")
    saved = {"model": v1, "threshold": v1.threshold, "syn_val": v1.predict_proba(data["syn_val"])[:, 1],
             "real_dev": v1_saved_oof(data)}
    candidates = {"v1 saved model": saved, "v1 config retrained": reference,
                  "grid best: " + config_name(best["config"]): best}

    rows = []
    for name, c in candidates.items():
        for use_filter in (False, True):
            f = (lambda s: filters[s]) if use_filter else (lambda s: None)
            model, t = c["model"], c["threshold"]
            row = {"name": name + (" + filter" if use_filter else ""), "threshold": t,
                   "synthetic_validation": block(data["syn_val"], syn_truth, c["syn_val"], t, f("syn_val"), True),
                   "real_dev": block(data["real_dev"], dev_truth, c["real_dev"], t, f("real_dev"))}
            row["selection_score"] = selection_score(
                syn_truth, decide(c["syn_val"], t, f("syn_val")), dev_truth, decide(c["real_dev"], t, f("real_dev")))
            row["after_selection"] = {
                "synthetic_test": block(data["syn_test"], [p.label for p in data["syn_test"]],
                                        model.predict_proba(data["syn_test"])[:, 1], t, f("syn_test"), True),
                "real_test": block(data["real_test"], [p.label for p in data["real_test"]],
                                   model.predict_proba(data["real_test"])[:, 1], t, f("real_test")),
                "hidden_post_exposure": block(hidden, hidden_truth, model.predict_proba(hidden)[:, 1], t,
                                              f("hidden")),
            }
            rows.append(row)

    by_name = {r["name"]: r for r in rows}
    ref_row = by_name["v1 config retrained"]
    best_row = by_name["grid best: " + config_name(best["config"])]
    model_ok, model_fails = promoted(best_row, ref_row) if best is not reference else (False, ["is the reference"])
    chosen = best if model_ok else saved
    chosen_name = best_row["name"] if model_ok else "v1 saved model"
    base, filtered = by_name[chosen_name], by_name[chosen_name + " + filter"]
    dev_pred = decide(chosen["real_dev"], chosen["threshold"])
    dev_filtered = decide(chosen["real_dev"], chosen["threshold"], filters["real_dev"])
    filter_checks = {
        "real-dev macro-F1 not lower": filtered["real_dev"]["macro_f1"] >= base["real_dev"]["macro_f1"],
        "no correct real-dev prediction flipped": all(
            not (a == t and b != t) for a, b, t in zip(dev_pred, dev_filtered, dev_truth)),
        "synthetic-validation macro-F1 higher": (filtered["synthetic_validation"]["macro_f1"]
                                                 > base["synthetic_validation"]["macro_f1"]),
    }
    filter_ok = all(filter_checks.values())
    decision = {"model": chosen_name, "model_promoted": model_ok, "model_promotion_failed_checks": model_fails,
                "filter_integrated": filter_ok, "filter_checks": filter_checks,
                "threshold": chosen["threshold"]}
    print(json.dumps(decision, indent=1), flush=True)
    write_outputs(output, data, hidden, hidden_truth, filters, grid, rows, decision, chosen, model_ok)


def per_grant_tables(data, hidden, hidden_truth, filters, chosen, filter_on):
    model, t = chosen["model"], chosen["threshold"]
    real = data["real_all"]
    final = model.predict_proba(real)[:, 1]
    oof = dict(zip((p.key for p in data["real_dev"]), chosen["real_dev"]))
    for_profit = filter_results(real, FOR_PROFIT)
    rows = []
    for p, prob, (status, evidence), (fp_status, _) in zip(real, final, filters["real_all"], for_profit):
        score = oof.get(p.key, prob)
        model_pred = LABELS[int(score >= t)]
        final_pred = "not_match" if filter_on and status == "fail" else model_pred
        rows.append({"source_row": int(p.key.removeprefix("real-row")), "program": p.grant["program"],
                     "tracker_fit": p.original_label, "target": p.label or "ambiguous (not scored)",
                     "role": {"dev": "dev (out-of-fold score)", "test": "held-out test (post-exposure)",
                              "ambiguous": "conditional (not scored)"}[p.split],
                     "p_match": round(float(score), 4), "model_prediction": model_pred,
                     "filter": status, "filter_fail_evidence": evidence,
                     "filter_if_for_profit": fp_status, "final_prediction": final_pred})
    hidden_scores = model.predict_proba(hidden)[:, 1]
    hidden_rows = []
    for p, truth, prob, (status, evidence) in zip(hidden, hidden_truth, hidden_scores, filters["hidden"]):
        model_pred = LABELS[int(prob >= t)]
        hidden_rows.append({"grant_id": int(p.key), "program": p.grant["program"], "ta_label": truth,
                            "p_match": round(float(prob), 4), "model_prediction": model_pred,
                            "filter": status, "filter_fail_evidence": evidence,
                            "final_prediction": "not_match" if filter_on and status == "fail" else model_pred})
    return pd.DataFrame(rows), pd.DataFrame(hidden_rows)


def write_outputs(output, data, hidden, hidden_truth, filters, grid, rows, decision, chosen, model_ok):
    output.mkdir(parents=True, exist_ok=False)
    model_path = V1_DIR / "model.joblib"
    if model_ok:
        model_path = output / "model.joblib"
        chosen["model"].threshold = chosen["threshold"]
        chosen["model"].save(model_path)
        reloaded = MatchClassifier.load(model_path)
        if not np.allclose(reloaded.predict_proba(hidden)[:, 1], chosen["model"].predict_proba(hidden)[:, 1]):
            raise AssertionError("Reloaded model gives different probabilities")
    config = {"model": _rel(model_path), "threshold": chosen["threshold"],
              "filter": FILTER_VERSION if decision["filter_integrated"] else None,
              "selected_by": "results/ardook_match_v2_protocol.md", "report": _rel(output / "report.json")}
    (output / "inference_config.json").write_text(json.dumps(config, indent=2) + "\n")
    real_table, hidden_table = per_grant_tables(data, hidden, hidden_truth, filters, chosen,
                                                decision["filter_integrated"])
    real_table.to_csv(output / "ardook_predictions.csv", index=False)
    hidden_table.to_csv(output / "hidden_test_predictions.csv", index=False)

    sensitivity = {}
    for name, attributes in (("provisional profile", ARDOOK_PROFILE["attributes"]), ("entity_type=for-profit", FOR_PROFIT)):
        pairs, truth = hidden_pairs(attributes)
        scores = chosen["model"].predict_proba(pairs)[:, 1]
        sensitivity[name] = {
            "model only": block(pairs, truth, scores, chosen["threshold"]),
            "model + filter": block(pairs, truth, scores, chosen["threshold"], filter_results(pairs))}
    grid_rows = [{"config": config_name(g["config"]), "threshold": g["threshold"],
                  "selection_score": round(g["selection_score"], 4)} for g in grid]
    pd.DataFrame(grid_rows).sort_values("selection_score", ascending=False).to_csv(output / "grid.csv", index=False)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "results/ardook_match_v2_protocol.md",
        "data_sha256": {"clean_synthetic_db": hashlib.sha256(DB_PATH.read_bytes()).hexdigest(),
                        "ardook_workbook": hashlib.sha256(DEFAULT_WORKBOOK.read_bytes()).hexdigest(),
                        "hidden_workbook": hashlib.sha256(HIDDEN_WORKBOOK.read_bytes()).hexdigest(),
                        "hidden_labels": hashlib.sha256(HIDDEN_LABELS.read_bytes()).hexdigest()},
        "code_sha256": {f: hashlib.sha256((REPO_ROOT / f).read_bytes()).hexdigest() for f in
                        ("src/match_classifier.py", "src/tune_match_classifier.py", "src/eligibility_filter.py")},
        "versions": {p: importlib.metadata.version(p) for p in
                     ("torch", "scikit-learn", "sentence-transformers", "numpy", "pandas")},
        "splits": "results/ardook_match_v1/split_manifest.json (asserted equal)",
        "ardook_profile": ARDOOK_PROFILE, "filter_version": FILTER_VERSION,
        "decision": decision, "inference_config": config,
        "candidates": rows, "grid": grid_rows,
        "hidden_profile_sensitivity_post_exposure": sensitivity,
        "notes": ["Hidden and real held-out results are post-exposure regression checks.",
                  "Real-dev numbers are optimistic: 27 configurations and a threshold were chosen on 19 grants.",
                  "Synthetic filter gains are optimistic: synthetic requirements are templated."],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results/ardook_match_v2")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists; choose a new --output directory")
    run(args.output)


if __name__ == "__main__":
    main()
