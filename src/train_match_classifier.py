"""Train, select and evaluate the binary applicant-to-grant match classifier.

Run: HF_HUB_OFFLINE=1 python -m src.train_match_classifier --output results/ardook_match_v1
Follows results/ardook_match_protocol.md: selection uses synthetic validation and real dev
grants only; the synthetic test split and held-out real grants are scored once, afterwards.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold

from src.ardook_extra_synthetic_data import DB_PATH
from src.match_classifier import (ARDOOK_PROFILE, FORBIDDEN_FIELDS, GRANT_FIELDS, LABELS, SEED,
                                  MatchClassifier, Pair, build_classifier, grant_family, grant_text, real_pairs,
                                  synthetic_pairs)
from src.paths import DEFAULT_WORKBOOK, REPO_ROOT

CANDIDATES = [("tfidf", "logreg"), ("embedding:minilm", "logreg"), ("embedding:mpnet", "logreg"),
              ("embedding:bge", "logreg"), ("simbag:minilm", "hgb"), ("simbag:mpnet", "hgb"),
              ("simbag:bge", "hgb"),
              # Added after dev-only diagnosis; see the protocol's Deviations section.
              ("twostage:none", "logreg"), ("twostage:minilm", "logreg"), ("twostage:mpnet", "logreg"),
              ("twostage:bge", "logreg")]
REAL_WEIGHT_SHARE = 0.10  # fixed in the protocol; never tuned
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.05), 2)
STOPWORDS = set("a an and the of for in on to with by or is are our we its it that this as at from "
                "be has have their who which not no applicant facts".split())


# ---------------------------------------------------------------- metrics

def binary_metrics(truth: list[str], predicted: list[str], original: list[str] | None = None) -> dict:
    y = np.array([LABELS.index(t) for t in truth])
    p = np.array([LABELS.index(t) for t in predicted])
    precision, recall, f1, _ = precision_recall_fscore_support(y, p, labels=[1], zero_division=0)
    result = {"n": len(y), "class_counts": dict(Counter(truth)),
              "accuracy": float((y == p).mean()) if len(y) else None,
              "macro_f1": float(f1_score(y, p, labels=[0, 1], average="macro", zero_division=0)),
              "match_precision": float(precision[0]), "match_recall": float(recall[0]),
              "match_f1": float(f1[0]),
              "confusion_matrix": {"rows_truth_cols_predicted": list(LABELS),
                                   "matrix": confusion_matrix(y, p, labels=[0, 1]).tolist()}}
    if original is not None:  # synthetic: where do predicted matches come from?
        result["predicted_match_by_original_label"] = dict(Counter(
            o for o, pred in zip(original, predicted) if pred == "match"))
        result["original_label_counts"] = dict(Counter(original))
    return result


def macro_f1_at(truth: list[str], scores: np.ndarray, threshold: float) -> float:
    predicted = [LABELS[int(s >= threshold)] for s in scores]
    return binary_metrics(truth, predicted)["macro_f1"]


def choose_threshold(syn_truth, syn_scores, real_truth, real_scores) -> tuple[float, float]:
    """Threshold maximizing the protocol's selection score; ties go to the lower threshold."""
    best = max(((0.5 * macro_f1_at(syn_truth, syn_scores, t) + 0.5 * macro_f1_at(real_truth, real_scores, t)), -t)
               for t in THRESHOLDS)
    return float(-best[1]), float(best[0])


# ---------------------------------------------------------------- baselines

def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z][a-z\-]+", text.casefold()) if w not in STOPWORDS}


def keyword_scores(pairs: list[Pair]) -> np.ndarray:
    """Share of the applicant's content words that also appear in the grant text."""
    scores = []
    for p in pairs:
        applicant = _words(p.applicant)
        scores.append(len(applicant & _words(grant_text(p.grant))) / max(1, len(applicant)))
    return np.array(scores)


# ---------------------------------------------------------------- checks

def leakage_checks(synthetic: list[Pair], real: list[Pair]) -> dict:
    """Raise on any violation; return a record of what was checked."""
    checks = {}
    for attribute, name in (("applicant", "applicant_description"), ("program", "grant_program")):
        seen = {}
        for p in synthetic:
            value = p.applicant if attribute == "applicant" else p.grant["program"]
            seen.setdefault(value, set()).add(p.split)
        crossing = [v for v, s in seen.items() if len(s) > 1]
        if crossing:
            raise AssertionError(f"{name} crosses synthetic splits: {crossing[:3]}")
        checks[f"synthetic_{name}_groups_disjoint"] = True
    for p in synthetic + real:
        if set(p.grant) != set(GRANT_FIELDS):
            raise AssertionError(f"Grant fields outside the allowlist for {p.key}")
    checks["grant_fields_allowlist"] = list(GRANT_FIELDS)
    checks["forbidden_fields_never_in_features"] = list(FORBIDDEN_FIELDS)
    families = {}
    for p in real:
        if p.split in ("dev", "test"):
            families.setdefault(grant_family(p.grant["program"]), set()).add(p.split)
    if any(len(s) > 1 for s in families.values()):
        raise AssertionError("A real grant family crosses dev and test")
    checks["real_families_disjoint_dev_test"] = True
    real_programs = {p.grant["program"].casefold() for p in real}
    if real_programs & {p.grant["program"].casefold() for p in synthetic}:
        raise AssertionError("A real programme name appears in the synthetic data")
    checks["no_real_programme_names_in_synthetic"] = True
    return checks


def overlap_report(synthetic: list[Pair], real: list[Pair]) -> list[dict]:
    """How close each real grant is to the nearest synthetic grant (the generator copied phrases)."""
    syn_grants = sorted({grant_text(p.grant) for p in synthetic})
    real_rows = [p for p in real]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(syn_grants + [grant_text(p.grant) for p in real_rows])
    similarity = (vectorizer.transform([grant_text(p.grant) for p in real_rows])
                  @ vectorizer.transform(syn_grants).T).toarray()

    def shingles(text, n=5):
        words = re.findall(r"\w+", text.casefold())
        return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}

    syn_shingles = set().union(*(shingles(t) for t in syn_grants))
    rows = []
    for p, sims in zip(real_rows, similarity):
        shared = sorted(shingles(grant_text(p.grant)) & syn_shingles)
        rows.append({"program": p.grant["program"], "split": p.split,
                     "max_tfidf_cosine_to_synthetic": round(float(sims.max()), 3),
                     "shared_5_word_phrases": shared})
    return rows


# ---------------------------------------------------------------- training

def real_weights(n_synthetic: int, n_real: int) -> np.ndarray:
    return np.full(n_real, REAL_WEIGHT_SHARE * n_synthetic / max(1, n_real))


def fit(spec, estimator, synthetic_train, real_train=()):
    pairs = list(synthetic_train) + list(real_train)
    weights = np.concatenate([np.ones(len(synthetic_train)),
                              real_weights(len(synthetic_train), len(real_train))])
    return build_classifier(spec, estimator).fit(pairs, sample_weight=weights)


def evaluate_candidate(spec, estimator, data) -> dict:
    """Out-of-sample P(match) on synthetic validation and real dev, for both variants."""
    syn_train, syn_val, real_dev = data["syn_train"], data["syn_val"], data["real_dev"]
    result = {}
    start = time.perf_counter()
    model = fit(spec, estimator, syn_train)
    result["synthetic_only"] = {"model": model, "syn_val": model.predict_proba(syn_val)[:, 1],
                                "real_dev": model.predict_proba(real_dev)[:, 1],
                                "seconds": time.perf_counter() - start}
    start = time.perf_counter()
    groups = [grant_family(p.grant["program"]) for p in real_dev]
    oof = np.zeros(len(real_dev))
    folds = GroupKFold(n_splits=5)
    for train_index, held_index in folds.split(real_dev, groups=groups):
        fold_model = fit(spec, estimator, syn_train, [real_dev[i] for i in train_index])
        oof[held_index] = fold_model.predict_proba([real_dev[i] for i in held_index])[:, 1]
    model = fit(spec, estimator, syn_train, real_dev)
    result["with_real_dev"] = {"model": model, "syn_val": model.predict_proba(syn_val)[:, 1],
                               "real_dev": oof, "seconds": time.perf_counter() - start}
    return result


def summarize(name, scores_syn, scores_real, data, threshold, selection_score, seconds):
    syn_truth = [p.label for p in data["syn_val"]]
    real_truth = [p.label for p in data["real_dev"]]
    pick = lambda scores: [LABELS[int(s >= threshold)] for s in scores]  # noqa: E731
    return {"name": name, "threshold": threshold, "selection_score": selection_score,
            "seconds": round(seconds, 1),
            "synthetic_validation": binary_metrics(syn_truth, pick(scores_syn),
                                                   [p.original_label for p in data["syn_val"]]),
            "real_dev": binary_metrics(real_truth, pick(scores_real))}


def test_metrics(scores_syn, scores_real, data, threshold):
    pick = lambda scores: [LABELS[int(s >= threshold)] for s in scores]  # noqa: E731
    return {"synthetic_test": binary_metrics([p.label for p in data["syn_test"]], pick(scores_syn),
                                             [p.original_label for p in data["syn_test"]]),
            "real_test": binary_metrics([p.label for p in data["real_test"]], pick(scores_real))}


def run(output: Path) -> None:
    synthetic, excluded = synthetic_pairs()
    real = real_pairs()
    checks = leakage_checks(synthetic, real)
    data = {"syn_train": [p for p in synthetic if p.split == "train"],
            "syn_val": [p for p in synthetic if p.split == "validation"],
            "syn_test": [p for p in synthetic if p.split == "test"],
            "real_dev": [p for p in real if p.split == "dev"],
            "real_test": [p for p in real if p.split == "test"]}
    syn_val_truth = [p.label for p in data["syn_val"]]
    real_dev_truth = [p.label for p in data["real_dev"]]

    rows, fitted = [], {}
    # Baselines share the same selection protocol.
    constant = {"always not_match": 0.0, "always match": 1.0}
    for name, value in constant.items():
        rows.append(summarize(name, np.full(len(syn_val_truth), value), np.full(len(real_dev_truth), value),
                              data, 0.5, 0.5 * macro_f1_at(syn_val_truth, np.full(len(syn_val_truth), value), 0.5)
                              + 0.5 * macro_f1_at(real_dev_truth, np.full(len(real_dev_truth), value), 0.5), 0))
        fitted[name] = ("constant", value)
    kw_syn, kw_real = keyword_scores(data["syn_val"]), keyword_scores(data["real_dev"])
    threshold, score = choose_threshold(syn_val_truth, kw_syn, real_dev_truth, kw_real)
    rows.append(summarize("keyword overlap", kw_syn, kw_real, data, threshold, score, 0))
    fitted["keyword overlap"] = ("keyword", None)

    for spec, estimator in CANDIDATES:
        print(f"Training {spec} + {estimator} ...", flush=True)
        result = evaluate_candidate(spec, estimator, data)
        for variant, values in result.items():
            name = f"{spec} + {estimator} [{variant}]"
            threshold, score = choose_threshold(syn_val_truth, values["syn_val"], real_dev_truth, values["real_dev"])
            row = summarize(name, values["syn_val"], values["real_dev"], data, threshold, score, values["seconds"])
            rows.append(row)
            fitted[name] = ("model", values["model"])
            print(f"  {name}: selection={score:.3f} threshold={threshold} "
                  f"syn-val macro-F1={row['synthetic_validation']['macro_f1']:.3f} "
                  f"real-dev macro-F1={row['real_dev']['macro_f1']:.3f}", flush=True)

    # ---- Selection is frozen here; test data is touched only below.
    selected = max(rows, key=lambda r: (r["selection_score"], -len(r["name"])))
    print(f"Selected: {selected['name']} (score {selected['selection_score']:.3f})", flush=True)
    for row in rows:
        kind, obj = fitted[row["name"]]
        if kind == "constant":
            s_syn, s_real = np.full(len(data["syn_test"]), obj), np.full(len(data["real_test"]), obj)
        elif kind == "keyword":
            s_syn, s_real = keyword_scores(data["syn_test"]), keyword_scores(data["real_test"])
        else:
            obj.threshold = row["threshold"]
            s_syn, s_real = obj.predict_proba(data["syn_test"])[:, 1], obj.predict_proba(data["real_test"])[:, 1]
        row["test"] = test_metrics(s_syn, s_real, data, row["threshold"])

    model = fitted[selected["name"]][1]
    write_outputs(output, model, selected, rows, data, real, checks, excluded, synthetic)


def write_outputs(output, model, selected, rows, data, real, checks, excluded, synthetic):
    output.mkdir(parents=True, exist_ok=False)
    model.save(output / "model.joblib")
    reloaded = MatchClassifier.load(output / "model.joblib")
    real_probabilities = model.predict_proba(real)[:, 1]
    if not np.allclose(reloaded.predict_proba(real)[:, 1], real_probabilities, atol=1e-7):
        raise AssertionError("Reloaded model gives different probabilities")

    oof = {}
    if "[with_real_dev]" in selected["name"]:  # out-of-fold P(match) for the dev grants
        spec, estimator = model.features_spec, model.estimator_name
        dev = data["real_dev"]
        groups = [grant_family(p.grant["program"]) for p in dev]
        for train_index, held_index in GroupKFold(n_splits=5).split(dev, groups=groups):
            fold_model = fit(spec, estimator, data["syn_train"], [dev[i] for i in train_index])
            for i, prob in zip(held_index, fold_model.predict_proba([dev[i] for i in held_index])[:, 1]):
                oof[dev[i].key] = float(prob)
    else:
        oof = {p.key: float(prob) for p, prob in zip(real, real_probabilities) if p.split == "dev"}

    per_grant = []
    for p, prob in zip(real, real_probabilities):
        role = {"dev": "dev (trained on; out-of-fold score shown)" if "[with_real_dev]" in selected["name"]
                else "dev (selection only)", "test": "held-out test", "ambiguous": "ambiguous (not scored)"}[p.split]
        score = oof.get(p.key, float(prob))
        predicted = LABELS[int(score >= model.threshold)]
        per_grant.append({"source_row": int(p.key.removeprefix("real-row")), "program": p.grant["program"],
                          "tracker_fit": p.original_label, "target": p.label or "ambiguous", "role": role,
                          "p_match": round(score, 4), "p_match_final_model": round(float(prob), 4),
                          "predicted": predicted,
                          "outcome": ("correct" if predicted == p.label else
                                      "missed match" if p.label == "match" else
                                      "false match" if p.label else "n/a")})
    pd.DataFrame(per_grant).to_csv(output / "ardook_predictions.csv", index=False)

    syn_test_probs = model.predict_proba(data["syn_test"])[:, 1]
    pd.DataFrame([{"key": p.key, "original_label": p.original_label, "target": p.label,
                   "p_match": round(float(s), 4), "predicted": LABELS[int(s >= model.threshold)]}
                  for p, s in zip(data["syn_test"], syn_test_probs)]).to_csv(
        output / "synthetic_test_predictions.csv", index=False)

    manifest = {split: sorted({p.key for p in data[split]}) for split in data}
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "protocol": "results/ardook_match_protocol.md",
        "data_sha256": {"clean_synthetic_db": hashlib.sha256(DB_PATH.read_bytes()).hexdigest(),
                        "ardook_workbook": hashlib.sha256(DEFAULT_WORKBOOK.read_bytes()).hexdigest()},
        "code_sha256": {f: hashlib.sha256((REPO_ROOT / f).read_bytes()).hexdigest()
                        for f in ("src/match_classifier.py", "src/train_match_classifier.py")},
        "versions": {p: importlib.metadata.version(p) for p in
                     ("torch", "scikit-learn", "sentence-transformers", "numpy", "pandas")},
        "ardook_profile": ARDOOK_PROFILE, "labels": list(LABELS),
        "real_weight_share": REAL_WEIGHT_SHARE, "leakage_checks": checks,
        "split_sizes": {s: {"pairs": len(v), "class_counts": dict(Counter(p.label for p in v)),
                            "original_label_counts": dict(Counter(p.original_label for p in v))}
                        for s, v in data.items()},
        "synthetic_cross_split_pairs_excluded": excluded,
        "real_ambiguous_grants": [p.grant["program"] for p in real if p.split == "ambiguous"],
        "selected": selected["name"], "selected_threshold": model.threshold,
        "candidates": rows,
        "real_synthetic_overlap": overlap_report(synthetic, real),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"Saved model and report to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results/ardook_match_v1")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists; choose a new --output directory")
    run(args.output)


if __name__ == "__main__":
    main()
