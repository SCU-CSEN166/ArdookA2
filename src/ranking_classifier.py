"""Small supervised relevance baseline; never predicts eligibility."""

from collections import Counter
from datetime import date
import json

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

LABELS = ["poor_fit", "possible_fit", "strong_fit"]
ELIGIBILITY = {"no_known_conflict", "ineligible", "unknown"}
REVIEW_FIELDS = ["pair_id", "relevance", "eligibility", "reviewer", "reviewed_on", "rationale", "evidence"]


def load_reviews(pairs_path, reviews_path):
    pairs = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]
    by_id = {p["pair_id"]: p for p in pairs}
    if len(by_id) != len(pairs):
        raise ValueError("Duplicate pair IDs in the frozen dataset")
    profiles = {}
    for pair in pairs:
        if pair["split"] not in {"train", "validation", "test"}:
            raise ValueError("Unknown dataset split")
        identity = (pair["split"], pair["profile_description"],
                    json.dumps(pair["profile_attributes"], sort_keys=True))
        previous = profiles.setdefault(pair["profile_id"], identity)
        if previous != identity:
            raise ValueError("An organization has inconsistent content or crosses dataset splits")
    reviews = pd.read_excel(reviews_path, sheet_name="Reviews", keep_default_na=False)
    if not set(REVIEW_FIELDS) <= set(reviews.columns):
        raise ValueError(f"Reviews sheet must include {REVIEW_FIELDS}")
    if reviews.pair_id.duplicated().any():
        raise ValueError("Duplicate pair IDs in reviews")
    if set(reviews.pair_id) != set(by_id):
        raise ValueError("Review pair IDs must exactly match the frozen dataset")
    completed = []
    counts = Counter(pending=0, insufficient_information=0)
    for row in reviews.to_dict("records"):
        values = {key: str(row[key]).strip() for key in REVIEW_FIELDS}
        label = values["relevance"]
        if not label:
            counts["pending"] += 1
            continue
        pair = by_id[values["pair_id"]]
        if label not in LABELS + ["insufficient_information"]:
            raise ValueError(f"{pair['pair_id']}: invalid relevance label {label!r}")
        if values["eligibility"] not in ELIGIBILITY:
            raise ValueError(f"{pair['pair_id']}: choose an eligibility label")
        if not values["reviewer"] or not values["rationale"]:
            raise ValueError(f"{pair['pair_id']}: reviewer and rationale are required")
        # Accept Excel dates as well as ISO text, but reject missing or invalid dates.
        date.fromisoformat(values["reviewed_on"].split(" ")[0])
        evidence = values["evidence"]
        passages = [pair[f] for f in ("core_requirements", "type", "amount_benefit", "deadline", "current_status")]
        if evidence and not any(evidence in passage for passage in passages):
            raise ValueError(f"{pair['pair_id']}: evidence must be an exact quote from a supplied grant field")
        if values["eligibility"] == "ineligible" and not evidence:
            raise ValueError(f"{pair['pair_id']}: ineligible requires a supporting quote")
        counts[label] += 1
        if label != "insufficient_information":
            completed.append({**pair, "label": label})
    return completed, dict(counts)


def pair_texts(rows):
    # Explicit allowlist: no review text, fit ratings, IDs, split, or tracker advice.
    return [(r["profile_description"], " ".join(r[f] for f in
             ("program", "type", "core_requirements", "amount_benefit"))) for r in rows]


class PairFeatures(TransformerMixin, BaseEstimator):
    """Learn vocabulary on training text; include shared terms for pair-specific fit."""

    def fit(self, X, y=None):
        self.vectorizer_ = TfidfVectorizer(ngram_range=(1, 2), max_features=10000,
                                         sublinear_tf=True)
        self.vectorizer_.fit([text for pair in X for text in pair])
        return self

    def transform(self, X):
        org = self.vectorizer_.transform([pair[0] for pair in X])
        grant = self.vectorizer_.transform([pair[1] for pair in X])
        return hstack([org, grant, org.multiply(grant), abs(org - grant)], format="csr")


def train(rows):
    partitions = {split: [r for r in rows if r["split"] == split]
                  for split in ("train", "validation", "test")}
    training = partitions["train"]
    class_counts = Counter(r["label"] for r in training)
    if len(training) < 30 or any(class_counts[label] < 5 for label in LABELS):
        raise ValueError("Need at least 30 human-labeled training pairs, including 5 per relevance class")
    for split in ("validation", "test"):
        if len(partitions[split]) < 10:
            raise ValueError(f"Need at least 10 human-labeled {split} pairs")
    model = Pipeline([("features", PairFeatures()),
                      ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000,
                                                        random_state=42))])
    model.fit(pair_texts(training), [r["label"] for r in training])
    majority = class_counts.most_common(1)[0][0]
    report = {"model": "TF-IDF pair features + balanced logistic regression",
              "classes": LABELS, "training_counts": dict(class_counts),
              "limitations": ["Small fictional-profile pilot, not production validation.",
                              "Organizations are held out; grant catalog is shared across splits.",
                              "Sampled candidates do not measure full-catalog recall.",
                              "Relevance predictions do not establish eligibility."]}
    for split in ("validation", "test"):
        subset = partitions[split]
        truth = [r["label"] for r in subset]
        predicted = model.predict(pair_texts(subset))
        report[split] = {
            "pairs": len(subset), "organizations": sorted({r["profile_id"] for r in subset}),
            "macro_f1": float(f1_score(truth, predicted, labels=LABELS, average="macro", zero_division=0)),
            "majority_macro_f1": float(f1_score(truth, np.repeat(majority, len(truth)),
                                               labels=LABELS, average="macro", zero_division=0)),
            "classification_report": classification_report(truth, predicted, labels=LABELS,
                                                            output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(truth, predicted, labels=LABELS).tolist(),
        }
    return model, report
