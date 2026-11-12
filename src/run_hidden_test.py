"""Run the recommendation pipeline on the Week 8 Ardook hidden-test workbook.

Writes hidden_test_outputs.csv (grant_id,label,reason). A grant is "Relevant" when it
passes every eligibility rule that can be checked and its dense relevance to the Ardook
description is at least --threshold. Deadlines and open/closed status are ignored, as the
lab instructions require.
"""

import argparse
import csv
from datetime import date
import json
from pathlib import Path

import pandas as pd

from src.ingest import COLUMNS, clean_grants
from src.pipeline import (
    DEFAULT_MIN_SIMILARITY,
    EligibilityChecker,
    EligibilityRule,
    OrganizationProfile,
    Pipeline,
)
from src.stores import MemoryStore

ARDOOK_DESCRIPTION = (
    "Ardook is an award-winning social enterprise in Yerevan, Armenia, expanding to Georgia. "
    "It employs mothers of children with disabilities, many of them single mothers, who wash "
    "and iron clothes professionally from home with equipment, training and support from Ardook, "
    "while couriers handle pickup and delivery. It serves individuals, hotels, restaurants, "
    "preschools and beauty salons, creating decent work and reducing inequalities (UN SDG 8 and 10) "
    "with eco-friendly reusable bags and recycled hangers."
)
ARDOOK_ATTRIBUTES = {
    "business_base_country": "Armenia",
    "founder_residence_country": "Armenia",
    "incorporation_country": "Armenia",
    "armenia_region": "Yerevan",
    "women_led": "yes",
}

# Transcribed from the hidden workbook's Core Requirements, same convention as
# src/real_eligibility.py: only unambiguous geography requirements are encoded.
RULES = {
    "Women's Fund Armenia - Small Grants for Women-Led Initiatives in Aragatsotn": [
        EligibilityRule("armenia_region", "in", ["Aragatsotn"], "Aragatsotn Province"),
    ],
    "Tory Burch Foundation Fellows Program": [
        EligibilityRule("business_base_country", "in", ["US"], "operating in the U.S."),
    ],
    "Women Founders Network Fast Pitch 2027": [
        EligibilityRule("business_base_country", "in", ["US"], "U.S.-based for-profit"),
    ],
    "IOM ELEVATE Business Grants": [
        EligibilityRule(
            "armenia_region", "in", ["Tavush", "Lori", "Shirak", "Gegharkunik"],
            "Tavush, Lori, Shirak and Gegharkunik provinces",
        ),
    ],
    "People in Need - Grant Call for Armenian Civil Society Organisations": [
        EligibilityRule(
            "armenia_region", "in",
            ["Aragatsotn", "Ararat", "Armavir", "Gegharkunik", "Kotayk", "Lori",
             "Shirak", "Syunik", "Tavush", "Vayots Dzor"],
            "working outside Yerevan",
        ),
    ],
}


def load_hidden(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0, engine="openpyxl", keep_default_na=False)
    raw.columns = raw.columns.str.strip()
    ids = raw.pop("Grant ID").astype(int).tolist()
    for column in COLUMNS:
        if column not in raw.columns:
            raw[column] = ""
    raw["Priority / Deadline Order"] = ids
    # Ignore deadlines and closed status: the lab judges eligibility and fit only.
    raw["Deadline"] = ""
    raw["Current Status (as last reported)"] = ""
    grants = clean_grants(raw[list(COLUMNS)])
    grants["grant_id"] = ids
    return grants


# Same vocabulary as the synthetic profiles the classifier was trained on.
CLASSIFIER_ATTRIBUTES = {"region": "Armenia", "woman_led": "yes"}


def run_pytorch(grants: pd.DataFrame, args) -> None:
    """Train the Week 7 MiniLM-L6 + PyTorch MLP exactly as compare_models does, then
    classify each (Ardook, hidden grant) pair. Relevant = predicted class 'match'."""
    import numpy as np
    import torch

    from src.compare_models import LABELS, embedding_features, load_partitions, train_mlp

    torch.set_num_threads(4)
    partitions = load_partitions()
    organization = ARDOOK_DESCRIPTION + "\nApplicant facts: " + json.dumps(
        CLASSIFIER_ATTRIBUTES, sort_keys=True)
    hidden = [(organization, "\n".join(str(row[f] if pd.notna(row[f]) else "") for f in
               ("program", "type", "core_requirements", "amount_benefit")))
              for _, row in grants.iterrows()]
    texts = {s: [(r["organization"], r["grant"]) for r in partitions[s]]
             for s in ("train", "validation")}
    texts["test"] = hidden  # train_mlp predicts on "test"; here that is the hidden set
    labels = {s: np.array([LABELS.index(r["label"]) for r in partitions[s]])
              for s in ("train", "validation")}
    print("Encoding with all-MiniLM-L6-v2 and training the PyTorch MLP...", flush=True)
    features = embedding_features(texts, "all-MiniLM-L6-v2")
    predictions, settings = train_mlp(features, labels)
    reasons = {"match": "Classifier predicted match (shared focus, no known rule violation)",
               "ineligible": "Classifier predicted ineligible (shared focus but a rule is violated)",
               "unrelated": "Classifier predicted unrelated (no shared focus)"}
    rows = []
    for index, predicted in enumerate(predictions["test"]):
        name = LABELS[int(predicted)]
        rows.append({"grant_id": int(grants.loc[index, "grant_id"]),
                     "label": "Relevant" if name == "match" else "Not relevant",
                     "reason": reasons[name], "predicted_class": name,
                     "program": grants.loc[index, "program"]})
    output = args.output
    if output == Path("results/hidden_test/hidden_test_outputs.csv"):
        output = Path("results/hidden_test/pytorch/hidden_test_outputs.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["grant_id", "label", "reason"])
        writer.writeheader()
        writer.writerows({k: r[k] for k in ("grant_id", "label", "reason")} for r in rows)
    details = output.with_name(output.stem + "_details.json")
    details.write_text(json.dumps({"classifier": "all-MiniLM-L6-v2 + PyTorch MLP",
                                   "threshold": "Relevant iff predicted class == match",
                                   "best_epoch": settings["best_epoch"],
                                   "organization_text": organization, "rows": rows}, indent=2))
    for r in rows:
        print(f"{r['grant_id']:>2}  {r['label']:<13} {r['predicted_class']:<10} {r['program']}")
    print(f"\nWrote {output} and {details}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/hidden_test/hidden_test_outputs.csv"))
    parser.add_argument("--threshold", type=float, default=DEFAULT_MIN_SIMILARITY)
    parser.add_argument("--no-rules", action="store_true", help="Semantic relevance only.")
    parser.add_argument("--classifier", choices=("pipeline", "pytorch"), default="pipeline",
                        help="pytorch: Week 7 MiniLM-L6 + PyTorch MLP (label 'match' = Relevant).")
    args = parser.parse_args()

    grants = load_hidden(args.workbook)
    if args.classifier == "pytorch":
        run_pytorch(grants, args)
        return
    checker = EligibilityChecker({} if args.no_rules else RULES)
    for program, rules in checker.rules.items():
        row = grants.loc[grants["program"] == program]
        if row.empty:
            raise ValueError(f"Rule program missing from workbook: {program}")
        for rule in rules:
            if rule.evidence not in row.iloc[0]["core_requirements"]:
                raise ValueError(f"Evidence changed for {program}: {rule.evidence}")
    pipeline = Pipeline(grants=grants, checker=checker, store=MemoryStore())
    profile = OrganizationProfile(ARDOOK_DESCRIPTION, ARDOOK_ATTRIBUTES)

    query = pipeline._normalize(pipeline.embedder.encode([ARDOOK_DESCRIPTION]))[0]
    _, relevance = pipeline._dense_scores(query)

    rows = []
    for index, grant in enumerate(pipeline.grants):
        checks = checker.check(grant, profile, date.today())
        failed = [c for c in checks if c.status == "fail"]
        score = float(relevance[index])
        if failed:
            label = "Not relevant"
            reason = "Ineligible: requires " + "; ".join(c.evidence for c in failed)
        elif score < args.threshold:
            label = "Not relevant"
            reason = f"Low mission fit (relevance {score:.3f} < {args.threshold})"
        else:
            label = "Relevant"
            reason = f"Mission fit (relevance {score:.3f} >= {args.threshold}); eligibility needs verification"
        rows.append({"grant_id": int(grants.loc[index, "grant_id"]), "label": label,
                     "reason": reason, "score": round(score, 4), "program": grant["program"]})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["grant_id", "label", "reason"])
        writer.writeheader()
        writer.writerows({k: r[k] for k in ("grant_id", "label", "reason")} for r in rows)
    details = args.output.with_name(args.output.stem + "_details.json")
    details.write_text(json.dumps({"threshold": args.threshold, "rules": not args.no_rules,
                                   "description": ARDOOK_DESCRIPTION, "rows": rows}, indent=2))
    for r in rows:
        print(f"{r['grant_id']:>2}  {r['label']:<13} {r['score']:.3f}  {r['program']}")
    print(f"\nWrote {args.output} and {details}")


if __name__ == "__main__":
    main()
