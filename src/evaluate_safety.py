"""Safety evaluation: does the pipeline avoid overclaiming, quote real text, and flag missing information?

Standalone and read-only: it imports the pipeline and datasets but changes none of them, keeps vectors in
memory (no vector_store folders), and ships its own keyword baseline. Datasets: messy and clean (labeled
synthetic) and real (Ardook_data.xlsx with example profiles; no labels, so only the safety checks apply).

Pass/fail rows follow the README's evaluation table. "Exposure" rows are warnings about uncertainty the
pipeline shows without saying why; they are not failures of the README targets.

Run `python -m src.evaluate_safety --dataset all --embedder keyword` for a fast run without model downloads.
"""

import argparse
import json
import re
from collections import Counter
from datetime import date

import numpy as np
import pandas as pd

from src import ardook_extra_synthetic_data as clean
from src import ardook_messy_synthetic_data as messy
from src.ingest import load_grants as load_real_grants
from src.pipeline import OrganizationProfile, Pipeline
from src.real_eligibility import load_checker as load_real_checker
from src.stores import KeywordEmbedder, MemoryStore

K = 5
NOT_RULES = {"deadline", "application round", "remaining requirements"}
NO_SOURCE = {"", "tbd", "n/a", "none"}

# Example applicants for the real workbook. Fictional; attribute names come from src.real_eligibility.
REAL_PROFILES = [
    ("A U.S. 501(c)(3) nonprofit running after-school STEM mentoring for low-income teens.",
     {"entity_type": "nonprofit", "us_501c3_status": "yes", "business_base_country": "US"}),
    ("A nonprofit that applies machine learning to disaster response and aid logistics.", {}),
    ("A woman-founded U.S. for-profit building affordable digital tools for small farms.",
     {"entity_type": "for-profit", "incorporation_country": "US", "founder_residence_country": "US",
      "women_ownership_percent": 60, "operating_months": 14, "business_base_country": "US",
      "recent_bank_statement_revenue_usd": 1200}),
    ("A social enterprise training smallholder farmers in regenerative agriculture.",
     {"entity_type": "for-profit", "business_base_country": "Kenya"}),
    ("We help people in our community.", {}),
]


def _norm(value) -> str:
    return re.sub(r"\s+", " ", "" if pd.isna(value) else str(value)).strip()


def _expect_unknown(rule, value) -> bool:
    """Mirror EligibilityChecker: a rule can be decided only from a usable attribute value."""
    if rule.operator == "in":
        return not isinstance(value, str) or not value.strip()
    return isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)


def _audit(recs, profile, grants_by_row, checker, labels=None, reasons=None, profile_id=None):
    """Return counters describing the safety of one profile's recommendations."""
    c = Counter()
    titles = [_norm(r.program).casefold() for r in recs]
    c["duplicate_in_top_k"] += len(set(titles)) != len(titles)
    for r in recs:
        c["shown"] += 1
        c["overclaim"] += r.eligibility != "needs_verification"
        c["wrong_eligibility"] += any(check.status == "fail" for check in r.checks)
        row = grants_by_row[r.source_row]
        # Quotes are real: every quoted passage must occur in the workbook row it cites.
        row_text = [_norm(row[f]) for f in Pipeline.EVIDENCE_FIELDS]
        for check in r.checks:
            c["quotes"] += 1
            c["quotes_real"] += _norm(check.evidence) in " ".join(row_text) or not _norm(check.evidence)
        c["evidence_fields_match"] += all(_norm(r.evidence[f]) == _norm(row[f]) for f in Pipeline.EVIDENCE_FIELDS)
        # Missing info is flagged: an undecidable rule must be "unknown", never pass or fail.
        rules = checker.rules.get(row["program"], [])
        rule_checks = [k for k in r.checks if k.criterion not in NOT_RULES]
        for rule, check in zip(rules, rule_checks):
            if _expect_unknown(rule, profile.attributes.get(rule.attribute)):
                c["missing_cases"] += 1
                c["missing_flagged"] += check.status == "unknown"
        deadline = next(k for k in r.checks if k.criterion == "deadline")
        c["deadline_cases"] += pd.isna(row["deadline_date"])
        c["deadline_flagged"] += pd.isna(row["deadline_date"]) and deadline.status == "unknown"
        # Exposure: uncertainty shown to the user without a specific cause.
        c["exposure_deadline_unknown"] += deadline.status == "unknown"
        c["exposure_no_source"] += _norm(r.source_url).casefold() in NO_SOURCE
        if labels is not None:
            label = labels.get(r.source_row)
            c[f"label_{label}"] += 1
            if label == "needs_review":
                why = (reasons or {}).get((profile_id, r.source_row), "")
                grant_side = [x for x in why.split("; ") if x and not x.startswith("profile is missing")]
                if grant_side:
                    c["review_unexplained"] += 1  # pipeline shows only a generic "remaining requirements: unknown"
                else:
                    c["review_explained"] += 1
    return c


def _pipeline(grants, checker, embedder):
    if embedder == "keyword":
        return Pipeline(grants, checker=checker, embedder=KeywordEmbedder(), store=MemoryStore(), lexical_weight=0, dedupe_threshold=None), 1e-12
    return Pipeline(grants, checker=checker, store=MemoryStore()), 0.25


def evaluate(name, embedder, limit=None):
    """Run one dataset. Returns {"counts": Counter, "ranking": {...}}."""
    labels_by_profile = reasons = None
    if name == "real":
        grants, checker, as_of = load_real_grants(), None, date.today()
        checker = load_real_checker(grants)
        profiles = [(i + 1, OrganizationProfile(d, a), None) for i, (d, a) in enumerate(REAL_PROFILES)]
    else:
        mod = messy if name == "messy" else clean
        grants, checker, as_of = mod.load_grants(), mod.load_checker(), clean.REFERENCE_DATE
        frame, matches = mod.load_profiles(), mod.load_matches()
        if limit:
            frame = frame.head(limit)
        labels_by_profile = {
            int(pid): dict(zip(g.source_row.astype(int), g.label)) for pid, g in matches.groupby("profile_id")
        }
        if "reason" in matches:
            reasons = {(int(r.profile_id), int(r.source_row)): r.reason for r in matches.itertuples()}
        profiles = [(int(p.id), OrganizationProfile(p.description, p.attributes), labels_by_profile[int(p.id)])
                    for p in frame.itertuples(index=False)]
    pipeline, min_similarity = _pipeline(grants, checker, embedder)
    grants_by_row = {int(g["source_row"]): g for g in grants.to_dict("records")}
    total, precision, empty = Counter(), [], 0
    for pid, profile, labels in profiles:
        recs = pipeline.recommend(profile, as_of=as_of, top_k=K, min_similarity=min_similarity)
        empty += not recs
        total += _audit(recs, profile, grants_by_row, checker, labels, reasons, pid)
        relevant = {r for r, label in (labels or {}).items() if label == "match"}
        if relevant:
            precision.append(sum(r.source_row in relevant for r in recs) / K)
    total["profiles"] = len(profiles)
    total["empty_results"] = empty
    return {"counts": total, "precision_at_5": float(np.mean(precision)) if precision else None,
            "evaluable_profiles": len(precision)}


def _rate(num, den):
    return None if not den else num / den


def summarize(name, embedder, result):
    c = result["counts"]
    shown = c["shown"] or 1
    rows = [  # (metric, value, target, passed)
        ("Wrong eligibility (shown with a failed check)", c["wrong_eligibility"] / shown, "0%", c["wrong_eligibility"] == 0),
        ("Overclaim (eligibility other than needs_verification)", c["overclaim"] / shown, "0%", c["overclaim"] == 0),
        ("Quotes are real (evidence found in cited row)", _rate(c["quotes_real"], c["quotes"]), "100%", c["quotes_real"] == c["quotes"]),
        ("Evidence fields match the row", _rate(c["evidence_fields_match"], c["shown"]), "100%", c["evidence_fields_match"] == c["shown"]),
        ("Missing applicant info flagged unknown", _rate(c["missing_flagged"], c["missing_cases"]), "100%", c["missing_flagged"] == c["missing_cases"]),
        ("Unparseable deadline flagged unknown", _rate(c["deadline_flagged"], c["deadline_cases"]), "100%", c["deadline_flagged"] == c["deadline_cases"]),
    ]
    if c["label_ineligible"] or "label_match" in c or "label_unrelated" in c:
        rows.append(("Shown pairs labeled ineligible", c["label_ineligible"] / shown, "0%", c["label_ineligible"] == 0))
    print(f"\n=== {name} | {embedder} | {c['profiles']} profiles, {c['shown']} recommendations ===")
    for metric, value, target, passed in rows:
        shown_value = "n/a (no cases)" if value is None else f"{100 * value:.1f}%"
        print(f"  {'PASS' if passed else 'FAIL'}  {metric:<56} {shown_value:>14}  (target {target})")
    print("  Exposure (shown without a stated cause):")
    print(f"        deadline unknown on {c['exposure_deadline_unknown']}/{shown} ({100 * c['exposure_deadline_unknown'] / shown:.0f}%)"
          f" | no source on {c['exposure_no_source']} | duplicate listing in top {K} for {c['duplicate_in_top_k']} profiles"
          f" | empty results for {c['empty_results']} profiles")
    if c["review_explained"] or c["review_unexplained"]:
        total = c["review_explained"] + c["review_unexplained"]
        print(f"        needs_review shown: {total}; cause the pipeline can name: {c['review_explained']}"
              f" ({100 * c['review_explained'] / total:.0f}%), cannot name: {c['review_unexplained']}")
    if result["precision_at_5"] is not None:
        print(f"  Ranking (info): precision@5 {result['precision_at_5']:.3f} on {result['evaluable_profiles']} profiles with a labeled match")
    return all(passed for _, _, _, passed in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("messy", "clean", "real", "all"), default="all")
    parser.add_argument("--embedder", choices=("semantic", "keyword"), default="semantic")
    parser.add_argument("--limit-profiles", type=int, help="Evaluate only the first N synthetic profiles.")
    parser.add_argument("--json", action="store_true", help="Print raw counts as JSON.")
    args = parser.parse_args()
    names = ["messy", "clean", "real"] if args.dataset == "all" else [args.dataset]
    ok = True
    for name in names:
        result = evaluate(name, args.embedder, args.limit_profiles)
        if args.json:
            print(json.dumps({"dataset": name, **result, "counts": dict(result["counts"])}, indent=1))
        else:
            ok &= summarize(name, args.embedder, result)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
