"""Extra synthetic training pairs worded like the real tracker, with more regions (incl. Armenia).

Phrases are paraphrased from the 37 tracker grants only, never from the hidden-test workbook.
Labels follow the same rule as ardook_extra_synthetic_data: different focus -> unrelated, a
violated rule -> ineligible, otherwise match. A rule fails only when the applicant states the fact.
"""

import json
import random

from src.ardook_extra_synthetic_data import FOCUS, PROFILE_REVENUE
from src.ranking_classifier import LABELS

REGION_NAME = {"US": "the U.S.", "EU": "the EU", "UK": "the UK", "India": "India",
               "Africa": "sub-Saharan Africa", "Armenia": "Armenia"}
ENTITY_PHRASES = {
    ("nonprofit",): ["Nonprofit", "Lead applicant must be an eligible 501(c)(3)", "Tech nonprofit / nonprofit business model"],
    ("for-profit",): ["Active for-profit", "Incorporated for-profit startup", "Venture must be a for-profit"],
    ("university",): ["Academic institution", "University-affiliated team"],
    ("for-profit", "nonprofit"): ["For-profit or nonprofit venture"],
    ("nonprofit", "university"): ["Nonprofits and academic institutions"],
}
REGION_PHRASES = {
    ("US",): ["U.S.-based", "U.S. resident founder", "Qualifying U.S. small business"],
    ("EU",): ["Eligible Horizon Europe country"],
    ("EU", "Armenia"): ["Eligible Horizon Europe country incl. Armenia"],
    ("UK",): ["UK-registered organization", "UK-based team"],
    ("India",): ["Operating in India", "Venture must aim to create social impact in India"],
    ("Africa",): ["Operating in sub-Saharan Africa"],
    ("Armenia",): ["Based in Armenia", "Operating in Armenia"],
}
WOMAN_PHRASES = ["Woman-led", "Woman founder or co-founder", "Women-owned business", "Female entrepreneur"]
OPEN_PHRASES = ["International", "Open to applicants worldwide", "Call-specific criteria", "Working MVP or pilot",
                "Measurable social or environmental impact", "Application fee reported"]
REGIONS = list(REGION_NAME)
ENTITIES = ["nonprofit", "for-profit", "university"]
NOUN = {"nonprofit": "nonprofit", "for-profit": "social enterprise", "university": "university research lab"}


def _grant(rng):
    rules, clauses = [], []
    if rng.random() < 0.6:
        entities = rng.choice(list(ENTITY_PHRASES))
        rules.append(("entity_type", entities))
        clauses.append(rng.choice(ENTITY_PHRASES[entities]))
    if rng.random() < 0.6:
        regions = rng.choice(list(REGION_PHRASES))
        rules.append(("region", regions))
        clauses.append(rng.choice(REGION_PHRASES[regions]))
    if rng.random() < 0.3:
        cap, label = rng.choice([(250_000, "$250k"), (500_000, "$500k"), (1_000_000, "$1M"), (5_000_000, "$5M")])
        rules.append(("annual_revenue", cap))
        clauses.append(rng.choice([f"under {label} annual revenue", f"Annual revenue under {label}"]))
    if rng.random() < 0.3:
        rules.append(("woman_led", ("yes",)))
        clauses.append(rng.choice(WOMAN_PHRASES))
    focuses = rng.sample(list(FOCUS), 2 if rng.random() < 0.25 else 1)
    clauses.append("Supports " + " and ".join(rng.choice(FOCUS[f][0]) for f in focuses))
    clauses += rng.sample(OPEN_PHRASES, rng.randint(1, 3))
    rng.shuffle(clauses)
    return "; ".join(clauses), rules, focuses


def _profile(rng):
    focus = rng.choice(list(FOCUS))
    facts = {"entity_type": rng.choices(ENTITIES, [0.5, 0.35, 0.15])[0], "region": rng.choice(REGIONS),
             "annual_revenue": float(rng.choice(PROFILE_REVENUE)), "woman_led": rng.choice(["yes", "no"])}
    for key in list(facts):  # missing means unknown, as in real intake data
        if rng.random() < 0.10:
            del facts[key]
    noun = NOUN.get(facts.get("entity_type"), "organization")
    where = f", serving communities in {REGION_NAME[facts['region']]}" if "region" in facts and rng.random() < 0.6 else ""
    text = f"A {noun} focused on {rng.choice(FOCUS[focus][1])}{where}."
    return text + "\nApplicant facts: " + json.dumps(facts), facts, focus


def _violates(facts, rules):
    for attribute, allowed in rules:
        value = facts.get(attribute)
        if value is None:
            continue
        if attribute == "annual_revenue":
            if value > allowed:
                return True
        elif value not in allowed:
            return True
    return False


def augment_rows(split, seed, n_grants=100, n_profiles=100, unrelated_per_other=3):
    """Pairs in train_pytorch's row format; unrelated pairs are subsampled to keep classes usable."""
    rng = random.Random(seed)
    grants = [_grant(rng) for _ in range(n_grants)]
    profiles = [_profile(rng) for _ in range(n_profiles)]
    rows, unrelated = [], []
    for text, facts, focus in profiles:
        for grant_text, rules, focuses in grants:
            label = ("unrelated" if focus not in focuses
                     else "ineligible" if _violates(facts, rules) else "match")
            row = {"split": split, "label": {"unrelated": LABELS[0], "ineligible": LABELS[1], "match": LABELS[2]}[label],
                   "profile_description": text, "program": grant_text, "type": "",
                   "core_requirements": "", "amount_benefit": ""}
            (unrelated if label == "unrelated" else rows).append(row)
    rng.shuffle(unrelated)
    return rows + unrelated[: unrelated_per_other * len(rows)]
