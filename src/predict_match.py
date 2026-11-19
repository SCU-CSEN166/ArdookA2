"""Score applicant-to-grant pairs with the saved match classifier.

Examples:
  # Ardook (provisional README profile) against every grant in Ardook_data.xlsx
  python -m src.predict_match --ardook
  # Any applicant against a grants file (.xlsx/.csv with Program, Type, Core Requirements,
  # Amount / Benefit columns; other columns are ignored)
  python -m src.predict_match --description "A U.S. nonprofit ..." \
      --attributes '{"entity_type": "nonprofit", "region": "US"}' --grants grants.csv
  # One grant given inline
  python -m src.predict_match --description "..." --grant-json '{"program": "X", "core_requirements": "..."}'

Only Program, Type, Core Requirements and Amount / Benefit are read from grants; Fit,
notes and advice columns are ignored. Output is CSV (stdout or --output).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from src.ingest import COLUMNS
from src.match_classifier import (ARDOOK_PROFILE, GRANT_FIELDS, LABELS, MatchClassifier,
                                  TwoStageClassifier, make_pair)
from src.paths import DEFAULT_WORKBOOK, REPO_ROOT

DEFAULT_MODEL = REPO_ROOT / "results/ardook_match_v1/model.joblib"
ID_COLUMNS = ("Grant ID", "grant_id", "id")


def read_grants(path: Path) -> pd.DataFrame:
    """Read a grants table; only the allowlisted fields (and an optional ID) are kept."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        sheets = pd.read_excel(path, sheet_name=None, keep_default_na=False)
        frame = sheets.get("Grant Tracker", next(iter(sheets.values())))
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, keep_default_na=False)
    else:
        raise ValueError("Grants file must be .xlsx or .csv")
    frame = frame.rename(columns=lambda c: COLUMNS.get(str(c).strip(), str(c).strip()))
    if "program" not in frame.columns:
        raise ValueError("Grants file needs a Program column")
    id_column = next((c for c in ID_COLUMNS if c in frame.columns), None)
    kept = frame[[c for c in GRANT_FIELDS if c in frame.columns]].copy()
    kept = kept[kept["program"].astype(str).str.strip() != ""]
    kept.insert(0, "grant_id", frame.loc[kept.index, id_column] if id_column else kept.index + 2)
    return kept.reset_index(drop=True)


def score(model: MatchClassifier, description: str, attributes: dict, grants: pd.DataFrame) -> pd.DataFrame:
    pairs = [make_pair(description, attributes, row, key=str(row["grant_id"]))
             for row in grants.to_dict("records")]
    probabilities = model.predict_proba(pairs)
    result = pd.DataFrame({"grant_id": grants["grant_id"], "program": grants["program"],
                           f"p_{LABELS[0]}": probabilities[:, 0].round(4),
                           f"p_{LABELS[1]}": probabilities[:, 1].round(4),
                           "predicted": [LABELS[int(p >= model.threshold)] for p in probabilities[:, 1]]})
    if isinstance(model, TwoStageClassifier):
        related, no_conflict = model.stage_probabilities(pairs)
        result["p_shared_focus"] = related.round(4)
        result["p_no_requirement_conflict"] = no_conflict.round(4)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    who = parser.add_mutually_exclusive_group(required=True)
    who.add_argument("--ardook", action="store_true",
                     help="use the provisional Ardook profile and, unless --grants is given, Ardook_data.xlsx")
    who.add_argument("--description", help="applicant description")
    parser.add_argument("--attributes", default="{}",
                        help='JSON facts: entity_type, region, annual_revenue, woman_led (omit unknowns)')
    grants_from = parser.add_mutually_exclusive_group()
    grants_from.add_argument("--grants", type=Path, help=".xlsx or .csv grants table")
    grants_from.add_argument("--grant-json", help="one grant as JSON with program/type/core_requirements/amount_benefit")
    parser.add_argument("--output", type=Path, help="write CSV here instead of stdout")
    args = parser.parse_args(argv)
    try:
        model = MatchClassifier.load(args.model)
        if args.ardook:
            description, attributes = ARDOOK_PROFILE["description"], ARDOOK_PROFILE["attributes"]
        else:
            description, attributes = args.description, json.loads(args.attributes)
            if not isinstance(attributes, dict):
                raise ValueError("--attributes must be a JSON object")
        if args.grant_json:
            grants = pd.DataFrame([{"grant_id": "input", **json.loads(args.grant_json)}])
        elif args.grants or args.ardook:
            grants = read_grants(args.grants or DEFAULT_WORKBOOK)
        else:
            raise ValueError("Give --grants or --grant-json")
        result = score(model, description, attributes, grants)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.output:
        result.to_csv(args.output, index=False)
        print(f"Wrote {len(result)} predictions to {args.output}", file=sys.stderr)
    else:
        result.to_csv(sys.stdout, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
