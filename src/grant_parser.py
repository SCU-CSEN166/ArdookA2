import csv
import io
from pathlib import Path
import re

import pandas as pd

FIELDS = ("grant_id", "program", "type", "amount_benefit", "core_requirements",
          "official_source")
ALIASES = {
    "grant_id": ["grant id", "id", "#", "no", "number", "priority / deadline order"],
    "program": ["program", "grant", "grant name", "program name", "name", "title",
                "opportunity", "opportunity name"],
    "type": ["type", "program type", "grant type", "category", "kind"],
    "amount_benefit": ["amount / benefit", "amount", "benefit", "award", "funding",
                       "award amount", "prize"],
    "core_requirements": ["core requirements", "requirements", "eligibility",
                          "eligibility criteria", "criteria", "description", "details",
                          "summary", "about"],
    "official_source": ["official source", "source", "url", "link", "website"],
}


def _key(header) -> str:
    return re.sub(r"[^a-z0-9#]", "", str(header).lower())


def read_table(data: bytes, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(io.BytesIO(data), sheet_name=0, dtype=str, keep_default_na=False)
    if suffix not in (".csv", ".tsv", ".txt", ""):
        raise ValueError(f"Unsupported file type {suffix!r}; upload a .csv, .tsv or .xlsx file")
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    try:
        delimiter = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = "\t" if suffix == ".tsv" else ","
    return pd.read_csv(io.StringIO(text), sep=delimiter, dtype=str, keep_default_na=False)


def normalize_grants(raw: pd.DataFrame) -> pd.DataFrame:
    by_key = {}
    for column in raw.columns:
        by_key.setdefault(_key(column), column)
    mapping = {}
    for field, names in ALIASES.items():
        for name in names:
            column = by_key.get(_key(name))
            if column is not None and column not in mapping.values():
                mapping[field] = column
                break
    if "program" not in mapping and "core_requirements" not in mapping:
        raise ValueError("Could not find a grant name or requirements column. Found columns: "
                         + ", ".join(map(str, raw.columns)))

    grants = pd.DataFrame({field: raw[mapping[field]] if field in mapping else ""
                           for field in FIELDS})
    for field in FIELDS:
        grants[field] = (grants[field].fillna("").astype(str)
                         .str.replace(r"\s+", " ", regex=True).str.strip())
    grants = grants[(grants[list(FIELDS[1:])] != "").any(axis=1)].reset_index(drop=True)
    if grants.empty:
        raise ValueError("The file has no non-empty grant rows")

    numbers = pd.Series(range(1, len(grants) + 1), dtype=str)
    grants["grant_id"] = grants["grant_id"].where(grants["grant_id"] != "", numbers)
    grants["grant_id"] = grants["grant_id"].str.replace(r"\.0$", "", regex=True)
    grants["program"] = grants["program"].where(grants["program"] != "", "Grant " + numbers)
    return grants


def parse_grant_file(data: bytes, filename: str) -> pd.DataFrame:
    return normalize_grants(read_table(data, filename))


def grant_from_text(text: str, name: str = "") -> pd.DataFrame:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise ValueError("Grant description is empty")
    name = name.strip() or (text[:60] + ("..." if len(text) > 60 else ""))
    return pd.DataFrame([{"grant_id": "1", "program": name, "type": "", "amount_benefit": "",
                          "core_requirements": text, "official_source": ""}])
