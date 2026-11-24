"""Requirement filter: check a grant's Core Requirements text against stated applicant facts.

Written from general requirement language (no grant names). Each recognised clause yields a
`Check` with status `pass`, `fail` or `unknown`; a missing applicant fact is always `unknown`.
Hedged clauses ("typically", "call-specific", ...) are skipped. Social impact is never read as
nonprofit status, and woman-led is never read as a percentage or majority ownership.
No deadline or open/closed checks. Protocol: results/ardook_match_v2_protocol.md.

Applicant facts read: entity_type (nonprofit, for-profit, university), region (US, EU, UK,
India, Africa), annual_revenue (USD), woman_led (yes, no), operating_months (number).
"""

from __future__ import annotations

from dataclasses import dataclass
import re

FILTER_VERSION = "requirement-filter-1"
FILTER_FACTS = ("entity_type", "region", "annual_revenue", "woman_led", "operating_months")

HEDGE = re.compile(r"\b(typically|usually|often|may|varies|vary|depending|depends|"
                   r"call-specific|program-specific|programme-specific|edition-specific|"
                   r"round-specific|geography/round-specific)\b")
NEGATION = re.compile(r"\b(not eligible|not open to|ineligible|excluding|excluded|except|"
                      r"cannot apply|are not|is not|must not)\b")

ENTITY_TERMS = {
    "nonprofit": re.compile(r"\bnon-?profits?\b|501\s*\(c\)\s*\(3\)|\bcharit(?:y|ies)\b|\bngos?\b|"
                            r"civil society organi[sz]ations?|\bcsos?\b"),
    "for-profit": re.compile(r"\bfor[- ]?profits?\b"),
    "university": re.compile(r"\buniversit(?:y|ies)\b|university-affiliated|academic institutions?"),
}
# Wording that admits entity types the clause does not name (a social enterprise can be either).
ENTITY_OPEN = re.compile(r"social enterprises?|any (?:legal )?entity|\bindividuals?\b")

REGION_TERMS = {
    "US": re.compile(r"(?<![a-z])u\.s\.(?:a\.)?|(?<![a-z])usa(?![a-z])|united states"),
    "EU": re.compile(r"horizon europe|european union|(?<![a-z])eu(?![a-z])|(?<![a-z])europe(?:an)?(?![a-z])"),
    "UK": re.compile(r"(?<![a-z])uk(?![a-z])|united kingdom"),
    "India": re.compile(r"(?<![a-z])india(?![a-z])"),
    "Africa": re.compile(r"(?<![a-z])(?:sub-saharan )?africa(?![a-z])"),
}
# Places outside the applicant vocabulary; any applicant region listed above fails them.
OTHER_PLACES = re.compile(r"(?<![a-z])(armenia|canada|australia|latin america|asia)(?![a-z])")
# U.S. states (Georgia is skipped: it is also a country). The applicant's state is never known.
US_STATES = re.compile(r"(?<![a-z])(california|texas|nebraska|new york|nyc|florida|illinois|"
                       r"massachusetts|washington state|colorado|ohio|michigan)(?![a-z])")
LOCATION_CUE = re.compile(r"based in|-based|\bbased\b|operating in|operate in|registered in|"
                          r"-registered|incorporated in|located in|\bresidents?\b|headquartered|"
                          r"formed and operating|\bcountry\b|\bcountries\b|\bprovinces?\b|\bonly\b|"
                          r"citizens?|serving [^;]* in")
EXCLUDED_PLACE = re.compile(r"(?:non-|outside (?:of )?(?:the )?)$")

MONEY = re.compile(r"(?:(usd|us\$|\$|€|£|₹)\s*)?(\d[\d,]*(?:\.\d+)?)\s*(k|m|million|thousand)?\b")
LOWER_BOUND = re.compile(r"(at least|≥|>=|minimum(?: of)?|more than|over|exceeding|above)\s*$")
UPPER_BOUND = re.compile(r"(under|below|less than|no more than|up to|≤|<=|maximum(?: of)?|"
                         r"not exceeding|not more than|within(?: the past)?)\s*$")
RANGE_SEP = r"\s*(?:–|-|to)\s*"

WOMAN_LED = re.compile(r"\bwom[ae]n[- ]led\b|\bwom[ae]n(?:/non-binary)? (?:founders?|entrepreneurs?)\b|"
                       r"\bfemale (?:founders?|entrepreneurs?)\b|\bwoman (?:founder|co-founder|ceo)\b|"
                       r"(?:founded|co-founded) by a woman|\bwoman entrepreneur\b")
OWNERSHIP = re.compile(r"\bowned\b|\bownership\b|\bequity\b|%")
WORD_NUMBERS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}
DURATION = r"(\d+(?:\.\d+)?|" + "|".join(WORD_NUMBERS) + r")\s*(months?|years?)"


@dataclass(frozen=True)
class Check:
    rule: str        # entity_type, region, annual_revenue, woman_led, ownership, operating_months
    status: str      # pass, fail or unknown
    clause: str      # the requirement text this came from
    detail: str      # what was required and what the applicant fact was


def clauses(requirements: str) -> list[str]:
    return [c.strip() for c in re.split(r"[;\n]", requirements or "") if c.strip()]


def _status(ok: bool | None) -> str:
    return "unknown" if ok is None else "pass" if ok else "fail"


def _entity(clause: str, text: str, facts: dict) -> Check | None:
    named = {e for e, pattern in ENTITY_TERMS.items() if pattern.search(text)}
    if not named:
        return None
    value = facts.get("entity_type")
    if NEGATION.search(text):
        ok = None if value is None else value not in named
        return Check("entity_type", _status(ok), clause, f"excludes {sorted(named)}; applicant {value}")
    open_ended = bool(ENTITY_OPEN.search(text))
    if value is None:
        ok = None
    elif value in named:
        ok = True
    else:
        ok = None if open_ended else False
    return Check("entity_type", _status(ok), clause, f"requires one of {sorted(named)}"
                 f"{' or similar' if open_ended else ''}; applicant {value}")


def _places(text: str) -> tuple[set[str], set[str]]:
    """(required, excluded) places named in a clause."""
    required, excluded = set(), set()
    found = [(m.start(), r) for r, p in REGION_TERMS.items() for m in p.finditer(text)]
    found += [(m.start(), "other:" + m.group(1)) for m in OTHER_PLACES.finditer(text)]
    found += [(m.start(), "us-state") for m in US_STATES.finditer(text)]
    for start, place in found:
        (excluded if EXCLUDED_PLACE.search(text[:start]) else required).add(place)
    return required, excluded


def _region(clause: str, text: str, facts: dict) -> Check | None:
    if not LOCATION_CUE.search(text):
        return None
    required, excluded = _places(text)
    if not required and not excluded:
        return None
    value = facts.get("region")
    if value is None:
        ok = None
    elif value in excluded:
        ok = False
    elif not required:
        ok = True
    elif value in required:
        ok = True
    elif value == "US" and "us-state" in required:
        ok = None  # a U.S. state is named; the applicant's state is unknown
    else:
        ok = False
    return Check("region", _status(ok), clause,
                 f"requires {sorted(required) or 'any'}, excludes {sorted(excluded) or 'none'}; applicant {value}")


def _usd(match: re.Match, index: int) -> float | None:
    """The index-th USD amount in a MONEY match; None for bare numbers and other currencies."""
    currency, number, unit = match.groups()[index * 3:(index + 1) * 3]
    unit = (unit or "").lower()
    if number is None or (not currency and not unit):
        return None  # a bare number (a year, a count) is not an amount
    if currency and currency.lower() not in ("usd", "us$", "$"):
        return None  # other currencies are not compared with USD revenue
    return float(number.replace(",", "")) * {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6}.get(unit, 1)


def _bounds(text: str, unit_pattern: str, parse) -> tuple[float | None, float | None]:
    """(minimum, maximum) from range, lower-bound and upper-bound wording."""
    range_match = re.search(unit_pattern + RANGE_SEP + unit_pattern, text)
    if range_match:
        low, high = parse(range_match, 0), parse(range_match, 1)
        if low is not None and high is not None:
            return low, high
    low = high = None
    for match in re.finditer(unit_pattern, text):
        value = parse(match, 0)
        before = text[:match.start()]
        if value is None:
            continue
        if LOWER_BOUND.search(before):
            low = value
        elif UPPER_BOUND.search(before):
            high = value
    return low, high


def _revenue(clause: str, text: str, facts: dict) -> Check | None:
    if "revenue" not in text or re.search(r"bank statement|raised|funding|pre-revenue", text):
        return None
    low, high = _bounds(text, MONEY.pattern, _usd)
    if low is None and high is None:
        return None
    value = facts.get("annual_revenue")
    ok = None if value is None else (low is None or value >= low) and (high is None or value <= high)
    return Check("annual_revenue", _status(ok), clause,
                 f"requires revenue between {low or 0:,.0f} and {high if high is not None else 'any'}; "
                 f"applicant {value}")


def _woman(clause: str, text: str, facts: dict) -> list[Check]:
    checks = []
    led = bool(WOMAN_LED.search(text))
    owned = bool(OWNERSHIP.search(text)) and bool(re.search(r"wom[ae]n|female|mother", text))
    value = facts.get("woman_led")
    if led:
        # "..., or majority woman-owned": ownership is an alternative, not an extra condition.
        alternative = owned and bool(re.search(
            r"\bor\s+(?:majority\s+|\d+%\s+)?(?:wom[ae]n[- ]owned|majority|owned)", text))
        if value is None:
            ok = None
        elif value == "yes":
            ok = True
        else:
            ok = None if alternative else False
        checks.append(Check("woman_led", _status(ok), clause, f"requires woman-led; applicant {value}"))
        if alternative and value == "yes":
            return checks  # woman-led already satisfies one of the alternatives
    if owned:
        # Ownership share is not an applicant fact we hold; woman-led does not establish it.
        checks.append(Check("ownership", "unknown", clause, "ownership share not stated by the applicant"))
    return checks


def _months(match: re.Match, index: int) -> float | None:
    number, unit = match.groups()[index * 2:(index + 1) * 2]
    if number is None:
        return None
    value = WORD_NUMBERS.get(number)
    value = float(number) if value is None else float(value)
    return value * (12 if unit.startswith("year") else 1)


def _age(clause: str, text: str, facts: dict) -> Check | None:
    if not re.search(r"operat|founded|registered|incorporated|company|organi[sz]ations?|business|old\b|age\b",
                     text):
        return None
    low, high = _bounds(text, DURATION, _months)
    if low is None and high is None:
        return None
    value = facts.get("operating_months")
    ok = None if value is None else (low is None or value >= low) and (high is None or value <= high)
    return Check("operating_months", _status(ok), clause,
                 f"requires {low or 0:g}-{high if high is not None else 'any'} months operating; applicant {value}")


def check_requirements(requirements: str, facts: dict) -> list[Check]:
    """Every recognised requirement in the text, checked against the applicant's facts."""
    facts = {k: v for k, v in facts.items() if v is not None and v != ""}
    checks = []
    for clause in clauses(requirements):
        text = clause.casefold()
        if HEDGE.search(text):
            continue
        for found in (_entity(clause, text, facts), _region(clause, text, facts),
                      _revenue(clause, text, facts), _age(clause, text, facts)):
            if found is not None:
                checks.append(found)
        checks += _woman(clause, text, facts)
    return checks


def overall(checks: list[Check]) -> str:
    """fail if any check fails; unknown if any is unknown; pass if all pass; none if no checks."""
    statuses = {c.status for c in checks}
    for status in ("fail", "unknown", "pass"):
        if status in statuses:
            return status
    return "none"


def explain(checks: list[Check], status: str) -> str:
    """Short evidence text for the checks with the given status."""
    return "; ".join(f"{c.rule}: \"{c.clause}\"" for c in checks if c.status == status)
