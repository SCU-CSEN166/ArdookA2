"""Partial eligibility rules transcribed from Ardook_data.xlsx, not live grant pages.

Only unambiguous requirements supported by the existing rule operators are encoded.
Evidence must still occur in the workbook; changed requirements require review.
Missing applicant facts remain unknown. These rules do not establish full eligibility.
"""

from src.pipeline import EligibilityChecker, EligibilityRule


# Omit unknown attributes. Country values use US for the United States; entity_type
# is nonprofit or for-profit (social enterprise alone does not establish either).
# Ownership is a percentage from 0 to 100; operating_months is numeric months.
# Bank-statement revenue is USD for that statement, not annual revenue.
# us_501c3_status is yes/no for the lead applicant. Geography refers to the specific
# requirement (incorporation, residence, or operations), never a generic region.
RULES = {
    "Hey Helen Grant (Visionaries)": [
        EligibilityRule("founder_residence_country", "in", ["US"], "U.S. resident"),
        EligibilityRule("incorporation_country", "in", ["US"], "U.S. for-profit"),
        EligibilityRule("entity_type", "in", ["for-profit"], "U.S. for-profit"),
        EligibilityRule("women_ownership_percent", "gte", 50, "≥50% ownership"),
    ],
    "Credibly Small Business Award": [
        EligibilityRule("business_base_country", "in", ["US"], "U.S.-based active for-profit"),
        EligibilityRule("entity_type", "in", ["for-profit"], "U.S.-based active for-profit"),
        EligibilityRule("operating_months", "gte", 6, "≥6 months operating"),
        EligibilityRule("recent_bank_statement_revenue_usd", "gte", 500, "≥$500 revenue on recent bank statement"),
    ],
    "GTIA Grants": [
        EligibilityRule("entity_type", "in", ["nonprofit"], "Nonprofit"),
    ],
    "America250 Community Futures Challenge": [
        EligibilityRule("us_501c3_status", "in", ["yes"], "Lead applicant must be eligible U.S. 501(c)(3)"),
    ],
    "NSF America’s Seed Fund (SBIR/STTR Phase I)": [
        EligibilityRule("business_base_country", "in", ["US"], "Qualifying U.S. small business"),
    ],
    "High Five Grant for Moms": [
        EligibilityRule("entity_type", "in", ["for-profit"], "≥50% ownership of for-profit"),
    ],
}


def load_checker(grants):
    """Bind rules to exact workbook program names and validate source excerpts."""
    rows = {row["program"]: row for row in grants.to_dict("records")}
    for program, rules in RULES.items():
        if program not in rows:
            raise ValueError(f"Eligibility rule program missing from workbook: {program}")
        requirements = str(rows[program]["core_requirements"])
        for rule in rules:
            if rule.evidence not in requirements:
                raise ValueError(f"Eligibility evidence changed for {program}: {rule.evidence}")
    return EligibilityChecker(RULES)
