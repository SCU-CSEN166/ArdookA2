"""Requirement filter: negation, numeric bounds, unknown facts and wording it must not over-read."""

import unittest

from src.eligibility_filter import check_requirements, overall


def statuses(requirements, facts, rule=None):
    return [c.status for c in check_requirements(requirements, facts) if rule in (None, c.rule)]


REVENUE_CASES = [
    ("Under $500k annual revenue", 400_000, "pass"),
    ("Under $500k annual revenue", 800_000, "fail"),
    ("At least USD 75,000 in annual revenue", 50_000, "fail"),
    ("At least USD 75,000 in annual revenue", 75_000, "pass"),
    ("$10k–$500k 2025 or trailing-12-month revenue", 5_000, "fail"),
    ("$10k–$500k 2025 or trailing-12-month revenue", 200_000, "pass"),
    ("Under $1M annual revenue", None, "unknown"),
]


class RequirementFilterTest(unittest.TestCase):
    def test_entity_requirement(self):
        for facts, expected in (({"entity_type": "nonprofit"}, "pass"),
                                ({"entity_type": "for-profit"}, "fail"), ({}, "unknown")):
            with self.subTest(facts=facts):
                self.assertEqual(overall(check_requirements("Registered nonprofit; defined MVP", facts)), expected)

    def test_entity_alternatives_and_open_wording(self):
        self.assertEqual(statuses("For-profit or nonprofit venture worldwide", {"entity_type": "nonprofit"}), ["pass"])
        # "social enterprises" may be for-profit, so a for-profit applicant is not ruled out
        self.assertEqual(statuses("Nonprofits/social enterprises/academic institutions", {"entity_type": "for-profit"},
                        "entity_type"), ["unknown"])

    def test_entity_negation(self):
        self.assertEqual(statuses("Nonprofits are not eligible", {"entity_type": "nonprofit"}), ["fail"])
        self.assertEqual(statuses("Nonprofits are not eligible", {"entity_type": "for-profit"}), ["pass"])

    def test_social_impact_is_not_nonprofit(self):
        self.assertEqual(check_requirements("Measurable social/economic/environmental impact", {"entity_type": "for-profit"}), [])

    def test_region_requirement_needs_a_location_cue(self):
        self.assertEqual(statuses("U.S.-based organization", {"region": "US"}), ["pass"])
        self.assertEqual(statuses("Based in an eligible Horizon Europe country", {"region": "US"}), ["fail"])
        self.assertEqual(statuses("Groups based in Aragatsotn Province, Armenia, only", {"region": "US"}, "region"), ["fail"])
        # where past finalists came from is not a residence requirement
        self.assertEqual(statuses("Women-led tech startups (past finalists from the U.S., Europe and Asia)",
                        {"region": "UK", "woman_led": "yes"}, "region"), [])

    def test_region_negation_and_us_state(self):
        self.assertEqual(statuses("Open to non-U.S. based companies", {"region": "US"}), ["fail"])
        self.assertEqual(statuses("Open to non-U.S. based companies", {"region": "UK"}), ["pass"])
        self.assertEqual(statuses("Small businesses based in California only", {"region": "US"}), ["unknown"])
        self.assertEqual(statuses("Small businesses based in California only", {"region": "India"}), ["fail"])

    def test_revenue_bounds(self):
        for requirements, revenue, expected in REVENUE_CASES:
            with self.subTest(requirements=requirements, revenue=revenue):
                facts = {} if revenue is None else {"annual_revenue": revenue}
                self.assertEqual(statuses(requirements, facts, "annual_revenue"), [expected])

    def test_funding_and_bank_statement_amounts_are_not_annual_revenue(self):
        self.assertEqual(statuses("No more than USD 750K in outside funding; pre-revenue welcome",
                        {"annual_revenue": 0}, "annual_revenue"), [])
        self.assertEqual(statuses("≥$500 revenue on recent bank statement", {"annual_revenue": 0}, "annual_revenue"), [])

    def test_woman_led_is_not_ownership(self):
        # a woman-led applicant still has an unverified ownership share
        checks = check_requirements("Founded or co-founded by a woman who holds at least 33% equity",
                                    {"woman_led": "yes"})
        self.assertEqual([(c.rule, c.status) for c in checks], [("woman_led", "pass"), ("ownership", "unknown")])
        self.assertEqual(statuses("Women-owned business", {"woman_led": "yes"}), ["unknown"])
        # ownership offered as an alternative: woman-led satisfies it, not woman-led stays open
        either = "Woman founder, co-founder or CEO, or majority woman-owned"
        self.assertEqual(overall(check_requirements(either, {"woman_led": "yes"})), "pass")
        self.assertEqual(overall(check_requirements(either, {"woman_led": "no"})), "unknown")
        self.assertEqual(statuses("Women-led; defined MVP", {"woman_led": "no"}), ["fail"])

    def test_operating_age_bounds_and_word_numbers(self):
        rule = "operating_months"
        self.assertEqual(statuses("Company 6 months–5 years", {"operating_months": 3}, rule), ["fail"])
        self.assertEqual(statuses("Company 6 months–5 years", {"operating_months": 24}, rule), ["pass"])
        self.assertEqual(statuses("Operating for less than seven years", {"operating_months": 90}, rule), ["fail"])
        self.assertEqual(statuses("Startups founded within the past 10 years", {}, rule), ["unknown"])
        self.assertEqual(statuses("8 weeks in London", {"operating_months": 1}, rule), [])

    def test_hedged_clauses_are_skipped_and_missing_facts_are_unknown(self):
        self.assertEqual(check_requirements("Typically nonprofit/digital-impact ecosystem; call-specific", {"entity_type": "for-profit"}), [])
        self.assertEqual(overall(check_requirements("Registered nonprofit; U.S.-based organization", {})), "unknown")
        self.assertEqual(overall(check_requirements("Defined MVP or pilot", {"region": "US"})), "none")


if __name__ == "__main__":
    unittest.main()
