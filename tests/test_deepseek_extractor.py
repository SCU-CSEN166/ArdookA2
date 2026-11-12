import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.deepseek_extractor import (
    DeepSeekGrantExtractor,
    GrantDetailExtractionError,
    grant_source_text,
    parse_extraction,
    parse_extraction_with_report,
)
from src.pipeline import OrganizationProfile, Pipeline
from src.stores import MemoryStore


class DeepSeekExtractorTests(unittest.TestCase):
    def setUp(self):
        self.source = grant_source_text(
            {
                "program": "Example Grant",
                "deadline": "2026-12-01",
                "current_status": "Open; applications opened 2026-09-29",
                "core_requirements": "Supports community health; U.S.-based nonprofit",
                "concerns": "Closed for Austin applicants",
            }
        )
        self.extraction = {
            "funding_type": "Example Grant",
            "amount_total": None,
            "currency": None,
            "deadline": "2026-12-01",
            "target_applicants": "U.S.-based nonprofit",
            "focus_area": "Supports community health",
            "geography": "U.S.-based nonprofit",
        }

    def test_parse_accepts_only_exact_source_excerpts(self):
        response = "<think>reasoning omitted</think>\n" + json.dumps(self.extraction)
        self.assertEqual(parse_extraction(response, self.source), self.extraction)

    def test_parse_nulls_only_the_nonverbatim_field(self):
        invalid = dict(self.extraction, geography="United States")
        details, rejected = parse_extraction_with_report(json.dumps(invalid), self.source)
        self.assertEqual(details, dict(self.extraction, geography=None))
        self.assertIn("verbatim", rejected["geography"])
        self.assertEqual(set(rejected), {"geography"})

    def test_parse_rejects_text_from_the_wrong_source_line(self):
        invalid = dict(
            self.extraction,
            target_applicants="applications opened 2026-09-29",
            geography="Austin",
            focus_area="concerns: Closed",
        )
        details, rejected = parse_extraction_with_report(json.dumps(invalid), self.source)
        self.assertIsNone(details["target_applicants"])
        self.assertIsNone(details["geography"])
        self.assertIsNone(details["focus_area"])
        self.assertIn("core_requirements", rejected["target_applicants"])
        self.assertEqual(details["deadline"], "2026-12-01")

    def test_parse_tolerates_schema_drift_but_records_it(self):
        drifted = dict(self.extraction, extra="ignored", currency=["$"])
        del drifted["amount_total"]
        details, rejected = parse_extraction_with_report(json.dumps(drifted), self.source)
        self.assertEqual(set(details), set(self.extraction))
        self.assertNotIn("extra", details)
        self.assertEqual(rejected["amount_total"], "missing")
        self.assertIn("non-empty string", rejected["currency"])
        self.assertIn("extra", rejected["_extra_keys"])

    def test_parse_raises_without_a_json_object(self):
        with self.assertRaisesRegex(GrantDetailExtractionError, "JSON object"):
            parse_extraction("<think>long reasoning</think> no answer", self.source)

    def test_grant_source_text_uses_only_present_source_fields(self):
        source = grant_source_text(
            {"program": "Example Grant", "type": None, "core_requirements": "Nonprofit"}
        )
        self.assertEqual(
            source, "program: Example Grant\ncore_requirements: Nonprofit"
        )

    def test_validated_extractions_are_cached_by_model_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "details.json"
            model = type("FakeModel", (), {"device": "cpu"})()
            extractor = DeepSeekGrantExtractor(
                "test-model", cache_path, tokenizer=object(), model=model
            )
            response = json.dumps(self.extraction)
            with patch.object(extractor, "_generate", return_value=response) as generate:
                self.assertEqual(extractor.extract(self.source), self.extraction)
                self.assertEqual(extractor.extract(self.source), self.extraction)
                generate.assert_called_once_with(self.source)
            self.assertEqual(extractor.cache_hits, 1)
            self.assertEqual(extractor.model_calls, 0)

            restored = DeepSeekGrantExtractor(
                "test-model",
                cache_path,
                tokenizer=object(),
                model=model,
            )
            with patch.object(restored, "_generate", side_effect=AssertionError):
                self.assertEqual(restored.extract(self.source), self.extraction)
            self.assertEqual(restored.cache_hits, 1)

    def test_extracted_source_details_are_used_in_ranking_and_evidence(self):
        class FakeEmbedder:
            model_name = "fake"

            def encode(self, texts):
                return np.asarray(
                    [
                        [1.0, 0.0]
                        if "community health" in text.casefold()
                        else [0.0, 1.0]
                        for text in texts
                    ]
                )

        grant = {
            "source_row": 2,
            "program": "Example Grant",
            "type": "General award",
            "core_requirements": "General applicants",
            "deadline_date": pd.NaT,
            "deadline": "Rolling",
            "current_status": "Open",
            "concerns": "",
            "amount_benefit": "$10,000",
            "official_source": "https://example.org",
        }
        pipeline = Pipeline(
            pd.DataFrame([grant]),
            embedder=FakeEmbedder(),
            store=MemoryStore(),
            lexical_weight=0,
            dedupe_threshold=None,
            additional_chunks={2: ["focus_area: Supports community health"]},
            extracted_details={2: self.extraction},
        )
        recommendation = pipeline.recommend(
            OrganizationProfile("community health nonprofit"),
            min_similarity=0.2,
        )[0]
        self.assertEqual(recommendation.score, 50.0)
        self.assertEqual(
            recommendation.evidence["deepseek_extracted_details"],
            "focus_area: Supports community health",
        )
        self.assertEqual(recommendation.extracted_details, self.extraction)


if __name__ == "__main__":
    unittest.main()
