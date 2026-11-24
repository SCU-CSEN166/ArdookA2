"""Shared inference: frozen config, requirement filter override, filter-only facts, v2 splits."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd

from src.app_predict import load_model, predict, results_csv
from src.match_classifier import MatchClassifier, TwoStageClassifier
from src.paths import REPO_ROOT
from src.predict_match import DEFAULT_CONFIG, load_inference_model, main as predict_main, score

CONFIG = json.loads(DEFAULT_CONFIG.read_text()) if DEFAULT_CONFIG.exists() else None
ARMENIA_ONLY = {"program": "Women in AI Prize (Armenia)", "type": "Grant",
                "core_requirements": "Women-led AI startups; based in Armenia only",
                "amount_benefit": "$10,000"}


def no_training(*_args, **_kwargs):
    raise AssertionError("inference must never train a model")


def cli(*args: str) -> pd.DataFrame:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        predict_main(list(args))
    return pd.read_csv(io.StringIO(stdout.getvalue()))


@unittest.skipUnless(CONFIG and (REPO_ROOT / CONFIG["model"]).exists(), "inference config or model missing")
class ConfiguredInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with mock.patch.object(MatchClassifier, "fit", no_training), \
                mock.patch.object(TwoStageClassifier, "fit", no_training):
            cls.model = load_inference_model()

    def test_saved_v2_model_reloads_with_the_reported_hidden_scores(self):
        expected = pd.read_csv(REPO_ROOT / "results/ardook_match_v2/hidden_test_predictions.csv")
        result = cli("--grants", str(REPO_ROOT / "Ardook_hidden_test_grants.xlsx"), "--ardook")
        self.assertEqual(list(result.p_match), list(expected.p_match))
        self.assertEqual(list(result.predicted), list(expected.final_prediction))

    def test_threshold_and_filter_come_from_the_frozen_config(self):
        self.assertEqual(self.model.threshold, CONFIG["threshold"])
        self.assertEqual(self.model.requirement_filter, bool(CONFIG["filter"]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps({**CONFIG, "threshold": 0.6, "filter": None}))
            other = load_inference_model(path)
        self.assertEqual((other.threshold, other.requirement_filter), (0.6, False))

    def test_cli_and_app_agree_on_every_output_column(self):
        grants = pd.DataFrame([{"grant_id": 1, **ARMENIA_ONLY},
                               {"grant_id": 2, "program": "Open AI grant", "type": "Grant",
                                "core_requirements": "Women-led AI startups", "amount_benefit": ""}])
        app = predict(load_model(), grants)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "grants.csv"
            grants.rename(columns={"program": "Program", "type": "Type", "core_requirements": "Core Requirements",
                                   "amount_benefit": "Amount / Benefit", "grant_id": "Grant ID"}).to_csv(path, index=False)
            command = cli("--ardook", "--grants", str(path))
        for column in command.columns:
            with self.subTest(column=column):
                self.assertEqual(list(command[column].fillna("")), list(app[column].fillna("")))

    def test_failed_requirement_overrides_a_model_match_with_evidence(self):
        model = load_inference_model()
        model.requirement_filter = True
        model.threshold = 0.0  # force a model match so the override is visible
        result = score(model, "A U.S. woman-led AI startup.", {"region": "US", "woman_led": "yes"},
                       pd.DataFrame([{"grant_id": 1, **ARMENIA_ONLY}]))
        row = result.iloc[0]
        self.assertEqual((row.model_prediction, row.requirement_check, row.predicted), ("match", "fail", "not_match"))
        self.assertIn("based in Armenia only", row.requirement_evidence)
        result["match"] = result.predicted == "match"
        csv = pd.read_csv(io.StringIO(results_csv(result, 0.0))).iloc[0]
        self.assertEqual(csv.label, "Not relevant")
        self.assertIn("stated requirement is not met", csv.reason)
        self.assertEqual(csv.p_match, row.p_match)  # the learned score is shown as is

    def test_filter_is_opt_in_on_the_cli(self):
        grant = json.dumps(ARMENIA_ONLY)
        plain = cli("--ardook", "--grant-json", grant)
        filtered = cli("--ardook", "--grant-json", grant, "--requirement-filter")
        self.assertEqual(list(plain.p_match), list(filtered.p_match))
        self.assertEqual(bool(CONFIG["filter"]), "requirement_check" in plain.columns)
        self.assertEqual(filtered.requirement_check.iloc[0], "fail")
        self.assertEqual(filtered.predicted.iloc[0], "not_match")

    def test_unknown_facts_never_override_and_filter_only_facts_skip_the_model(self):
        grant = pd.DataFrame([{"grant_id": 1, "program": "Seed grant", "type": "Grant",
                               "core_requirements": "Registered nonprofit; company 6 months–5 years",
                               "amount_benefit": ""}])
        model = load_inference_model()
        model.requirement_filter = True
        base = score(model, "An AI startup.", {"region": "US"}, grant).iloc[0]
        self.assertEqual(base.requirement_check, "unknown")
        self.assertEqual(base.predicted, base.model_prediction)
        young = score(model, "An AI startup.", {"region": "US", "operating_months": 2}, grant).iloc[0]
        self.assertEqual(young.p_match, base.p_match)
        self.assertEqual(young.requirement_check, "fail")
        with self.assertRaises(ValueError):
            score(model, "x", {"operating_months": -1}, grant)


class TuningProvenance(unittest.TestCase):
    def test_tuning_reuses_v1_splits_and_never_reads_messy_data(self):
        source = (REPO_ROOT / "src/tune_match_classifier.py").read_text()
        self.assertIn("split_manifest.json", source)
        for module in ("src/tune_match_classifier.py", "src/eligibility_filter.py", "src/predict_match.py"):
            self.assertNotIn("messy", (REPO_ROOT / module).read_text())

    def test_filter_reads_no_grant_names_or_labels(self):
        source = (REPO_ROOT / "src/eligibility_filter.py").read_text().casefold()
        for leak in ("fit", "tory burch", "women who tech", "aws", "expected_labels", "program"):
            self.assertNotIn(f'"{leak}', source)


if __name__ == "__main__":
    unittest.main()
