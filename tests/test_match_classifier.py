import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.match_classifier import (GRANT_FIELDS, LABELS, MatchClassifier, TwoStageClassifier,
                                  grant_family, make_pair, real_fit_label, real_pairs,
                                  synthetic_pairs)
from src.paths import REPO_ROOT
from src.predict_match import DEFAULT_MODEL, main as predict_main
from src.train_match_classifier import leakage_checks

NEW_MODULES = ("src/match_classifier.py", "src/train_match_classifier.py", "src/predict_match.py")


class LabelsAndSplits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.synthetic, cls.excluded = synthetic_pairs()
        cls.real = real_pairs()

    def test_real_fit_mapping_keeps_conditional_ratings_ambiguous(self):
        self.assertEqual(real_fit_label("Excellent"), "match")
        self.assertEqual(real_fit_label("Strong"), "match")
        self.assertEqual(real_fit_label("Moderate"), "not_match")
        self.assertEqual(real_fit_label("Weak–Moderate"), "not_match")
        for fit in ("Strong if India impact", "Excellent only if nonprofit", "Moderate–Strong", None, ""):
            self.assertIsNone(real_fit_label(fit))

    def test_real_split_counts_and_family_grouping(self):
        splits = pd.Series([p.split for p in self.real]).value_counts().to_dict()
        self.assertEqual(len(self.real), 37)
        self.assertEqual(splits["ambiguous"], 9)
        self.assertEqual(splits["dev"] + splits["test"], 28)
        google = {p.split for p in self.real if grant_family(p.grant["program"]) == "google" and p.label}
        self.assertEqual(len(google), 1)
        test_labels = {p.label for p in self.real if p.split == "test"}
        self.assertEqual(test_labels, set(LABELS))

    def test_leakage_checks_pass_and_splits_cover_all_classes(self):
        checks = leakage_checks(self.synthetic, self.real)
        self.assertTrue(checks["synthetic_applicant_description_groups_disjoint"])
        self.assertTrue(checks["real_families_disjoint_dev_test"])
        for split in ("train", "validation", "test"):
            labels = {p.original_label for p in self.synthetic if p.split == split}
            self.assertEqual(labels, {"unrelated", "ineligible", "match"})
        self.assertEqual(len(self.synthetic) + self.excluded, 30000)

    def test_only_allowlisted_grant_fields_reach_pairs(self):
        for pair in self.synthetic[:50] + self.real:
            self.assertEqual(set(pair.grant), set(GRANT_FIELDS))

    def test_new_code_never_reads_messy_data(self):
        for module in NEW_MODULES:
            self.assertNotIn("ardook_messy", (REPO_ROOT / module).read_text().casefold())


class ModelAndCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        synthetic, _ = synthetic_pairs()
        train = [p for p in synthetic if p.split == "train"][::4]
        cls.model = TwoStageClassifier("twostage:none", "logreg").fit(train)
        cls.validation = [p for p in synthetic if p.split == "validation"][:200]

    def test_probabilities_are_valid(self):
        probabilities = self.model.predict_proba(self.validation)
        self.assertEqual(probabilities.shape, (200, 2))
        np.testing.assert_allclose(probabilities.sum(1), 1.0)
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())

    def test_save_load_gives_identical_probabilities(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            self.model.save(path)
            reloaded = MatchClassifier.load(path)
        np.testing.assert_array_equal(reloaded.predict_proba(self.validation),
                                      self.model.predict_proba(self.validation))

    def test_forbidden_columns_do_not_change_predictions(self):
        grant = {"program": "X Fund", "type": "Grant", "core_requirements": "Registered nonprofit; Supports AI for social good",
                 "amount_benefit": "$10,000"}
        leaky = {**grant, "fit": "Excellent", "why_it_fits": "perfect", "concerns": "none"}
        a = make_pair("A nonprofit building AI for disaster response.", {"entity_type": "nonprofit"}, grant)
        b = make_pair("A nonprofit building AI for disaster response.", {"entity_type": "nonprofit"}, leaky)
        np.testing.assert_array_equal(self.model.predict_proba([a]), self.model.predict_proba([b]))

    def test_make_pair_rejects_unknown_facts(self):
        with self.assertRaises(ValueError):
            make_pair("x", {"entity_type": "llc"}, {"program": "X"})
        with self.assertRaises(ValueError):
            make_pair("x", {"ownership": 50}, {"program": "X"})

    def test_cli_scores_inline_grant_and_grants_file(self):
        with tempfile.TemporaryDirectory() as folder:
            model_path = Path(folder) / "model.joblib"
            self.model.save(model_path)
            grants = Path(folder) / "grants.csv"
            pd.DataFrame([{"Program": "A", "Type": "Grant", "Core Requirements": "Supports youth education",
                           "Amount / Benefit": "$5,000", "Fit": "Excellent"},
                          {"Program": "B", "Type": "Prize", "Core Requirements": "For-profit company",
                           "Amount / Benefit": "", "Fit": "Weak"}]).to_csv(grants, index=False)
            out = Path(folder) / "out.csv"
            predict_main(["--model", str(model_path), "--description", "A nonprofit tutoring students.",
                          "--attributes", '{"entity_type": "nonprofit", "region": "US"}',
                          "--grants", str(grants), "--output", str(out)])
            result = pd.read_csv(out)
            self.assertEqual(list(result.program), ["A", "B"])
            self.assertTrue(set(result.predicted) <= set(LABELS))
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                predict_main(["--model", str(model_path), "--ardook", "--grant-json",
                              '{"program": "Women in AI Prize", "core_requirements": "Women-led AI startup"}'])
            self.assertIn("p_match", stdout.getvalue())


@unittest.skipUnless(DEFAULT_MODEL.exists(), "trained model not present")
class SavedModel(unittest.TestCase):
    def test_saved_model_scores_all_real_tracker_grants(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            predict_main(["--ardook"])
        result = pd.read_csv(io.StringIO(stdout.getvalue()))
        self.assertEqual(len(result), 37)
        expected = pd.read_csv(REPO_ROOT / "results/ardook_match_v1/ardook_predictions.csv")
        np.testing.assert_allclose(result.p_match.values, expected.p_match_final_model.values, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
