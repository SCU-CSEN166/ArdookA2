"""Frontend inference: the Streamlit app scores with the saved classifier exactly like the CLI."""

import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd

from src import app_predict
from src.app_predict import ModelUnavailable, load_model, predict, results_csv
from src.grant_parser import grant_from_text, parse_grant_file
from src.match_classifier import Encoder, MatchClassifier, TwoStageClassifier
from src.paths import REPO_ROOT
from src.predict_match import DEFAULT_MODEL, main as predict_main
from src.score_hidden_test import read_labels

HIDDEN = REPO_ROOT / "Ardook_hidden_test_grants.xlsx"
HAS_STREAMLIT = importlib.util.find_spec("streamlit") is not None


def cli(*args: str) -> pd.DataFrame:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        predict_main(["--ardook", *args])
    return pd.read_csv(io.StringIO(stdout.getvalue()))


def no_training(*_args, **_kwargs):
    raise AssertionError("the frontend must never train a model")


@unittest.skipUnless(DEFAULT_MODEL.exists(), "saved model not present")
class SavedModelInFrontend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with mock.patch.object(MatchClassifier, "fit", no_training), \
                mock.patch.object(TwoStageClassifier, "fit", no_training):
            cls.model = load_model()
        cls.table = pd.read_excel(HIDDEN, dtype=str, keep_default_na=False)

    def upload(self, data: bytes, name: str) -> pd.DataFrame:
        with mock.patch.object(MatchClassifier, "fit", no_training), \
                mock.patch.object(TwoStageClassifier, "fit", no_training):
            return predict(self.model, parse_grant_file(data, name))

    def test_saved_model_and_threshold_are_used(self):
        self.assertIsInstance(self.model, TwoStageClassifier)
        self.assertEqual(self.model.threshold, 0.25)

    def test_csv_tsv_xlsx_uploads_match_cli(self):
        expected = cli("--grants", str(HIDDEN))
        uploads = {"grants.csv": self.table.to_csv(index=False).encode(),
                   "grants.tsv": self.table.to_csv(index=False, sep="\t").encode(),
                   "grants.xlsx": HIDDEN.read_bytes()}
        for name, data in uploads.items():
            with self.subTest(name):
                result = self.upload(data, name)
                self.assertEqual(list(result.program), list(expected.program))
                self.assertEqual(list(result.p_match.round(4)), list(expected.p_match))
                self.assertEqual(list(result.predicted), list(expected.predicted))

    def test_real_tracker_upload_matches_cli(self):
        expected = cli()
        workbook = REPO_ROOT / "Ardook_data.xlsx"
        result = self.upload(workbook.read_bytes(), workbook.name)
        self.assertEqual(len(result), 37)
        self.assertEqual(list(result.p_match), list(expected.p_match))

    def test_pasted_grant_matches_cli_inline_grant(self):
        text = ("Women-led AI startups in the U.S. building technology for social impact; "
                "early-stage companies welcome.")
        grant = grant_from_text(text).iloc[0]
        result = predict(self.model, grant_from_text(text))
        expected = cli("--grant-json", pd.Series({"program": grant.program,
                                                   "core_requirements": text}).to_json())
        self.assertEqual(len(result), 1)
        self.assertEqual(result.p_match.iloc[0], expected.p_match.iloc[0])
        self.assertEqual(result.predicted.iloc[0], expected.predicted.iloc[0])

    def test_stored_threshold_not_argmax(self):
        result = self.upload(HIDDEN.read_bytes(), HIDDEN.name)
        self.assertEqual(list(result.match), list(result.p_match >= 0.25))
        between = result[(result.p_match >= 0.25) & (result.p_match < 0.5)]
        self.assertFalse(between.empty, "need a 0.25-0.5 score to tell the threshold from argmax")
        self.assertTrue(between.match.all())

    def test_download_csv_reflects_predictions(self):
        result = self.upload(HIDDEN.read_bytes(), HIDDEN.name)
        csv = results_csv(result, self.model.threshold)
        frame = pd.read_csv(io.StringIO(csv))
        self.assertEqual(list(frame.columns), ["grant_id", "program", "label", "predicted",
                                               "p_match", "threshold", "reason"])
        self.assertEqual(list(frame.grant_id.astype(str)), list(self.table["Grant ID"]))
        self.assertEqual(list(frame.p_match), list(result.p_match))
        self.assertEqual(list(frame.predicted), list(result.predicted))
        self.assertTrue((frame.threshold == 0.25).all())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "out.csv"
            path.write_text(csv, encoding="utf-8")
            labels = read_labels(path)  # the hidden-test scorer still reads it
        self.assertEqual(list(labels.values()), list(result.match))


class InvalidInputsAndResources(unittest.TestCase):
    def test_malformed_and_empty_inputs_raise_clear_errors(self):
        cases = {"empty.csv": b"", "headers.csv": b"Program,Core Requirements\n",
                 "blank.csv": b"Program,Core Requirements\n , \n",
                 "other.csv": b"colour,size\nred,big\n", "grants.pdf": b"%PDF"}
        for name, data in cases.items():
            with self.subTest(name), self.assertRaises(Exception) as caught:
                parse_grant_file(data, name)
            self.assertTrue(str(caught.exception))
        with self.assertRaises(ValueError):
            grant_from_text("   \n ")

    def test_missing_model_file(self):
        with self.assertRaises(ModelUnavailable) as caught:
            load_model(Path("/nonexistent/model.joblib"))
        self.assertIn("not found", str(caught.exception))

    def test_unreadable_model_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.joblib"
            path.write_bytes(b"not a model")
            with self.assertRaises(ModelUnavailable):
                load_model(path)

    @unittest.skipUnless(DEFAULT_MODEL.exists(), "saved model not present")
    def test_missing_local_encoder_does_not_train_or_download(self):
        with mock.patch.object(Encoder, "encode", side_effect=OSError("not in local cache")), \
                mock.patch.object(TwoStageClassifier, "fit", no_training), \
                self.assertRaises(ModelUnavailable) as caught:
            load_model()
        self.assertIn("offline", str(caught.exception))

    def test_frontend_never_imports_training_code(self):
        for path in (REPO_ROOT / "app.py", REPO_ROOT / "src/app_predict.py"):
            source = path.read_text()
            for banned in ("train_", "PairingGrantClassifier", "run_hidden_test", ".fit("):
                self.assertNotIn(banned, source, f"{path.name} mentions {banned}")


@unittest.skipUnless(HAS_STREAMLIT and DEFAULT_MODEL.exists(), "streamlit or saved model missing")
class StreamlitApp(unittest.TestCase):
    def run_app(self, text: str):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(REPO_ROOT / "app.py"), default_timeout=180)
        app.run()
        app.segmented_control[0].set_value("Paste grant").run()
        app.text_area[0].input(text).run()
        return app

    def setUp(self):
        import streamlit as st
        st.cache_resource.clear()

    def test_pasted_grant_shows_model_result(self):
        text = "Women-led AI startups in the U.S. building technology for social impact."
        app = self.run_app(text)
        self.assertFalse(app.exception)
        score = predict(load_model(), grant_from_text(text)).iloc[0]
        self.assertEqual(app.subheader[0].value, "Match" if score.match else "Not a match")
        self.assertIn(f"Model score {score.p_match:.2f}", app.caption[0].value)

    def test_missing_model_shows_error_without_training(self):
        def unavailable():
            raise ModelUnavailable("Saved model not found at x. Restore it.")
        with mock.patch.object(app_predict, "load_model", unavailable), \
                mock.patch.object(TwoStageClassifier, "fit", no_training):
            app = self.run_app("Any grant text")
        self.assertFalse(app.exception)
        self.assertIn("Saved model not found", app.error[0].value)
        self.assertEqual(len(app.subheader), 0)


if __name__ == "__main__":
    unittest.main()
