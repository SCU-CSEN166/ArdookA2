"""Frontend adapter for the saved match classifier (results/ardook_match_v1/model.joblib).

Loads the model once (no training, no downloads) and scores grants for the provisional Ardook
profile with the same code path as `python -m src.predict_match --ardook`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.match_classifier import ARDOOK_PROFILE, MatchClassifier
from src.predict_match import DEFAULT_MODEL, score


class ModelUnavailable(RuntimeError):
    """The saved model or its local sentence encoder cannot be loaded."""


def load_model(path: Path = DEFAULT_MODEL) -> MatchClassifier:
    """Load the saved classifier and its offline encoder; never trains or downloads."""
    path = Path(path)
    if not path.is_file():
        raise ModelUnavailable(f"Saved model not found at {path}. Restore "
                               "results/ardook_match_v1/model.joblib from the repository.")
    try:
        model = MatchClassifier.load(path)
    except Exception as error:  # unpickling errors vary by version mismatch
        raise ModelUnavailable(f"Could not load the saved model at {path}: {error}") from error
    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        try:
            encoder.encode(["warm-up"])
        except Exception as error:  # OSError when the encoder is not in the local HF cache
            raise ModelUnavailable(
                f"The local sentence encoder for the model ({encoder.name}) is not available "
                f"offline: {error}. Cache it once with network access, then restart.") from error
    return model


def predict(model: MatchClassifier, grants: pd.DataFrame) -> pd.DataFrame:
    """Score parsed grants for the Ardook profile; one row per grant, input order kept."""
    profile = ARDOOK_PROFILE
    result = score(model, profile["description"], profile["attributes"], grants)
    result["match"] = result["predicted"] == "match"
    return result


def results_csv(predictions: pd.DataFrame, threshold: float) -> str:
    """CSV of the shown predictions; `label` keeps the Relevant/Not relevant scorer format."""
    return pd.DataFrame({
        "grant_id": predictions["grant_id"],
        "program": predictions["program"],
        "label": predictions["match"].map({True: "Relevant", False: "Not relevant"}),
        "predicted": predictions["predicted"],
        "p_match": predictions["p_match"],
        "threshold": threshold,
        "reason": [f"Model score {p:.4f} {'>=' if m else '<'} threshold {threshold}"
                   for p, m in zip(predictions["p_match"], predictions["match"])],
    }).to_csv(index=False)
