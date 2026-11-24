"""Frontend adapter for the configured match classifier (results/ardook_match_v2/inference_config.json).

Loads the model once (no training, no downloads) with the same frozen threshold and requirement
filter as `python -m src.predict_match --ardook`, and scores grants for the provisional Ardook profile.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.match_classifier import ARDOOK_PROFILE, MatchClassifier
from src.predict_match import DEFAULT_CONFIG, load_inference_model, score


class ModelUnavailable(RuntimeError):
    """The saved model or its local sentence encoder cannot be loaded."""


def load_model(path: Path | None = None, config: Path = DEFAULT_CONFIG) -> MatchClassifier:
    """Load the configured classifier and its offline encoder; never trains or downloads."""
    try:
        model = load_inference_model(config, path)
    except FileNotFoundError as error:
        raise ModelUnavailable(f"Saved model not found at {error.filename or path}. Restore the "
                               "saved model from the repository.") from error
    except Exception as error:  # unpickling errors vary by version mismatch
        raise ModelUnavailable(f"Could not load the saved model: {error}") from error
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


def reason(row: pd.Series, threshold: float) -> str:
    """Why a row got its result: the model score first, then any failed stated requirement."""
    text = f"Model score {row['p_match']:.4f} {'>=' if row['p_match'] >= threshold else '<'} threshold {threshold}"
    if row.get("requirement_check") == "fail":
        return f"{text}; not a match because a stated requirement is not met: {row['requirement_evidence']}"
    if row["match"]:
        return f"{text}; eligibility not verified"
    return text


def results_csv(predictions: pd.DataFrame, threshold: float) -> str:
    """CSV of the shown predictions; `label` keeps the Relevant/Not relevant scorer format."""
    frame = pd.DataFrame({
        "grant_id": predictions["grant_id"],
        "program": predictions["program"],
        "label": predictions["match"].map({True: "Relevant", False: "Not relevant"}),
        "predicted": predictions["predicted"],
        "p_match": predictions["p_match"],
        "threshold": threshold,
        "reason": [reason(row, threshold) for _, row in predictions.iterrows()],
    })
    if "requirement_check" in predictions:
        frame["model_prediction"] = predictions["model_prediction"]
        frame["requirement_check"] = predictions["requirement_check"]
    return frame.to_csv(index=False)
