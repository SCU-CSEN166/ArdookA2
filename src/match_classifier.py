"""Binary applicant-to-grant match classifier: data, label mapping, features and model.

Protocol and label definitions: results/ardook_match_protocol.md. Training script:
src/train_match_classifier.py. Inference CLI: src/predict_match.py.
Only the clean synthetic database and Ardook_data.xlsx are read; never the messy data.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.ardook_extra_synthetic_data import load_grants as load_synthetic_grants
from src.ardook_extra_synthetic_data import load_matches, load_profiles
from src.ingest import load_grants as load_real_grants
from src.ranking_classifier import PairFeatures

LABELS = ("not_match", "match")
SEED = 42
GRANT_FIELDS = ("program", "type", "core_requirements", "amount_benefit")
# Columns that must never reach features: human judgements, advice, labels, identifiers.
FORBIDDEN_FIELDS = ("fit", "why_it_fits", "concerns", "recommended_action", "applied",
                    "applied_bool", "official_source", "Focus", "label", "source_row",
                    "priority_order", "deadline", "current_status")
ARDOOK_PROFILE = {
    "description": ("Ardook is an early-stage, woman-led U.S. startup building AI, physical-AI "
                    "and data products with a social-impact mission."),
    # Only facts stated in the README sentence; entity type and revenue stay unknown.
    "attributes": {"region": "US", "woman_led": "yes"},
}
REAL_POSITIVE = {"Excellent", "Strong"}
REAL_NEGATIVE = {"Moderate", "Weak–Moderate"}
FACT_VALUES = {
    "entity_type": ("nonprofit", "for-profit", "university"),
    "region": ("US", "EU", "UK", "India", "Africa"),
    "woman_led": ("yes", "no"),
}
ENCODERS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "bge": "BAAI/bge-large-en-v1.5",
}


def real_fit_label(fit: str | None) -> str | None:
    """Map a tracker Fit rating to the target; None means ambiguous (never scored)."""
    fit = _field(fit).strip()
    if fit in REAL_POSITIVE:
        return "match"
    if fit in REAL_NEGATIVE:
        return "not_match"
    return None


def _field(value) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def grant_text(grant: dict) -> str:
    return "\n".join(_field(grant.get(f)) for f in GRANT_FIELDS)


def applicant_text(description: str, attributes: dict) -> str:
    return description + "\nApplicant facts: " + json.dumps(attributes, sort_keys=True)


def grant_family(program: str) -> str:
    """Group related real programmes so they never straddle dev and test."""
    return "google" if "google" in program.casefold() else program


def _normalized_requirements(text: str) -> str:
    return re.sub(r"[\d,.]+", "#", text.casefold()).strip()


@dataclass(frozen=True)
class Pair:
    applicant: str          # description only
    attributes: dict
    grant: dict             # allowlisted grant fields only
    label: str | None
    source: str             # "synthetic" or "real"
    split: str
    original_label: str     # synthetic 3-way label or tracker Fit, for reporting only
    key: str                # e.g. "p12-g40" or "real-row5", reporting only

    @property
    def texts(self) -> tuple[str, str]:
        return applicant_text(self.applicant, self.attributes), grant_text(self.grant)


def _group_split(keys: list[str], seed: int = SEED) -> dict[str, str]:
    unique = sorted(set(keys))
    train, rest = train_test_split(unique, test_size=0.4, random_state=seed)
    validation, test = train_test_split(rest, test_size=0.5, random_state=seed)
    return {k: s for s, group in (("train", train), ("validation", validation), ("test", test))
            for k in group}


def synthetic_pairs() -> tuple[list[Pair], int]:
    """Pairs whose applicant group and grant group fall in the same split; plus excluded count."""
    profiles = load_profiles().set_index("id")
    grants = load_synthetic_grants().set_index("source_row")
    org_split = _group_split(profiles.description.tolist())
    grant_groups = {int(i): _normalized_requirements(r.core_requirements) for i, r in grants.iterrows()}
    grant_split = _group_split(list(grant_groups.values()))
    pairs, excluded = [], 0
    for row in load_matches().itertuples(index=False):
        profile = profiles.loc[int(row.profile_id)]
        split = org_split[profile.description]
        if split != grant_split[grant_groups[int(row.source_row)]]:
            excluded += 1
            continue
        grant = {f: _field(grants.loc[int(row.source_row), f]) for f in GRANT_FIELDS}
        pairs.append(Pair(profile.description, dict(profile.attributes), grant,
                          "match" if row.label == "match" else "not_match", "synthetic", split,
                          row.label, f"p{int(row.profile_id)}-g{int(row.source_row)}"))
    return pairs, excluded


def real_pairs(seed: int = SEED) -> list[Pair]:
    """All 37 Ardook pairs; split is dev, test or ambiguous (grouped by family, stratified)."""
    grants = load_real_grants()
    labels = grants.fit.map(real_fit_label)
    families = grants.program.map(grant_family)
    labelled = pd.DataFrame({"family": families, "label": labels}).dropna()
    # Stratify families by their majority label so both classes reach the test split.
    family_label = labelled.groupby("family").label.agg(lambda s: s.mode().iloc[0])
    dev, test = train_test_split(sorted(family_label.index), test_size=1 / 3, random_state=seed,
                                 stratify=[family_label[f] for f in sorted(family_label.index)])
    split_of = {**{f: "dev" for f in dev}, **{f: "test" for f in test}}
    pairs = []
    for row, label, family in zip(grants.to_dict("records"), labels, families):
        split = "ambiguous" if label is None else split_of[family]
        pairs.append(Pair(ARDOOK_PROFILE["description"], dict(ARDOOK_PROFILE["attributes"]),
                          {f: _field(row[f]) for f in GRANT_FIELDS}, label, "real", split,
                          _field(row["fit"]), f"real-row{row['source_row']}"))
    return pairs


# ---------------------------------------------------------------- features

def fact_features(attributes: list[dict]) -> np.ndarray:
    """One-hot structured facts with an explicit 'unknown' column; revenue as log + unknown."""
    columns = []
    for key, values in FACT_VALUES.items():
        for value in values + ("unknown",):
            columns.append([float((a.get(key) or "unknown") == value) for a in attributes])
    revenue = [a.get("annual_revenue") for a in attributes]
    columns.append([np.log10(1 + r) if r is not None else 0.0 for r in revenue])
    columns.append([float(r is None) for r in revenue])
    return np.array(columns, dtype=np.float32).T


def clauses(requirements: str) -> list[str]:
    parts = [c.strip() for c in re.split(r"[;\n]", requirements or "") if c.strip()]
    return parts or [requirements or ""]


_VECTORS: dict[str, dict[str, np.ndarray]] = {}
_MODELS: dict = {}


class Encoder:
    """Frozen sentence encoder (local files only, CPU); vectors cached per process, not pickled."""

    def __init__(self, name: str):
        if name not in ENCODERS:
            raise ValueError(f"Unknown encoder {name}")
        self.name = name

    def encode(self, texts: list[str]) -> np.ndarray:
        cache = _VECTORS.setdefault(self.name, {})
        missing = sorted({t for t in texts if t not in cache})
        if missing:
            if self.name not in _MODELS:
                from sentence_transformers import SentenceTransformer
                _MODELS[self.name] = SentenceTransformer(ENCODERS[self.name], device="cpu",
                                                         local_files_only=True)
            vectors = _MODELS[self.name].encode(missing, batch_size=32, normalize_embeddings=True,
                                                convert_to_numpy=True, show_progress_bar=False)
            cache.update(zip(missing, vectors))
        return np.stack([cache[t] for t in texts])


class TfidfPairFeatures:
    kind = "tfidf"

    def fit(self, pairs: list[Pair]):
        self.vectorizer = PairFeatures().fit([p.texts for p in pairs])
        return self

    def transform(self, pairs: list[Pair]):
        return self.vectorizer.transform([p.texts for p in pairs])


class EmbeddingPairFeatures:
    """[a, g, a*g, |a-g|] of frozen embeddings plus similarities and structured facts."""
    kind = "embedding"

    def __init__(self, encoder: str):
        self.encoder = Encoder(encoder)

    def _raw(self, pairs: list[Pair]) -> np.ndarray:
        a = self.encoder.encode([p.applicant for p in pairs])
        g = self.encoder.encode([grant_text(p.grant) for p in pairs])
        return np.hstack([a, g, a * g, np.abs(a - g), self._similarities(pairs, a, g),
                          fact_features([p.attributes for p in pairs])])

    def _similarities(self, pairs, a, g) -> np.ndarray:
        clause_sims = []
        for p, vector in zip(pairs, a):
            c = self.encoder.encode(clauses(p.grant["core_requirements"]))
            sims = c @ vector
            clause_sims.append((sims.max(), sims.mean()))
        return np.column_stack([(a * g).sum(1), np.array(clause_sims)]).astype(np.float32)

    def fit(self, pairs: list[Pair]):
        self.scaler = StandardScaler().fit(self._raw(pairs))
        return self

    def transform(self, pairs: list[Pair]) -> np.ndarray:
        return self.scaler.transform(self._raw(pairs)).astype(np.float32)


class SimilarityBagFeatures(EmbeddingPairFeatures):
    """Similarities + facts + binary word indicators, for a tree model to find interactions."""
    kind = "simbag"

    def _raw_parts(self, pairs: list[Pair]):
        a = self.encoder.encode([p.applicant for p in pairs])
        g = self.encoder.encode([grant_text(p.grant) for p in pairs])
        dense = np.hstack([self._similarities(pairs, a, g), fact_features([p.attributes for p in pairs])])
        return dense, [p.applicant for p in pairs], [grant_text(p.grant) for p in pairs]

    def fit(self, pairs: list[Pair]):
        _, applicants, grants = self._raw_parts(pairs)
        self.applicant_words = CountVectorizer(binary=True, max_features=200, min_df=2).fit(applicants)
        self.grant_words = CountVectorizer(binary=True, ngram_range=(1, 2), max_features=400,
                                           min_df=2).fit(grants)
        return self

    def transform(self, pairs: list[Pair]) -> np.ndarray:
        dense, applicants, grants = self._raw_parts(pairs)
        return hstack([csr_matrix(dense), self.applicant_words.transform(applicants),
                       self.grant_words.transform(grants)], format="csr").toarray().astype(np.float32)


def make_features(spec: str):
    if spec == "tfidf":
        return TfidfPairFeatures()
    kind, encoder = spec.split(":")
    return {"embedding": EmbeddingPairFeatures, "simbag": SimilarityBagFeatures}[kind](encoder)


def make_estimator(name: str):
    if name == "logreg":
        return LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000, random_state=SEED)
    if name == "hgb":
        return HistGradientBoostingClassifier(learning_rate=0.1, max_iter=300, max_leaf_nodes=31,
                                              class_weight="balanced", random_state=SEED)
    raise ValueError(f"Unknown estimator {name}")


class MatchClassifier:
    """Feature pipeline + estimator + decision threshold on P(match)."""

    def __init__(self, features: str, estimator: str, threshold: float = 0.5):
        self.features_spec, self.estimator_name, self.threshold = features, estimator, threshold

    def fit(self, pairs: list[Pair], sample_weight: np.ndarray | None = None):
        if any(p.label not in LABELS for p in pairs):
            raise ValueError("Training pairs must be labelled match or not_match")
        self.features = make_features(self.features_spec).fit(pairs)
        self.estimator = make_estimator(self.estimator_name)
        y = np.array([LABELS.index(p.label) for p in pairs])
        self.estimator.fit(self.features.transform(pairs), y, sample_weight=sample_weight)
        return self

    def predict_proba(self, pairs: list[Pair]) -> np.ndarray:
        """Columns follow LABELS: P(not_match), P(match)."""
        return self.estimator.predict_proba(self.features.transform(pairs))

    def predict(self, pairs: list[Pair]) -> list[str]:
        return [LABELS[int(p >= self.threshold)] for p in self.predict_proba(pairs)[:, 1]]

    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "MatchClassifier":
        model = joblib.load(path)
        if not isinstance(model, MatchClassifier):
            raise ValueError(f"{path} is not a MatchClassifier")
        return model


WORD = re.compile(r"[a-z][a-z0-9\-']+|\$?\d[\d,]*k?|\$\d+m")
_HASHER = FeatureHasher(2**20, input_type="string", alternate_sign=False)


def _words(text: str) -> set[str]:
    return set(WORD.findall(text.casefold()))


def _fact_tokens(attributes: dict) -> set[str]:
    return {f"fact:{k}={attributes.get(k, 'unknown')}" for k in (*FACT_VALUES, "annual_revenue")}


def relatedness_tokens(pair: Pair) -> list[str]:
    """Applicant-word x grant-word pairs: learns which wordings describe the same focus."""
    grant_words = _words(grant_text(pair.grant))
    return [f"{a}|{g}" for a in _words(pair.applicant) for g in grant_words]


def conflict_tokens(pair: Pair) -> list[str]:
    """Applicant-fact x requirement-word pairs: learns which requirements a fact violates."""
    requirement_words = _words(pair.grant["core_requirements"])
    return ([f"{f}|{w}" for f in _fact_tokens(pair.attributes) for w in requirement_words]
            + [f"req:{w}" for w in requirement_words])


class TwoStageClassifier(MatchClassifier):
    """P(match) = P(shared focus) x P(no requirement conflict | shared focus).

    Stage targets come from the synthetic 3-way labels (unrelated vs related; ineligible vs
    match among related). Real Ardook grants are all topically curated, so each counts as
    related, with its binary target used for the second stage.
    """

    RELATED_C, CONFLICT_C = 0.1, 1.0  # chosen on synthetic validation only

    def _relatedness_x(self, pairs):
        x = _HASHER.transform(relatedness_tokens(p) for p in pairs)
        if self.encoder is None:
            return x
        a = self.encoder.encode([p.applicant for p in pairs])
        g = self.encoder.encode([grant_text(p.grant) for p in pairs])
        sims = EmbeddingPairFeatures._similarities(self, pairs, a, g)
        return hstack([x, csr_matrix(self.sim_scaler.transform(sims))], format="csr")

    def fit(self, pairs: list[Pair], sample_weight: np.ndarray | None = None):
        if any(p.label not in LABELS for p in pairs):
            raise ValueError("Training pairs must be labelled match or not_match")
        weight = np.ones(len(pairs)) if sample_weight is None else np.asarray(sample_weight)
        name = self.features_spec.split(":")[1]
        self.encoder = None if name == "none" else Encoder(name)
        if self.encoder is not None:
            a = self.encoder.encode([p.applicant for p in pairs])
            g = self.encoder.encode([grant_text(p.grant) for p in pairs])
            self.sim_scaler = StandardScaler().fit(EmbeddingPairFeatures._similarities(self, pairs, a, g))
        related = np.array([p.source == "real" or p.original_label != "unrelated" for p in pairs])
        self.related_model = LogisticRegression(C=self.RELATED_C, class_weight="balanced",
                                                max_iter=3000, random_state=SEED)
        self.related_model.fit(self._relatedness_x(pairs), related.astype(int), sample_weight=weight)
        subset = [p for p, r in zip(pairs, related) if r]
        self.conflict_model = LogisticRegression(C=self.CONFLICT_C, class_weight="balanced",
                                                 max_iter=3000, random_state=SEED)
        self.conflict_model.fit(_HASHER.transform(conflict_tokens(p) for p in subset),
                                np.array([p.label == "match" for p in subset]).astype(int),
                                sample_weight=weight[related])
        return self

    def stage_probabilities(self, pairs: list[Pair]) -> tuple[np.ndarray, np.ndarray]:
        related = self.related_model.predict_proba(self._relatedness_x(pairs))[:, 1]
        no_conflict = self.conflict_model.predict_proba(
            _HASHER.transform(conflict_tokens(p) for p in pairs))[:, 1]
        return related, no_conflict

    def predict_proba(self, pairs: list[Pair]) -> np.ndarray:
        related, no_conflict = self.stage_probabilities(pairs)
        match = related * no_conflict
        return np.column_stack([1 - match, match])


def build_classifier(features: str, estimator: str) -> MatchClassifier:
    cls = TwoStageClassifier if features.startswith("twostage:") else MatchClassifier
    return cls(features, estimator)


def make_pair(description: str, attributes: dict, grant: dict, key: str = "input") -> Pair:
    """Build an unlabelled pair for inference; unknown grant fields become empty text."""
    unknown = set(attributes) - set(FACT_VALUES) - {"annual_revenue"}
    if unknown:
        raise ValueError(f"Unsupported applicant facts: {sorted(unknown)}; "
                         f"use {sorted(FACT_VALUES) + ['annual_revenue']}")
    for key_, value in attributes.items():
        if key_ in FACT_VALUES and value not in FACT_VALUES[key_]:
            raise ValueError(f"{key_} must be one of {FACT_VALUES[key_]}, got {value!r}")
    if "annual_revenue" in attributes:
        attributes = {**attributes, "annual_revenue": float(attributes["annual_revenue"])}
    if not _field(grant.get("program")).strip():
        raise ValueError("Each grant needs a program name")
    clean = {f: _field(grant.get(f)) for f in GRANT_FIELDS}
    return Pair(description, attributes, clean, None, "input", "input", "", key)
