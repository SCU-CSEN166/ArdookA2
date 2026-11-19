import json

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch

from src.compare_models import LABELS, fit_mlp, load_partitions

ENCODER = "all-MiniLM-L6-v2"
GRANT_FIELDS = ("program", "type", "core_requirements", "amount_benefit")
REASONS = {
    "match": "Shared focus and no known rule violation",
    "ineligible": "Shared focus, but a requirement likely rules this applicant out",
    "unrelated": "No shared focus with the organization",
}


def organization_text(description: str, facts: dict) -> str:
    facts = {k: v for k, v in facts.items() if v not in (None, "")}
    return description.strip() + "\nApplicant facts: " + json.dumps(facts, sort_keys=True)


def grant_text(grant) -> str:
    return "\n".join(str(grant.get(field, "") or "") for field in GRANT_FIELDS)


class PytorchGrantClassifier:
    def __init__(self, encoder_name: str = ENCODER):
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(4)
        self.encoder = SentenceTransformer(f"sentence-transformers/{encoder_name}",
                                           device="cpu", local_files_only=True)
        partitions = load_partitions()
        pairs = {s: [(r["organization"], r["grant"]) for r in partitions[s]]
                 for s in ("train", "validation")}
        labels = {s: np.array([LABELS.index(r["label"]) for r in partitions[s]])
                  for s in ("train", "validation")}
        raw = self._pair_features({s: p for s, p in pairs.items()})
        self.scaler = StandardScaler().fit(raw["train"])
        features = {s: self.scaler.transform(x).astype(np.float32) for s, x in raw.items()}
        self.model, self.best_epoch, _ = fit_mlp(features, labels)

    def _pair_features(self, pairs_by_split: dict) -> dict:
        unique = sorted({text for pairs in pairs_by_split.values() for pair in pairs for text in pair})
        vectors = self.encoder.encode(unique, batch_size=32, normalize_embeddings=True,
                                      convert_to_numpy=True, show_progress_bar=False)
        lookup = dict(zip(unique, vectors))
        out = {}
        for split, pairs in pairs_by_split.items():
            org = np.stack([lookup[p[0]] for p in pairs])
            grant = np.stack([lookup[p[1]] for p in pairs])
            out[split] = np.concatenate([org, grant, org * grant, abs(org - grant)], axis=1)
        return out

    def predict(self, organization: str, grants: list) -> list[dict]:
        if not grants:
            return []
        pairs = [(organization, grant_text(g)) for g in grants]
        x = self.scaler.transform(self._pair_features({"x": pairs})["x"]).astype(np.float32)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(torch.from_numpy(x)), dim=1).numpy()
        results = []
        for row in probabilities:
            name = LABELS[int(row.argmax())]
            results.append({
                "label": "Relevant" if name == "match" else "Not relevant",
                "predicted_class": name,
                "reason": REASONS[name],
                "match_probability": float(row[LABELS.index("match")]),
                "probabilities": {label: float(p) for label, p in zip(LABELS, row)},
            })
        return results
