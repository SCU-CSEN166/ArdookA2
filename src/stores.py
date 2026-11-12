"""Evaluation helpers shared by the evaluators: an in-memory store and the keyword baseline."""

import hashlib
import re

import numpy as np

TOKEN = re.compile(r"[a-z0-9]+")


class KeywordEmbedder:
    """Deterministic token-overlap vectors, the baseline the semantic model must beat."""

    model_name = "keyword-hash-4096"

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), 4096))
        for row, text in enumerate(texts):
            for token in TOKEN.findall(text.casefold()):
                index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big") % 4096
                vectors[row, index] += 1
        return vectors


class MemoryStore:
    """Vectors held in memory only, so evaluation never writes an index to disk."""

    def __init__(self):
        self.vectors = {}

    def child(self, suffix: str) -> "MemoryStore":
        return MemoryStore()

    def sync(self, ids, documents, embed, embedder, *, read_only=False) -> int:
        if read_only:
            raise RuntimeError("MemoryStore has no persisted index to serve read-only")
        vectors = np.asarray(embed(list(documents)))
        self.vectors = dict(zip(ids, vectors))
        return len(ids)

    def embeddings(self, ids) -> np.ndarray:
        if not ids:
            return np.empty((0, 0))
        return np.asarray([self.vectors[id_] for id_ in ids])
