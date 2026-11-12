"""Persistent embedding store: vectors are computed once and only changed rows are re-embedded.

Chroma is the durable cache. Scoring is done by the pipeline with an exact matrix product over the
loaded vectors, not an approximate top-k query: eligibility filtering happens *after* retrieval, so
an approximate cut-off could drop grants that belong in the final list.
"""

import hashlib
import os
from pathlib import Path
from typing import Callable

import numpy as np

from src.paths import REPO_ROOT

STORE_DIR_ENV = "ARDOOK_STORE_DIR"

Embed = Callable[[list[str]], np.ndarray]
BATCH = 500  # stays under Chroma's max batch size and SQLite's variable limit


def resolve_store_path(path: str | os.PathLike) -> str:
    """Anchor relative store paths to the repo (or ARDOOK_STORE_DIR) so the cwd never matters."""
    path = Path(path)
    if path.is_absolute():
        return str(path)
    base = os.environ.get(STORE_DIR_ENV)
    base = Path(base).expanduser().resolve() if base else REPO_ROOT
    return str(base / path)


def content_hash(document: str, embedder: str) -> str:
    """Identify a row's vector: it changes if the text or the embedding model changes."""
    return hashlib.sha256(f"{embedder}\0{document}".encode()).hexdigest()


class IndexOutOfDateError(RuntimeError):
    """A read-only store was opened against an index that needs (re)building."""


class VectorStore:
    def __init__(self, path: str = "vector_store", name: str = "grants"):
        import chromadb

        self.path = resolve_store_path(path)
        self.name = name
        self.client = chromadb.PersistentClient(path=self.path)
        self.collection = self._create()

    def _create(self):
        return self.client.get_or_create_collection(self.name, metadata={"hnsw:space": "cosine"})

    def child(self, suffix: str) -> "VectorStore":
        """A sibling collection in the same index directory (used for chunk vectors)."""
        return VectorStore(path=self.path, name=f"{self.name}_{suffix}")

    def sync(
        self,
        ids: list[str],
        documents: list[str],
        embed: Embed,
        embedder: str,
        *,
        read_only: bool = False,
    ) -> int:
        """Bring the store in line with (ids, documents); embed only new or changed rows.

        Returns the number of rows that had to be embedded. A different embedding model rebuilds
        the collection, because a collection locks in its vector size on first insert.

        New vectors are computed before anything is deleted, so a failed embedding (for example a
        model that cannot load) leaves the existing index untouched. With read_only=True nothing is
        ever written: an out-of-date index raises IndexOutOfDateError so serving processes never
        race the ingest job.
        """
        if len(ids) != len(documents):
            raise ValueError("Expected one document per id")
        if len(set(ids)) != len(ids):
            raise ValueError("Store ids must be unique")

        stored = self.collection.get(include=["metadatas"])
        saved = {id_: metadata or {} for id_, metadata in zip(stored["ids"], stored["metadatas"])}
        rebuild = any(metadata.get("embedder") != embedder for metadata in saved.values())
        if rebuild:
            saved = {}

        wanted = set(ids)
        removed = [id_ for id_ in stored["ids"] if rebuild or id_ not in wanted]
        hashes = [content_hash(document, embedder) for document in documents]
        stale = [
            index for index, (id_, digest) in enumerate(zip(ids, hashes))
            if saved.get(id_, {}).get("hash") != digest
        ]
        if not stale and not removed:
            return 0
        if read_only:
            raise IndexOutOfDateError(
                f"Index '{self.name}' at {self.path} is out of date "
                f"({len(stale)} rows to embed, {len(removed)} to remove); run the ingest job first."
            )

        vectors = np.asarray(embed([documents[i] for i in stale]), dtype=np.float32) if stale else None
        if vectors is not None and len(vectors) != len(stale):
            raise ValueError("Expected one embedding per document")

        if rebuild:
            self.client.delete_collection(self.name)
            self.collection = self._create()
        elif removed:
            for start in range(0, len(removed), BATCH):
                self.collection.delete(ids=removed[start:start + BATCH])
        for start in range(0, len(stale), BATCH):
            part = stale[start:start + BATCH]
            self.collection.upsert(
                ids=[ids[i] for i in part],
                documents=[documents[i] for i in part],
                embeddings=vectors[start:start + BATCH].tolist(),
                metadatas=[{"embedder": embedder, "hash": hashes[i]} for i in part],
            )
        return len(stale)

    def embeddings(self, ids: list[str]) -> np.ndarray:
        """Return stored embeddings in the requested ID order."""
        if not ids:
            return np.empty((0, 0))
        stored = {}
        for start in range(0, len(ids), BATCH):
            result = self.collection.get(ids=ids[start:start + BATCH], include=["embeddings"])
            stored.update(zip(result["ids"], result["embeddings"]))
        missing = [id_ for id_ in ids if id_ not in stored]
        if missing:
            raise KeyError(f"Index '{self.name}' is missing vectors for ids {missing[:5]}")
        return np.asarray([stored[id_] for id_ in ids], dtype=np.float32)
