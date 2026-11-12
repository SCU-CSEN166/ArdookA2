"""Small dependency-free BM25 (Okapi) so exact-term matches can complement dense retrieval."""

import math
import re
from collections import Counter

import numpy as np

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in into is it its of on or that the their this to "
    "was we with our your who which will can not no".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.casefold()) if t not in STOPWORDS]


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tokens = [Counter(tokenize(d)) for d in documents]
        self.lengths = np.array([sum(c.values()) for c in self.tokens], dtype=float)
        self.avg_length = float(self.lengths.mean()) if len(documents) else 0.0
        n = len(documents)
        df = Counter(term for c in self.tokens for term in c)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of the query against every document (zeros when nothing matches)."""
        out = np.zeros(len(self.tokens))
        if not len(self.tokens) or self.avg_length == 0:
            return out
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avg_length)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([c.get(term, 0) for c in self.tokens], dtype=float)
            out += idf * tf * (self.k1 + 1) / (tf + norm)
        return out

    def normalized(self, query: str) -> np.ndarray:
        """Scores scaled to [0, 1] by the best match for this query (all zeros if none)."""
        s = self.scores(query)
        top = s.max() if len(s) else 0.0
        return s / top if top > 0 else s
