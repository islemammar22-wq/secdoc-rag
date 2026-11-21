"""Small BM25 keyword search (no extra dependencies).

Why: embeddings are weak at exact strings such as "Safeguard 1.2".
BM25 matches the literal words, so we combine both (see rag.py).
"""
import math
import re
from collections import Counter

import numpy as np

# Keeps dotted numbers together: "1.2" and "v8.1.2" stay single tokens.
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")


def tokenize(text):
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        docs = [tokenize(t) for t in texts]
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.tf = [Counter(d) for d in docs]
        self.doc_len = np.array([len(d) for d in docs], dtype="float32")
        self.avg_len = float(self.doc_len.mean()) if self.n else 1.0
        df = Counter()
        for tf in self.tf:
            df.update(tf.keys())
        self.idf = {
            w: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for w, c in df.items()
        }

    def scores(self, query):
        out = np.zeros(self.n, dtype="float32")
        for w in set(tokenize(query)):
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(w)
                if f:
                    norm = 1 - self.b + self.b * self.doc_len[i] / self.avg_len
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * norm)
        return out
