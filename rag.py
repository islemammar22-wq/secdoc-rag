"""Retrieval + optional LLM answer with citations."""
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from bm25 import BM25

INDEX_DIR = Path("index")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60  # standard constant for reciprocal rank fusion
POOL = 100  # how many top results from each method are fused
LEX_WEIGHT = 1.0  # raise (e.g. 1.5 or 2.0) to trust keyword matches more

_model = None
_emb = None
_chunks = None
_bm25 = None


def _load():
    global _model, _emb, _chunks, _bm25
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
        _emb = np.load(INDEX_DIR / "embeddings.npy")
        _chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
        _bm25 = BM25([c["text"] for c in _chunks])


def search(query, k=5, mode="hybrid"):
    """Return top-k chunks.

    mode="dense":  embedding (cosine) similarity only. This is the baseline.
    mode="hybrid": embeddings + BM25 keywords, merged with reciprocal rank fusion.
    """
    _load()
    q = _model.encode([query], normalize_embeddings=True)[0]
    dense = _emb @ q

    if mode == "dense":
        top = np.argsort(-dense)[:k]
        return [{**_chunks[i], "score": float(dense[i])} for i in top]

    lexical = _bm25.scores(query)
    fused = np.zeros(len(_chunks), dtype="float32")
    for scores, weight in ((dense, 1.0), (lexical, LEX_WEIGHT)):
        order = [i for i in np.argsort(-scores)[:POOL] if scores[i] > 0]
        for rank, i in enumerate(order, start=1):
            fused[i] += weight / (RRF_K + rank)
    top = np.argsort(-fused)[:k]
    return [{**_chunks[i], "score": float(fused[i])} for i in top]


def _ollama_answer(question, passages):
    """Optional: set OLLAMA_MODEL (e.g. llama3.1:8b) to get a generated answer."""
    model = os.environ.get("OLLAMA_MODEL")
    if not model:
        return None
    context = "\n\n".join(
        f"[{i + 1}] ({p['source']}, p.{p['page']}) {p['text']}"
        for i, p in enumerate(passages)
    )
    prompt = (
        "Answer the question using ONLY the numbered passages. "
        "Cite passages like [1]. If the passages do not contain the answer, "
        "say you cannot find it.\n\n"
        f"{context}\n\nQuestion: {question}\nAnswer:"
    )
    body = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(
        os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate"),
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["response"].strip()


def answer(question, k=5):
    passages = search(question, k)
    return {
        "question": question,
        "answer": _ollama_answer(question, passages),  # None if no LLM configured
        "sources": [
            {
                "source": p["source"],
                "page": p["page"],
                "label": p.get("label", str(p["page"])),
                "score": round(p["score"], 3),
                "excerpt": p["text"][:300],
            }
            for p in passages
        ],
    }
