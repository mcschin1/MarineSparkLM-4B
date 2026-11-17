"""Retrieval over the training corpus, for grounding generation in real
source text instead of trusting this small model's recall. Self-contained:
the index (rag/*.npz,*.npy,*.json,*.jsonl) ships with this package.

Usage:
    from rag import retrieve
    for hit in retrieve("mooring line tension", k=4):
        print(hit["score"], hit["theme"], hit["doc_id"], hit["text"][:80])
"""
import json
import re
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent
RAG_DIR = ROOT / "rag"

TOKEN_RE = re.compile(r"[a-z][a-z0-9_]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


_cache = {}


def _load():
    if _cache:
        return _cache
    if not (RAG_DIR / "tfidf.npz").exists():
        raise FileNotFoundError(f"No RAG index found at {RAG_DIR} — this package should ship one")
    _cache["matrix"] = sparse.load_npz(RAG_DIR / "tfidf.npz")
    _cache["idf"] = np.load(RAG_DIR / "idf.npy")
    _cache["vocab"] = json.load(open(RAG_DIR / "vocab.json"))
    _cache["chunks"] = [json.loads(line) for line in open(RAG_DIR / "chunks.jsonl")]
    return _cache


def retrieve(query, k=4, min_score=0.3, exclude_doc_ids=None):
    """Top-k chunks by cosine similarity to query, as a list of dicts with
    score/theme/doc_id/text. Empty list if nothing clears min_score — on
    this corpus real topical matches score ~0.37-0.53 and off-topic/garbage
    queries score ~0.15-0.28, so 0.3 separates signal from noise.

    exclude_doc_ids: optional set of doc_ids to exclude from retrieval —
    for leave-one-out eval only (so a question can't retrieve its own
    source passage); production callers (the app) never pass this."""
    idx = _load()
    vocab, idf, matrix, chunks = idx["vocab"], idx["idf"], idx["matrix"], idx["chunks"]

    counts = {}
    for term in tokenize(query):
        col = vocab.get(term)
        if col is not None:
            counts[col] = counts.get(col, 0) + 1
    if not counts:
        return []
    cols = sorted(counts)
    data = [counts[c] * idf[c] for c in cols]
    q = sparse.csr_matrix((data, ([0] * len(cols), cols)), shape=(1, len(vocab)))
    norm = sparse.linalg.norm(q)
    if norm == 0:
        return []
    q = q.multiply(1.0 / norm)

    scores = (matrix @ q.T).toarray().ravel()
    if exclude_doc_ids:
        for i, c in enumerate(chunks):
            if c["doc_id"] in exclude_doc_ids:
                scores[i] = -1.0
    top = np.argsort(-scores)[:k]
    return [
        {"score": float(scores[i]), **chunks[i]}
        for i in top if scores[i] >= min_score
    ]


if __name__ == "__main__":
    hits = retrieve("catenary mooring line tension", k=3)
    assert hits, "expected at least one hit for a term that appears verbatim in the corpus"
    assert hits == sorted(hits, key=lambda h: -h["score"]), "hits must be sorted by descending score"
    assert all(0 < h["score"] <= 1.0001 for h in hits), "cosine similarity scores must be in (0, 1]"

    noise = retrieve("purple elephant dancing banana", k=3)
    assert noise == [], "off-topic query should be rejected by the min_score floor, not returned as false context"

    print(f"rag.py self-check passed ({len(hits)} hits, top score {hits[0]['score']:.3f}; noise query correctly empty)")
