"""Dense (sentence-embedding) retrieval over the same corpus rag.py's
TF-IDF retrieve() covers, for the embedding-retrieval-vs-TF-IDF comparison.
Separate module so it never changes the app's default (TF-IDF) behavior --
not imported by app.py.

Usage:
    from rag_dense import retrieve_dense
    for hit in retrieve_dense("mooring line tension", k=4):
        print(hit["score"], hit["theme"], hit["doc_id"], hit["text"][:80])
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
RAG_DIR = ROOT / "rag"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
# bge models need this instruction prefix on queries (not on indexed
# passages) for proper retrieval discrimination -- without it, off-topic
# queries score almost as high as genuine topical matches.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_cache = {}


def _load():
    if _cache:
        return _cache
    emb_path = RAG_DIR / "dense_embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"No dense index at {emb_path} -- run build_dense_index.py first")
    _cache["embeddings"] = np.load(emb_path)
    _cache["chunks"] = [json.loads(line) for line in open(RAG_DIR / "chunks.jsonl")]
    _cache["model"] = SentenceTransformer(MODEL_NAME, device="cuda")
    return _cache


def retrieve_dense(query, k=4, min_score=0.75, exclude_doc_ids=None):
    """Top-k chunks by cosine similarity (embeddings are pre-normalized, so
    this is a dot product) to query, same return shape as rag.retrieve().
    min_score is on a different scale than TF-IDF's -- bge cosine similarity
    (with the query prefix) for genuine topical matches on this corpus runs
    ~0.78-0.86, and off-topic queries ~0.56-0.72, calibrated empirically the
    same way the TF-IDF floor was (4 topical + 4 off-topic probe queries)."""
    idx = _load()
    embeddings, chunks, model = idx["embeddings"], idx["chunks"], idx["model"]

    q_emb = model.encode([QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True)[0]
    scores = embeddings @ q_emb
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
    hits = retrieve_dense("catenary mooring line tension", k=3)
    assert hits, "expected at least one hit for a term that appears in the corpus"
    assert hits == sorted(hits, key=lambda h: -h["score"]), "hits must be sorted by descending score"
    assert all(0 < h["score"] <= 1.0001 for h in hits), "cosine similarity scores must be in (0, 1]"

    noise = retrieve_dense("the history of impressionist painting", k=3)
    assert noise == [], "off-topic query should be rejected by the min_score floor, not returned as false context"

    print(f"rag_dense.py self-check passed ({len(hits)} hits, top score {hits[0]['score']:.3f}; noise query correctly empty)")
    for h in hits:
        print(f"  [{h['score']:.3f}] {h['theme']}/{h['doc_id']}: {h['text'][:80]}")
