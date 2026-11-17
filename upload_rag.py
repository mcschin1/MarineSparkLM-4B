"""Per-session RAG over documents a user uploads (not the fixed marine
corpus in rag/ -- see rag.py/rag_dense.py for that). Two-stage retrieval:
dense bge-small over all chunks for recall, then a cross-encoder reranker
for precision, so the LLM only has to read a few already-verified-relevant
passages rather than search a context window itself.

Usage:
    chunks, embeddings = build_index(["notes.pdf", "spec.txt"])
    hits = retrieve(query, chunks, embeddings, k_dense=20, k_final=4)
"""
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sentence_transformers import CrossEncoder, SentenceTransformer

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

_cache = {}


def _load_models():
    if not _cache:
        _cache["embed_model"] = SentenceTransformer(EMBED_MODEL_NAME, device="cuda")
        _cache["reranker"] = CrossEncoder(RERANK_MODEL_NAME, device="cuda")
    return _cache["embed_model"], _cache["reranker"]


def extract_text(path):
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(errors="ignore")


def chunk_text(text, doc_id):
    text = " ".join(text.split())  # collapse whitespace/newlines
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"theme": "uploaded", "doc_id": doc_id, "text": chunk})
        start = end - CHUNK_OVERLAP
    return chunks


def build_index(file_paths):
    """Extracts, chunks, and embeds every uploaded file. Returns (chunks,
    embeddings) -- keep both together in session state; there's no disk
    persistence, this is an in-memory index for one session's documents."""
    embed_model, _ = _load_models()
    chunks = []
    for fp in file_paths:
        text = extract_text(fp)
        if text.strip():
            chunks.extend(chunk_text(text, Path(fp).name))
    if not chunks:
        return [], np.zeros((0, 384), dtype=np.float32)
    embeddings = embed_model.encode(
        [c["text"] for c in chunks], normalize_embeddings=True, convert_to_numpy=True, batch_size=32,
    )
    return chunks, embeddings


def retrieve(query, chunks, embeddings, k_dense=20, k_final=4, min_score=0.0005):
    """Stage 1: dense top-k_dense for recall. Stage 2: cross-encoder
    reranks those candidates and keeps the top k_final clearing min_score
    -- this is the "heavy lifting" step, since dense cosine similarity
    alone is a cheap but coarse filter and the reranker reads query+passage
    together. Without a floor, this always returns k_final passages even
    when the uploaded document doesn't cover the query at all, which risks
    the model being primed with irrelevant context and confidently
    grounding an answer in it anyway.

    min_score=0.0005 is calibrated against 20 genuine (question, its own
    real corpus passage) pairs and 20 mismatched (a different item's
    question, this passage) pairs, using actual corpus text rather than
    reference-answer text (an earlier calibration pass used reference
    answers as the stand-in document, which are far more lexically similar
    to their own question than a real document's prose is -- that inflated
    scores and produced a floor of 0.01 that rejected 8 of 20 genuine
    matches in a real test). On real passage text, genuine matches scored
    0.00004-0.9999 and mismatches scored 0.00004-0.0069 -- the two
    distributions overlap substantially at the low end, so no threshold
    separates them cleanly. 0.0005 keeps 16 of the 20 genuine matches while
    accepting 3 of 20 mismatches as false positives; this is a tradeoff,
    not a clean fix, and a short, weakly-matched real document can still
    score below floor and fall back to ungrounded generation."""
    if not chunks:
        return []
    embed_model, reranker = _load_models()
    q_emb = embed_model.encode([QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True)[0]
    scores = embeddings @ q_emb
    k_dense = min(k_dense, len(chunks))
    candidate_idx = np.argsort(-scores)[:k_dense]

    pairs = [[query, chunks[i]["text"]] for i in candidate_idx]
    rerank_scores = reranker.predict(pairs)
    order = np.argsort(-rerank_scores)[:k_final]
    return [
        {"score": float(rerank_scores[j]), **chunks[candidate_idx[j]]}
        for j in order if rerank_scores[j] >= min_score
    ]


if __name__ == "__main__":
    import tempfile
    sample = (
        "The hull resistance of a ship depends on wetted surface area, "
        "speed, and Froude number. Wave-making resistance dominates at "
        "high Froude numbers, while frictional resistance dominates at low "
        "speeds. Catenary mooring lines provide restoring force through "
        "their own weight and geometry, distinct from taut mooring systems."
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(sample)
        path = f.name

    chunks, embeddings = build_index([path])
    assert len(chunks) >= 1, "expected at least one chunk from the sample text"
    assert embeddings.shape == (len(chunks), 384), "bge-small embeddings should be 384-dim"

    hits = retrieve("What provides restoring force in catenary mooring lines?", chunks, embeddings, k_dense=5, k_final=2)
    assert hits, "expected at least one reranked hit for an on-topic query"
    assert hits == sorted(hits, key=lambda h: -h["score"]), "hits must be sorted by descending rerank score"

    noise = retrieve("What is the capital of France?", chunks, embeddings, k_dense=5, k_final=2)
    assert noise == [], "off-topic query should be rejected by the min_score floor, not returned as false context"

    print(f"upload_rag.py self-check passed ({len(chunks)} chunks, top hit score {hits[0]['score']:.3f}; noise query correctly empty)")
    for h in hits:
        print(f"  [{h['score']:.3f}] {h['doc_id']}: {h['text'][:80]}")

    Path(path).unlink()
