#!/usr/bin/env python3
"""Build a dense (sentence-embedding) retrieval index over the same
rag/chunks.jsonl corpus the TF-IDF index already covers, for the
embedding-retrieval comparison against TF-IDF. Model: BAAI/bge-small-en-v1.5
(130MB, retrieval-tuned, small enough to run alongside everything else).

Usage: python build_dense_index.py [--limit N]   # --limit for a timed benchmark first
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve().parent
RAG_DIR = HERE / "rag"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = all chunks")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    chunks = [json.loads(l) for l in open(RAG_DIR / "chunks.jsonl")]
    if args.limit:
        chunks = chunks[: args.limit]
    texts = [c["text"][:1000] for c in chunks]

    model = SentenceTransformer(MODEL_NAME, device="cuda")
    t0 = time.time()
    # bge models expect a "passage: " / no prefix for documents (no query prefix needed at index time)
    embeddings = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True,
                               normalize_embeddings=True, convert_to_numpy=True)
    elapsed = time.time() - t0
    rate = len(texts) / elapsed
    print(f"Encoded {len(texts)} chunks in {elapsed:.1f}s ({rate:.0f} chunks/sec)")

    if args.limit:
        print(f"Benchmark only (--limit {args.limit}); not writing the index.")
        return

    np.save(RAG_DIR / "dense_embeddings.npy", embeddings.astype(np.float32))
    (RAG_DIR / "dense_model.txt").write_text(MODEL_NAME)
    print(f"Wrote {embeddings.shape} embeddings to {RAG_DIR / 'dense_embeddings.npy'}")


if __name__ == "__main__":
    main()
