#!/usr/bin/env python3
"""Auto-scores comparison_outputs_v4.jsonl with bge-small cosine similarity
(the same automatic metric validated in correlate_auto_metric.py: r=0.55
combined with human scores on rounds 1-2). Writes a per-item CSV and prints
per-condition means plus paired Wilcoxon + bootstrap stats on the
similarity scores, so the enlarged (100+) round doesn't require hand-scoring
every item.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
IN = ROOT / "comparison_outputs_v4.jsonl"
OUT = ROOT / "comparison_auto_scores_v4.csv"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
N_BOOT = 10_000
rng = np.random.default_rng(42)

LABELS = {
    "base": "Base (no adapter)",
    "continuation_adapter": "Continuation adapter",
    "instruction_adapter": "Instruction adapter",
    "instruction_adapter_rag_tfidf": "Instruction adapter + RAG (TF-IDF)",
}


def bootstrap_mean_ci(values, n_boot=N_BOOT):
    values = np.asarray(values, dtype=float)
    n = len(values)
    boot_means = np.array([rng.choice(values, size=n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return values.mean(), lo, hi


def main():
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    rows = [json.loads(l) for l in open(IN)]

    gen_texts = [r["generated"][:800] for r in rows]
    ref_texts = [r["reference_answer"][:800] for r in rows]
    gen_emb = model.encode(gen_texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=64)
    ref_emb = model.encode(ref_texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=64)
    sims = (gen_emb * ref_emb).sum(axis=1)

    by_cond_item = defaultdict(dict)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item", "theme", "condition", "doc_id", "auto_score"])
        item_idx = defaultdict(int)
        for r, s in zip(rows, sims):
            idx = item_idx[r["condition"]]
            item_idx[r["condition"]] += 1
            w.writerow([idx, r["theme"], r["condition"], r["doc_id"], f"{s:.4f}"])
            by_cond_item[r["condition"]][idx] = float(s)

    conditions = list(LABELS)
    n_items = len(rows) // len(conditions)
    print(f"Per-condition mean auto-score (bge-small cosine similarity to reference), bootstrap 95% CI (n={n_items}, {N_BOOT} resamples):\n")
    scores = {}
    for c in conditions:
        vals = [by_cond_item[c][i] for i in range(n_items)]
        scores[c] = vals
        mean, lo, hi = bootstrap_mean_ci(vals)
        print(f"  {LABELS[c]:36s} mean={mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    print("\nPaired comparisons (Wilcoxon signed-rank on per-item auto-scores, bootstrap CI on mean paired difference):\n")
    pairs = [
        ("base", "instruction_adapter"),
        ("continuation_adapter", "instruction_adapter"),
        ("instruction_adapter", "instruction_adapter_rag_tfidf"),
        ("base", "continuation_adapter"),
    ]
    for a, b in pairs:
        diffs = np.array(scores[b]) - np.array(scores[a])
        stat, p = stats.wilcoxon(scores[a], scores[b])
        mean_diff, lo, hi = bootstrap_mean_ci(diffs)
        print(f"  {LABELS[a]:24s} vs {LABELS[b]:28s}  mean diff={mean_diff:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  Wilcoxon p={p:.4g}")

    print(f"\nPer-item scores written to {OUT}")


if __name__ == "__main__":
    main()
