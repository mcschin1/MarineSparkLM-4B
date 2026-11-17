#!/usr/bin/env python3
"""Paired significance test + bootstrap CIs on the comparison harness scores
(comparison_scores.csv). Addresses a reviewer comment: the paper reported
raw totals with no uncertainty or significance measure.

For each condition: bootstrap 95% CI on the mean total score (10,000
resamples over the 20 paired items). For each pair of conditions: Wilcoxon
signed-rank test (paired, non-parametric -- appropriate for n=20 ordinal
0-6 scores) plus a bootstrap CI on the mean paired difference.

Caveat (keep in any writeup): n=20 gives the Wilcoxon very little power:
treat p-values as a sanity check on direction, not a strong significance
claim.
"""
import csv
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

CONDITIONS = ["base", "continuation_adapter", "instruction_adapter", "instruction_adapter_rag"]
LABELS = {
    "base": "Base (no adapter)",
    "continuation_adapter": "Continuation adapter",
    "instruction_adapter": "Instruction adapter",
    "instruction_adapter_rag": "Instruction adapter + RAG",
    "instruction_adapter_rag_tfidf": "Instruction adapter + RAG (TF-IDF)",
    "base_rag_tfidf": "Base + RAG (TF-IDF)",
    "instruction_adapter_synthetic_only": "Instruction adapter (synthetic-only)",
    "instruction_adapter_rag_dense": "Instruction adapter + RAG (dense)",
}
N_BOOT = 10_000
rng = np.random.default_rng(42)


def bootstrap_mean_ci(values, n_boot=N_BOOT):
    values = np.asarray(values, dtype=float)
    n = len(values)
    boot_means = np.array([rng.choice(values, size=n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return values.mean(), lo, hi


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "comparison_scores.csv"
    conditions = sys.argv[2].split(",") if len(sys.argv) > 2 else CONDITIONS

    by_item_cond = defaultdict(dict)
    for row in csv.DictReader(open(csv_path)):
        by_item_cond[int(row["item"])][row["condition"]] = int(row["total"])

    items = sorted(by_item_cond)
    scores = {c: [by_item_cond[i][c] for i in items] for c in conditions}

    n_items = len(items)
    print(f"Per-condition mean total score /6, with bootstrap 95% CI (n={n_items} items, {N_BOOT} resamples):\n")
    for c in conditions:
        mean, lo, hi = bootstrap_mean_ci(scores[c])
        print(f"  {LABELS[c]:36s} mean={mean:.2f}  95% CI [{lo:.2f}, {hi:.2f}]  sum={sum(scores[c])}/{n_items*6}")

    print(f"\nPaired comparisons (Wilcoxon signed-rank on per-item totals, plus bootstrap CI on the mean paired difference):\n")
    if len(sys.argv) > 3:
        pairs = [tuple(p.split(":")) for p in sys.argv[3].split(";")]
    elif conditions == CONDITIONS:
        pairs = [
            ("base", "instruction_adapter"),
            ("continuation_adapter", "instruction_adapter"),
            ("instruction_adapter", "instruction_adapter_rag"),
            ("base", "continuation_adapter"),
        ]
    else:
        pairs = [(conditions[0], c) for c in conditions[1:]]
    for a, b in pairs:
        diffs = np.array(scores[b]) - np.array(scores[a])
        if np.all(diffs == 0):
            print(f"  {LABELS[a]} vs {LABELS[b]}: no differences, Wilcoxon undefined")
            continue
        stat, p = stats.wilcoxon(scores[a], scores[b])
        mean_diff, lo, hi = bootstrap_mean_ci(diffs)
        print(f"  {LABELS[a]:24s} vs {LABELS[b]:28s}  mean diff={mean_diff:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  Wilcoxon W={stat:.1f}, p={p:.4f}")


if __name__ == "__main__":
    main()
