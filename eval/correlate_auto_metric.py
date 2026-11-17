#!/usr/bin/env python3
"""Checks whether a cheap automatic metric (embedding cosine similarity
between generated answer and reference answer) tracks the manual rubric
scores well enough to be a substitute at larger eval scale. No new
generation needed -- reuses the outputs and scores we already have.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_round(outputs_path, scores_path):
    outputs = {}
    for l in open(outputs_path):
        r = json.loads(l)
        outputs[(r["condition"], r["theme"], r["question"])] = (r["generated"], r["reference_answer"])

    rows = []
    for row in csv.DictReader(open(scores_path)):
        key = None
        # scores csv doesn't carry question text directly keyed the same way;
        # match by (item, condition) order against outputs grouped by condition+theme
        rows.append(row)
    return outputs, rows


def main():
    model = SentenceTransformer(MODEL_NAME, device="cuda")

    all_auto, all_human = [], []
    for label, outputs_path, scores_path in [
        ("round1", ROOT / "comparison_outputs.jsonl", ROOT / "comparison_scores.csv"),
        ("round2", ROOT / "comparison_outputs_v3.jsonl", ROOT / "comparison_scores_v3.csv"),
    ]:
        gen_rows = [json.loads(l) for l in open(outputs_path)]
        score_rows = list(csv.DictReader(open(scores_path)))

        # Both files are written/scored in the same (condition, then item-order) sequence,
        # so zip them positionally within each condition block.
        by_cond_gen = {}
        for r in gen_rows:
            by_cond_gen.setdefault(r["condition"], []).append(r)
        by_cond_score = {}
        for r in score_rows:
            by_cond_score.setdefault(r["condition"], []).append(r)

        auto_scores, human_scores = [], []
        for cond in by_cond_gen:
            gens = by_cond_gen[cond]
            scores = by_cond_score.get(cond, [])
            if len(gens) != len(scores):
                print(f"[{label}] WARNING: {cond} has {len(gens)} generations but {len(scores)} scores, skipping")
                continue
            gen_texts = [g["generated"][:800] for g in gens]
            ref_texts = [g["reference_answer"][:800] for g in gens]
            gen_emb = model.encode(gen_texts, normalize_embeddings=True, convert_to_numpy=True)
            ref_emb = model.encode(ref_texts, normalize_embeddings=True, convert_to_numpy=True)
            sims = (gen_emb * ref_emb).sum(axis=1)
            totals = [int(s["total"]) for s in scores]
            auto_scores.extend(sims.tolist())
            human_scores.extend(totals)

        r_pearson, p_pearson = stats.pearsonr(auto_scores, human_scores)
        r_spearman, p_spearman = stats.spearmanr(auto_scores, human_scores)
        print(f"[{label}] n={len(auto_scores)}  Pearson r={r_pearson:.3f} (p={p_pearson:.4f})  "
              f"Spearman rho={r_spearman:.3f} (p={p_spearman:.4f})")
        all_auto.extend(auto_scores)
        all_human.extend(human_scores)

    r_pearson, p_pearson = stats.pearsonr(all_auto, all_human)
    r_spearman, p_spearman = stats.spearmanr(all_auto, all_human)
    print(f"\n[combined] n={len(all_auto)}  Pearson r={r_pearson:.3f} (p={p_pearson:.4g})  "
          f"Spearman rho={r_spearman:.3f} (p={p_spearman:.4g})")


if __name__ == "__main__":
    main()
