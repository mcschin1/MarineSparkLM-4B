#!/usr/bin/env python3
"""Draws a stratified random sample from comparison_outputs_v4.jsonl (6 per
condition = 24 items, ~20% of the enlarged set) for hand-scoring with the
real 0-6 rubric, to validate the auto-metric held up on this fresh batch
before trusting its full-set scores (correlate_auto_metric.py only
validated it on rounds 1-2, not this new set).
"""
import json
import random
from collections import defaultdict
from pathlib import Path

random.seed(7)
ROOT = Path(__file__).resolve().parent
IN = ROOT / "comparison_outputs_v4.jsonl"
OUT_READABLE = ROOT / "spot_check_v4_readable.txt"
OUT_TEMPLATE = ROOT / "spot_check_v4_scores_template.csv"
N_PER_COND = 6

rows = [json.loads(l) for l in open(IN)]
by_cond = defaultdict(list)
for i, r in enumerate(rows):
    by_cond[r["condition"]].append((i, r))

sample = []
for cond, items in by_cond.items():
    sample.extend(random.sample(items, min(N_PER_COND, len(items))))
random.shuffle(sample)

with open(OUT_READABLE, "w") as f, open(OUT_TEMPLATE, "w") as g:
    g.write("sample_id,condition,theme,accuracy,topicality,specificity,total\n")
    for sid, (idx, r) in enumerate(sample):
        f.write(f"=== sample {sid} (condition={r['condition']}, theme={r['theme']}) ===\n")
        f.write(f"Question: {r['question']}\n")
        f.write(f"Reference: {r['reference_answer']}\n")
        f.write(f"Generated: {r['generated']}\n\n")
        g.write(f"{sid},{r['condition']},{r['theme']},,,,\n")

print(f"{len(sample)} items sampled. Review {OUT_READABLE}, score into {OUT_TEMPLATE}.")
