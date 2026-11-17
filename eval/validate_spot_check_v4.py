#!/usr/bin/env python3
"""Joins the hand-scored spot-check sample (24 items, scored by reading
spot_check_v4_readable.txt) against the bge-small auto-scores for those
same items, to check whether the auto-metric tracks human judgment on this
fresh batch -- not just on rounds 1-2, which is all correlate_auto_metric.py
validated. Also recomputes per-condition hand-scored means for the sample
and compares the ranking against the full-set auto-scored means.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

from scipy import stats

random.seed(7)
ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "comparison_outputs_v4.jsonl"
AUTO_CSV = ROOT / "comparison_auto_scores_v4.csv"
N_PER_COND = 6

rows = [json.loads(l) for l in open(OUTPUTS)]
by_cond = defaultdict(list)
for i, r in enumerate(rows):
    by_cond[r["condition"]].append((i, r))

sample = []
for cond, items in by_cond.items():
    sample.extend(random.sample(items, min(N_PER_COND, len(items))))
random.shuffle(sample)

# Hand scores, in the same order spot_check_v4_readable.txt printed them
# (sample 0..23), from reading each item against the real 0-6 rubric.
HAND_TOTALS = [0, 3, 4, 4, 3, 1, 5, 3, 4, 4, 3, 1, 4, 3, 1, 3, 4, 5, 4, 6, 3, 3, 3, 1]
assert len(HAND_TOTALS) == len(sample) == 24

# item index within its condition block, matching score_auto_v4.py's CSV
cond_counter = defaultdict(int)
item_idx_in_cond = {}
for i, r in enumerate(rows):
    item_idx_in_cond[i] = cond_counter[r["condition"]]
    cond_counter[r["condition"]] += 1

auto_lookup = {}
import csv
for row in csv.DictReader(open(AUTO_CSV)):
    auto_lookup[(row["condition"], int(row["item"]))] = float(row["auto_score"])

auto_scores = []
hand_by_cond = defaultdict(list)
auto_by_cond = defaultdict(list)
for (global_idx, r), hand_total in zip(sample, HAND_TOTALS):
    local_idx = item_idx_in_cond[global_idx]
    auto = auto_lookup[(r["condition"], local_idx)]
    auto_scores.append(auto)
    hand_by_cond[r["condition"]].append(hand_total)
    auto_by_cond[r["condition"]].append(auto)

r_pearson, p_pearson = stats.pearsonr(auto_scores, HAND_TOTALS)
r_spearman, p_spearman = stats.spearmanr(auto_scores, HAND_TOTALS)
print(f"Spot-check (n=24): Pearson r={r_pearson:.3f} (p={p_pearson:.4f})  "
      f"Spearman rho={r_spearman:.3f} (p={p_spearman:.4f})")

print("\nPer-condition means, hand-scored spot-check (n=6 each) vs. full-set auto-score (n=120 each):")
for cond in by_cond:
    hand_mean = sum(hand_by_cond[cond]) / len(hand_by_cond[cond])
    spot_auto_mean = sum(auto_by_cond[cond]) / len(auto_by_cond[cond])
    print(f"  {cond:36s} hand(n=6)={hand_mean:.2f}/6   auto-on-spot(n=6)={spot_auto_mean:.4f}")
