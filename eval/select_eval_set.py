#!/usr/bin/env python3
"""Reproduces the exact held-out validation split used by
finetune_oceangpt_instruct.py (seed 42, 10% held out), then picks the first
2 rows per theme as a fixed 20-question comparison eval set. None of these
rows were trained on by any instruction adapter.
"""
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
QA_PATH = ROOT / "data" / "qa_dataset" / "qa_seed.jsonl"
OUT = Path(__file__).resolve().parent / "comparison_eval_set.jsonl"


def main():
    rows = [json.loads(l) for l in open(QA_PATH)]
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(rows), generator=g).tolist()
    n_val = max(1, int(len(rows) * 0.1))
    val_rows = [rows[i] for i in perm[:n_val]]

    by_theme = {}
    for r in val_rows:
        by_theme.setdefault(r["theme"], []).append(r)

    eval_set = []
    for theme in sorted(by_theme):
        eval_set.extend(by_theme[theme][:2])

    with open(OUT, "w") as f:
        for r in eval_set:
            f.write(json.dumps({
                "theme": r["theme"], "doc_id": r["doc_id"],
                "question": r["messages"][0]["content"],
                "reference_answer": r["messages"][1]["content"],
            }) + "\n")
    print(f"{len(eval_set)} eval prompts written to {OUT}")


if __name__ == "__main__":
    main()
