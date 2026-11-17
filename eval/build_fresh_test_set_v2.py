#!/usr/bin/env python3
"""Filters fresh_test_generated_v2.jsonl (287 raw pairs) down to a balanced
12-per-theme (120 total) held-out set: drops pure bibliographic/metadata
questions (title/authors/publication date -- these test nothing about
domain knowledge) and picks one question per unique doc_id for document
diversity, matching fresh_test_set.jsonl's schema.
"""
import json
import re
from collections import defaultdict

IN = "/home/mcschin1/marine-llm/run_500m_v1/eval/fresh_test_generated_v2.jsonl"
OUT = "/home/mcschin1/marine-llm/run_500m_v1/eval/fresh_test_set_v2.jsonl"
N_PER_THEME = 12

BIBLIOGRAPHIC_RE = re.compile(
    r"\b(title of the (article|book|paper)|who (are|is) the author|"
    r"when was the article published|published in its original form|"
    r"which (volume|issue) of|what (journal|conference) was)\b",
    re.IGNORECASE,
)

rows = [json.loads(l) for l in open(IN)]
by_theme = defaultdict(list)
for r in rows:
    q, a = r["messages"][0]["content"], r["messages"][1]["content"]
    if BIBLIOGRAPHIC_RE.search(q):
        continue
    if len(a) < 40:
        continue
    by_theme[r["theme"]].append({"doc_id": r["doc_id"], "question": q, "reference_answer": a})

final = []
for theme, items in by_theme.items():
    by_doc = defaultdict(list)
    for it in items:
        by_doc[it["doc_id"]].append(it)
    picked = []
    for doc_id, cands in by_doc.items():
        picked.append(cands[0])  # one question per doc_id for document diversity
        if len(picked) >= N_PER_THEME:
            break
    for p in picked:
        final.append({"theme": theme, **p})
    print(f"{theme:<24} {len(items):>3} candidates after filtering, {len(picked)} picked "
          f"({len(by_doc)} unique docs available)")

with open(OUT, "w") as f:
    for row in final:
        f.write(json.dumps(row) + "\n")
print(f"\n{len(final)} questions written to {OUT}")
