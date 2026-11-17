#!/usr/bin/env python3
"""Manual quality pass over fresh_test_set_v2.jsonl: the automated filter
(bibliographic-question regex + short-answer length) missed several
keyword-collision false positives (same class of corpus noise already
documented in the paper -- off-topic content that matched a theme's anchor
terms by coincidence) and one OCR-garbled item. Drops those 10 items and
replaces each with a manually-verified clean question from the same theme's
unused candidate pool, keeping the set at exactly 12/theme.
"""
import json
from collections import defaultdict

IN = "/home/mcschin1/marine-llm/run_500m_v1/eval/fresh_test_set_v2.jsonl"
GENERATED = "/home/mcschin1/marine-llm/run_500m_v1/eval/fresh_test_generated_v2.jsonl"
OUT = "/home/mcschin1/marine-llm/run_500m_v1/eval/fresh_test_set_v2.jsonl"

# (theme, doc_id, question-prefix) of items to drop -- off-topic keyword
# collisions (spacecraft propulsion, automobile aerodynamics, firearms,
# fish biology, an unrelated geopolitical conflict) or OCR-garbled math.
DROP = {
    ("naval_architecture", "arxiv:2411.01434v1"),
    ("naval_architecture", "wikipedia:Papua conflict"),
    ("hydrodynamics", "local:8-Problems.pdf"),
    ("hydrodynamics", "arxiv:2609.05494v1"),
    ("propulsion_machinery", "wikipedia:Spacecraft propulsion"),
    ("propulsion_machinery", "wikipedia:Spacecraft electric propulsion"),
    ("propulsion_machinery", "wikipedia:Tesla Roadster (second generation)"),
    ("offshore_structures", "wikipedia:Muzzle booster"),
    ("control_autonomy", "local:nonlin tracking control USING SLIDING SURFACES WITH APPLICATION TO ROBOT MANIPULATORS.pdf"),
    ("marine_vehicles", "wikipedia:Oarfish"),
}

# Replacements: manually verified on-topic, clean questions pulled from the
# same theme's unused candidates in fresh_test_generated_v2.jsonl.
REPLACEMENTS = [
    ("naval_architecture", "wikipedia:Research vessel",
     "Can you name three examples of polar research vessels mentioned in the text?"),
    ("naval_architecture", "wikipedia:International Code on Intact Stability",
     "What specific content is covered under the mandatory provisions of Part A?"),
    ("hydrodynamics", "arxiv:1310.5977v1",
     "What is the relationship between the critical gap and the pitch angle for fixed contour period?"),
    ("hydrodynamics", "arxiv:2408.03222v1",
     "How many categories of wing-wake interactions are identified in the passage?"),
    ("propulsion_machinery", "frontiers:10.1007/s11804-023-00359-1",
     "What method was used to analyze the performance of the Kappel tip-rake propellers in the study?"),
    ("propulsion_machinery", "frontiers:10.54097/ajst.v3i1.1824",
     "What is the primary function of the propeller in a ship's electric propulsion system?"),
    ("propulsion_machinery", "wikipedia:Voith Schneider Propeller",
     "What are the advantages of the Voith-Schneider drive over azimuth thrusters and Kort nozzles?"),
    ("offshore_structures", "wikipedia:Malaysia Marine and Heavy Engineering",
     "What types of projects does MHB offer construction and engineering services for in the offshore oil and gas industry?"),
    ("control_autonomy", "wikipedia:Dynamic positioning",
     "What is dynamic positioning and how does it maintain a vessel's position and heading?"),
    ("marine_vehicles", "wikipedia:Large Unmanned Surface Vehicle",
     "What is the primary purpose of the Large Unmanned Surface Vehicle (LUSV) in the United States Navy?"),
]

gen_lookup = {}
for l in open(GENERATED):
    r = json.loads(l)
    gen_lookup[(r["theme"], r["doc_id"], r["messages"][0]["content"])] = r["messages"][1]["content"]

rows = [json.loads(l) for l in open(IN)]
kept = [r for r in rows if (r["theme"], r["doc_id"]) not in DROP]
print(f"Dropped {len(rows) - len(kept)} items")

for theme, doc_id, question in REPLACEMENTS:
    ans = gen_lookup.get((theme, doc_id, question))
    assert ans is not None, f"could not find {theme}/{doc_id}/{question!r} in generated pairs"
    kept.append({"theme": theme, "doc_id": doc_id, "question": question, "reference_answer": ans})

by_theme = defaultdict(int)
for r in kept:
    by_theme[r["theme"]] += 1
for t, n in sorted(by_theme.items()):
    assert n == 12, f"{t} has {n} items, expected 12"
    print(f"{t:<24} {n}")

with open(OUT, "w") as f:
    for r in kept:
        f.write(json.dumps(r) + "\n")
print(f"\n{len(kept)} questions written to {OUT}")
