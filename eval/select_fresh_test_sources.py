#!/usr/bin/env python3
"""Select passages for a genuinely held-out test set -- excludes every
doc_id already in qa_seed.jsonl (train AND val), so these passages were
never seen in training, validation, or checkpoint selection. Fixes the
selection-leakage issue: the old 20-prompt eval set was drawn from the
validation split used for early stopping, so the checkpoint itself was
picked using data that included those exact questions.

Reuses the same anchor-term lists and junk-title blocklist as
select_qa_sources_v2.py. 6 candidates/theme (buffer for quality filtering
down to 2/theme after QA generation).
"""
import json
import random
import re

random.seed(99)  # different seed from the training-data selection rounds
RAG_DIR = "/home/mcschin1/marine-llm/package/MarineSparkLM/rag"
QA_SEED_PATH = "/home/mcschin1/marine-llm/run_500m_v1/data/qa_dataset/qa_seed.jsonl"
OUT = "/home/mcschin1/marine-llm/run_500m_v1/eval/fresh_test_candidates.jsonl"

ANCHORS = {
    "naval_architecture": ["hull", "ship design", "naval architect", "displacement", "freeboard",
                            "ship stability", "intact stability", "shipbuilding", "vessel design", "midship", "resistance and powering"],
    "hydrodynamics": ["hydrodynamic", "wave load", "added mass", "diffraction", "radiation damping",
                       "panel method", "morison", "seakeeping", "wave-structure"],
    "propulsion_machinery": ["propeller", "marine engine", "marine diesel", "thruster", "shaft power",
                              "propulsion system", "marine propulsion", "cavitation"],
    "offshore_structures": ["jacket structure", "fpso", "mooring", "riser", "subsea", "offshore platform",
                             "semi-submersible", "spar platform", "tension leg platform", "jack-up"],
    "offshore_energy": ["offshore wind", "wind turbine", "wave energy converter", "tidal energy",
                         "floating wind", "offshore oil and gas", "fowt"],
    "control_autonomy": ["dynamic positioning", "autonomous underwater vehicle", " auv ", " rov ",
                          "ship autopilot", "marine vehicle control", "vessel control", "station keeping",
                          "unmanned surface vehicle", "trajectory tracking control", "sliding mode control",
                          "marine control system", "waypoint tracking"],
    "marine_vehicles": ["autonomous underwater vehicle", "remotely operated vehicle", " auv ", " rov ",
                         "unmanned surface vehicle", "underwater glider", "underwater vehicle", "submersible"],
    "sensors_communications": ["sonar", "underwater acoustic", "underwater communication", "acoustic imaging",
                                "underwater localization", "underwater camera", "hydrophone"],
    "simulation_modelling": ["computational fluid dynamics", "cfd", "openfoam", "finite element", "seakeeping simulation",
                              "ship simulation", "digital twin ship", "mooring simulation"],
    "maritime_operations": ["ballast water", "port operation", "maritime safety", "vessel traffic",
                             "cargo ship", "container ship", "maritime regulation", "shipping industry",
                             "voyage planning", "maritime logistics", "corrosion"],
}

JUNK_TITLE_PATTERNS = re.compile(
    r"wikipedia:(Jim Mattis|Memory|Waymo|CrossFit|Toilet seat riser|Godzilla|Tidal$|"
    r"List of Evercade games|Paul Riser|Ash Riser|Hull City|Stairs|Frank Sobotka|CSIS|"
    r"Electronic stability control|Smith.Purcell|Plasma propulsion|Ion thruster|Resistojet|"
    r"1998 Yosu submersible incident)",
    re.IGNORECASE,
)

used_doc_ids = {json.loads(l)["doc_id"] for l in open(QA_SEED_PATH)}
print(f"Excluding {len(used_doc_ids)} doc_ids already in qa_seed.jsonl (train+val)")

N_PER_THEME = 6
chunks_by_theme = {}
with open(f"{RAG_DIR}/chunks.jsonl") as f:
    for line in f:
        c = json.loads(line)
        if len(c["text"]) < 400:
            continue
        chunks_by_theme.setdefault(c["theme"], []).append(c)

selected = []
for theme, anchors in ANCHORS.items():
    pool = chunks_by_theme.get(theme, [])
    anchor_re = re.compile("|".join(re.escape(a) for a in anchors), re.IGNORECASE)
    matches = [c for c in pool if anchor_re.search(c["text"]) and not JUNK_TITLE_PATTERNS.search(c["doc_id"])
               and c["doc_id"] not in used_doc_ids]
    random.shuffle(matches)
    seen_docs = set()
    picked = []
    for c in matches:
        if c["doc_id"] in seen_docs:
            continue
        seen_docs.add(c["doc_id"])
        picked.append(c)
        if len(picked) >= N_PER_THEME:
            break
    selected.extend(picked)
    print(f"{theme:<24} {len(matches):>6} matches, {len(picked)} picked")

with open(OUT, "w") as f:
    for c in selected:
        f.write(json.dumps(c) + "\n")
print(f"\n{len(selected)} candidates written to {OUT}")
