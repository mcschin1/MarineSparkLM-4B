# MarineSparkLM-4Bv1

A LoRA instruction fine-tune of [OceanGPT-basic-4B-Instruct](https://huggingface.co/zjunlp/OceanGPT-basic-4B-Instruct)
(Qwen3-4B architecture) for marine and offshore engineering question answering,
with optional retrieval-augmented generation (RAG) grounding over a 10-theme
corpus.

**Author:** Dr Cheng Siong Chin, Faculty of Science, Agriculture and Engineering,
Newcastle University (Singapore). Contact: cheng.chin@newcastle.ac.uk

**Paper:** [MarineSparkLM-4B: A Lightweight Language Model for Marine and
Offshore Engineering Built by Parameter-Efficient Adaptation](https://doi.org/10.48550/arXiv.2611.19151)

If you use this model or code, please cite:
```bibtex
@misc{chin2026marinesparklm,
  title         = {MarineSparkLM-4B: A Lightweight Language Model for Marine and Offshore Engineering Built by Parameter-Efficient Adaptation},
  author        = {Chin, Cheng Siong},
  year          = {2026},
  eprint        = {2611.19151},
  archivePrefix = {arXiv},
  doi           = {10.48550/arXiv.2611.19151},
  url           = {https://doi.org/10.48550/arXiv.2611.19151}
}
```

## Getting the model weights

This repository holds the app code, RAG index metadata, evaluation
artifacts, and the paper — not the large binary files (adapter weights,
dense embeddings, TF-IDF matrix, chunk text), which exceed GitHub's
per-file size limit. Get those from one of:

- **Just the LoRA adapter** (for loading with `transformers`/`peft`
  directly from the Hub): [`<hf-username>/MarineSparkLM-4Bv1`](https://huggingface.co/) *(link added once published)*
- **The complete runnable package** (adapter + full RAG index + this
  README + the paper), as a single archive: see the
  [`MarineSparkLM-4Bv1` release](../../releases/tag/MarineSparkLM-4Bv1)
  on this repo and download `MarineSparkLM-4B.zip`, then extract it over
  this checkout (it fills in `adapter/best_adapter/adapter_model.safetensors`,
  `rag/dense_embeddings.npy`, `rag/chunks.jsonl`, and `rag/tfidf.npz`).

## Background

OceanGPT-basic-4B-Instruct already has broad ocean-science knowledge, but a
fixed set of evaluation prompts showed it drifting off-topic on some marine
and offshore engineering questions (for example, a ballast-water-treatment
prompt would drift into general marine pollution). To fix that, this project
built a grounded instruction-tuning set and LoRA fine-tuned the base model on
it, independently of the base model's own training (no stacking with any
other adapter).

**Themes covered (10):** control_autonomy, hydrodynamics, marine_vehicles,
maritime_operations, naval_architecture, offshore_energy,
offshore_structures, propulsion_machinery, sensors_communications,
simulation_modelling.

**Training data:** 2,295 grounded question/answer pairs (`data/qa_seed.jsonl`),
in two layers:
- 114 hand-authored pairs, each written from a real corpus passage with the
  answer taken directly from its content.
- 2,181 model-generated pairs, produced by prompting the OceanGPT base model
  itself for JSON question/answer pairs over anchor-term-selected passages
  from the corpus, then filtered for topicality and self-containedness
  (dropping journal-metadata and off-topic pairs, and questions that only
  make sense next to a figure the model never saw).

Each QA pair is prompt/answer structured, with the loss masked to the answer
tokens only — this is the standard fix for the off-topic-drift failure mode,
as opposed to continuation (raw-text) fine-tuning.

**LoRA config:** rank 16, alpha 32, applied to the attention/MLP projection
layers, trained to convergence with early stopping on held-out validation
loss (best checkpoint: val_loss 1.85).

**RAG index:** 119,762 text chunks (`rag/chunks.jsonl` + TF-IDF matrix) over
the same 10-theme corpus, for grounding answers in real source passages
instead of relying on the model's recall alone.

**What this is not:** not a fully human-verified corpus (the model-generated
QA layer carries a small accepted amount of residual topical noise), and not
a verified source of engineering fact — check anything load-bearing
independently.

**How it compares to the plain base model:** two rounds of scored comparison
against the bare base model and an earlier continuation-only adapter both
show this instruction adapter roughly doubling total score (round 1: 84/120
vs 42/120 base, 56/120 continuation-only; round 2, on a test set built from
passages never touched by training, validation, or checkpoint selection of
any adapter: 79/120 vs 41/120 base, 51/120 continuation-only, p<0.001 both
ways). RAG grounding helps the plain base model significantly but shows no
significant benefit on top of this instruction adapter with either TF-IDF
or dense (embedding) retrieval. Full methodology, generations, scores, and
the paired-significance stats are in `eval/`; the full writeup is in
`MarineSparkLM-4B_paper.docx`.

## What's in this package

```
app.py                     Gradio web UI: "Marine corpus" tab (uses rag.py / TF-IDF by
                           default) + "Upload your own documents" tab (upload_rag.py)
rag.py                     TF-IDF retrieval over the corpus (used by app.py)
rag_dense.py               Dense (sentence-embedding) retrieval, not wired into the main tab
upload_rag.py              Per-session RAG over user-uploaded documents: dense retrieval
                           + cross-encoder reranker, feeding a 4-bit quantized backbone
build_dense_index.py       Rebuilds the dense index from rag/chunks.jsonl
adapter/best_adapter/      LoRA adapter weights (~137MB)
rag/                       chunks.jsonl, tfidf.npz, idf.npy, vocab.json, dense_embeddings.npy
data/qa_seed.jsonl         The 2,295 QA pairs the adapter was trained on
eval/                      Comparison harness + results vs. base/continuation models,
                           four rounds (including two leakage-free held-out test sets)
MarineSparkLM-4B_paper.docx  Full writeup (background, training, results, comparison)
requirements.txt
```

### Upload your own documents (retrieval-augmented generation, small backbone)

The second tab lets you upload `.txt`/`.pdf` files and ask questions grounded
in them, instead of the fixed marine corpus. The design goal is to get most
of the accuracy of a large model at a fraction of the inference cost, by
making retrieval do the heavy lifting rather than the model's own recall:

1. **Index** -- each uploaded file is chunked (1200 chars, 150-char overlap)
   and embedded with `BAAI/bge-small-en-v1.5`, in memory for that session
   only (nothing is written to disk or shared across sessions).
2. **Retrieve** -- dense cosine similarity over all chunks returns the top 20
   candidates (cheap, coarse recall).
3. **Rerank** -- a cross-encoder (`BAAI/bge-reranker-base`) reads
   query+passage together and keeps the top 3-5 (the step doing the actual
   precision work).
4. **Generate** -- those passages are the entire context given to a 4-bit
   NF4-quantized load of the same 4B-parameter base model + instruction
   adapter (~3GB instead of ~8GB; validated in `eval/quantization_benchmark.py`
   to give comparable output quality to bf16).

This tab's backbone is loaded lazily (only once you use it) and is a
separate model instance from the main tab's bf16 model, so both tabs work
independently in the same running app.

The base model (~8GB) is **not** bundled — it downloads automatically from
Hugging Face the first time you run the app.

## How to run

**1. Requirements:** Python 3.10+, a CUDA GPU strongly recommended (the base
model is a 4B-parameter model; CPU inference will be slow but works).

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Run the app:**
```bash
python app.py
```
This downloads `zjunlp/OceanGPT-basic-4B-Instruct` from Hugging Face on
first run (cached afterward in `~/.cache/huggingface`) and loads the LoRA
adapter from `adapter/best_adapter/`. Open the local URL Gradio prints
(default `http://127.0.0.1:7860`).

If you already have the base model downloaded locally, skip the download by
pointing at it instead:
```bash
OCEANGPT_BASE_DIR=/path/to/OceanGPT-basic-4B-Instruct python app.py
```

**4. Use it:** type a prompt, adjust generation settings if you like, and
check "Ground with retrieval (RAG)" for anything factual — it retrieves real
passages from the corpus and primes generation with them before the model
answers.

### Loading the adapter in your own code

Directly from the Hugging Face Hub (no local files needed):
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

adapter_id = "<hf-username>/MarineSparkLM-4Bv1"  # link added once published
tok = AutoTokenizer.from_pretrained(adapter_id)
model = AutoModelForCausalLM.from_pretrained(
    "zjunlp/OceanGPT-basic-4B-Instruct", dtype="bfloat16", device_map="cuda"
)
model = PeftModel.from_pretrained(model, adapter_id)
```

Or from a local copy, once you've extracted the release zip (see "Getting
the model weights" above) over this checkout:
```python
model = PeftModel.from_pretrained(model, "adapter/best_adapter")
```

## Limitations

- The adapter was trained independently of the base model's own tuning —
  not stacked with any other adapter.
- Not a verified source of engineering fact; treat outputs as domain-biased
  drafting assistance, not ground truth.
