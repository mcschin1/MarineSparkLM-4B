# Comparison harness: MarineSparkLM-4B vs. base and continuation-only models

Answers whether the instruction fine-tune actually helps, by comparing four
conditions on the same held-out prompts:

1. **base** -- plain OceanGPT-basic-4B-Instruct, no adapter.
2. **continuation_adapter** -- the continuation-only LoRA (Section 5 of the paper).
3. **instruction_adapter** -- MarineSparkLM-4B (Section 6), no retrieval.
4. **instruction_adapter_rag** -- MarineSparkLM-4B with RAG grounding (top-3 passages).

## Eval set

20 questions (2 per theme, all 10 themes), drawn from the **held-out
validation split** of `data/qa_dataset/qa_seed.jsonl` (`select_eval_set.py`
reproduces the exact split used at training time:
`torch.Generator().manual_seed(42)`, 10% held out). None of these questions
or their reference answers were seen by any adapter during training.

## Method

`generate_comparison.py` runs all 20 prompts through all 4 conditions
(greedy decoding, same settings as the app) and writes
`comparison_outputs.jsonl`. Each of the 80 generations was then read and
scored by hand against its reference answer on three 0-2 dimensions
(`comparison_scores.csv`):

- **Accuracy**: does it align with the reference answer's actual facts (not
  necessarily verbatim)?
- **Topicality**: does it stay on the correct marine-engineering subtopic,
  or drift to an unrelated domain?
- **Specificity**: concrete domain detail, or vague filler?

This is a single-rater manual rubric, not an automated LLM-judge pipeline --
appropriate at n=20, and avoids introducing a second model's bias into the
judging. A larger run should sample more of the 229-row held-out set and/or
use a second rater. The 20-item eval set is also drawn from the same
validation split used for early-stopping/checkpoint selection during
training, which is a real form of selection leakage not yet fixed here --
noted as a follow-up, not something this round corrects.

## RAG index leakage (found and fixed)

The RAG condition originally retrieved from the full `rag/chunks.jsonl`
index, which includes the exact source document each eval question's
reference answer was drawn from -- confirmed directly: all 20/20 eval
`doc_id`s are present in the RAG index. That means the RAG condition could
retrieve a question's own source passage and effectively look up the
answer, which is a much easier task than the other three conditions face
and inflates RAG's apparent benefit in a way that doesn't reflect how RAG
performs on an arbitrary real query (which usually isn't an exact chunk
already in the index). `rag.py`'s `retrieve()` now takes an
`exclude_doc_ids` parameter (used only by this eval harness, never by the
production app) to exclude a question's own source document before
scoring; `generate_comparison.py` passes the question's own `doc_id`.
Re-running the RAG condition with this fix changed 14 of the 20 generations.
The results below are the leakage-corrected numbers; the original
(leaked) RAG outputs are kept at `comparison_outputs_with_leakage.jsonl.bak`
for comparison.

## Results

| Condition | Accuracy /40 | Topicality /40 | Specificity /40 | Total /120 |
|---|---|---|---|---|
| base | 4 | 29 | 9 | 42 |
| continuation_adapter | 4 | 33 | 19 | 56 |
| instruction_adapter | 14 | 38 | 32 | 84 |
| instruction_adapter_rag (leakage-corrected) | 16 | 38 | 27 | 81 |

Paired Wilcoxon signed-rank test + bootstrap 95% CIs on the per-item totals
(`compute_stats.py`, n=20, 10,000 resamples -- treat p-values as a direction
check, not a strong significance claim at this sample size):

| Comparison | Mean diff | 95% CI | Wilcoxon p |
|---|---|---|---|
| base vs. instruction_adapter | +2.10 | [+1.40, +2.75] | 0.0005 |
| continuation_adapter vs. instruction_adapter | +1.40 | [+0.85, +1.95] | 0.0008 |
| instruction_adapter vs. instruction_adapter_rag | -0.15 | [-0.50, +0.20] | 0.4054 |
| base vs. continuation_adapter | +0.70 | [+0.05, +1.40] | 0.0681 |

Instruction fine-tuning's improvement over both the base model and the
continuation-only adapter is large and clearly significant at this sample
size (p < 0.001 both ways). **RAG's benefit is not**: once the leakage is
removed, instruction_adapter_rag actually scores slightly *lower* than
plain instruction_adapter (81 vs 84), and the difference is not
distinguishable from noise (p = 0.41, CI crosses zero). The original
unleaked comparison (88 vs 84) made RAG look like a clear win; it wasn't --
most of that gain was the model retrieving its own answer. This doesn't
mean RAG is useless in general (a real user query usually isn't an exact
chunk already sitting in the index, so this leakage pathway mostly can't
happen in production use), only that this particular eval can't demonstrate
a RAG benefit, because the eval questions were sourced from the very corpus
RAG retrieves from.

The clearest single example is item 18 (`sensors_communications`, "What is
the primary goal of the research described in the document?"): the base
model answered about anti-cancer marine natural compounds and the
continuation adapter answered about building insulation thermal performance
-- both a complete domain drift on a vague prompt -- while both instruction
conditions stayed within marine engineering (wrong specific answer, but
on-topic). This reproduces the original ballast-water drift failure on a
new, held-out example, not just the one case that originally motivated
Section 6.

One failure mode shared by all four conditions: item 11 (`offshore_energy`,
the specific metric used to optimize a neural network) is answered wrong by
every condition, naming MSE or similar instead of the reference's balanced
accuracy score -- none of the models have real knowledge of this one
paper's methodology, as expected. Also, on two of the 20 items (13 and 16)
the RAG condition produced output identical to plain instruction adapter --
evidence that retrieval returned no useful passage for those vague queries
and the model fell back to the same answer as without RAG.

## Why MarineGPT is not in this comparison

OceanGPT is already covered: the "base" condition above *is* plain
OceanGPT-basic-4B-Instruct. MarineGPT (HKUST) was considered separately and
excluded, because it isn't a comparable model for this eval. MarineGPT is a
vision-language model (BLIP-2 / MiniGPT-4 architecture, Vicuna-7B/13B or
Gemma-2B/7B backbone plus a vision encoder) built for captioning and
answering questions about marine images, with its own non-`transformers`
inference pipeline and multi-gigabyte checkpoints distributed outside
Hugging Face. This eval set is text-only with no associated images, so
running MarineGPT against it would test it entirely outside the modality it
was built for -- a wrong or generic answer would reflect the missing image
input, not its actual marine-domain knowledge. A fair comparison against
MarineGPT specifically would need a separate image-based evaluation set,
which this work does not include.

## Why Llamarine is not in this comparison

Llamarine (Aitomatic + Furuno Electric, arXiv 2503.00203) is a
maritime-industry LLM (navigation, regulatory compliance) and is text-only,
otherwise a direct architectural fit for this harness. It was excluded for
practical reasons on this hardware, not a modality mismatch like MarineGPT:

- No official or quantized release exists. The only copy found is a
  third-party upload (`pentagoniac/llamarine` on Hugging Face), a full merge
  of Llama-3.1-70B with Llamarine's maritime LoRA adapters, 141GB in bf16.
  Provenance could not be confirmed (attribution is inconsistent between the
  paper and the upload, and the listed license tag does not match the base
  model's actual license).
- 141GB exceeds this machine's 121GB total unified memory, so a direct bf16
  load is impossible. A 4-bit quantized load would fit (~35-40GB) but still
  requires downloading the full 141GB first.
- Measured download throughput on this connection: single-stream 2-3MB/s,
  plateauing around 5.5-6MB/s with multiple parallel connections (a
  connection-level cap, not server throttling). At that rate the download
  alone is roughly 7-8 hours, with no guarantee of a stable connection for
  that long.

Given the unverified provenance plus a multi-hour cost for a single
third-party upload, this comparison was not run. It remains a good
candidate for a future round if an official or pre-quantized release
appears, or on hardware with more memory/bandwidth headroom.

## Round 2: a genuinely held-out test set, plus three new conditions

Round 1 (above) had a real validity problem beyond the RAG leakage: its
20-prompt eval set was drawn from the **validation** split of
`qa_seed.jsonl`, the same split used for early stopping (checkpoint
selection) during training. Evaluating "generalization" on data that
helped pick the checkpoint is a form of selection leakage.

**Fix.** Rather than retrain on a new 3-way split (which would force
swapping the production adapter), `select_fresh_test_sources.py` +
`generate_qa_pairs.py` built a new 20-question test set from corpus
passages whose `doc_id` does not appear anywhere in `qa_seed.jsonl` (not
train, not val) -- `fresh_test_set.jsonl`. These questions were never seen
in training, validation, or checkpoint selection of any adapter, and the
production adapter was never touched.

**Three new conditions**, run via `generate_comparison_v3.py` against the
same fresh set:

- **base_rag_tfidf** -- the plain base model with TF-IDF RAG, isolating
  whether retrieval grounding helps independent of instruction tuning.
- **instruction_adapter_synthetic_only** -- a separate adapter trained on
  only the 2,181 model-generated QA pairs (`qa_seed_synthetic_only.jsonl`),
  excluding the 114 hand-authored pairs, isolating whether the hand-authored
  seed set matters.
- **instruction_adapter_rag_dense** -- the instruction adapter with dense
  (sentence-embedding) retrieval instead of TF-IDF. `build_dense_index.py`
  encodes the same 119,762 chunks with `BAAI/bge-small-en-v1.5`
  (`rag/dense_embeddings.npy`); `rag_dense.py` mirrors `rag.py`'s interface.
  Calibrated empirically the same way as TF-IDF's floor (4 topical + 4
  off-topic probe queries): genuine matches score ~0.78-0.86 cosine
  similarity with the bge query-instruction prefix applied, off-topic
  queries ~0.56-0.72, so `min_score=0.75` separates them.

All 7 conditions (the original 4 plus these 3) were re-run against the
fresh set and scored with the same rubric (`comparison_scores_v3.csv`).

### Results

| Condition | Accuracy /40 | Topicality /40 | Specificity /40 | Total /120 |
|---|---|---|---|---|
| base | 3 | 24 | 14 | 41 |
| continuation_adapter | 1 | 28 | 22 | 51 |
| base + RAG (TF-IDF) | 5 | 32 | 18 | 55 |
| instruction_adapter + RAG (TF-IDF) | 9 | 34 | 30 | 73 |
| instruction_adapter | 8 | 40 | 31 | 79 |
| instruction_adapter + RAG (dense) | 8 | 40 | 30 | 78 |
| instruction_adapter (synthetic-only) | 9 | 40 | 33 | 82 |

Paired Wilcoxon + bootstrap 95% CIs (`compute_stats.py comparison_scores_v3.csv ...`):

| Comparison | Mean diff | 95% CI | Wilcoxon p |
|---|---|---|---|
| base vs. instruction_adapter | +1.90 | [+1.45, +2.25] | 0.0001 |
| continuation_adapter vs. instruction_adapter | +1.40 | [+1.00, +1.85] | 0.0002 |
| base vs. base + RAG (TF-IDF) | +0.70 | [+0.40, +1.00] | 0.0021 |
| instruction_adapter vs. synthetic-only | +0.15 | [-0.15, +0.50] | 0.3340 |
| instruction_adapter vs. +RAG (TF-IDF) | -0.30 | [-0.60, +0.00] | 0.0833 |
| instruction_adapter vs. +RAG (dense) | -0.05 | [-0.15, +0.00] | 0.3173 |
| +RAG (TF-IDF) vs. +RAG (dense) | +0.25 | [-0.10, +0.60] | 0.1655 |

**The core result holds, now on a genuinely clean test set.** Instruction
tuning's improvement over both the base model and the continuation-only
adapter is large and highly significant (p < 0.001 both ways) -- this was
the main thing round 1's validation-split leakage put in question, and it
survives on data that was never touched by training or model selection.

**RAG helps the base model, not clearly the instruction adapter.** Adding
TF-IDF RAG to the plain base model is a significant improvement (p=0.002):
retrieval grounding has real, measurable value on its own. But adding RAG
(either TF-IDF or dense) on top of the already instruction-tuned adapter
shows no significant effect (p=0.08 and p=0.32), with TF-IDF trending
slightly negative and dense roughly neutral. The most plausible reading:
instruction tuning already captures most of what RAG was providing
(topicality anchoring, commitment to a concrete on-topic answer), so the
marginal value of retrieval on top of it is small on this prompt mix.

**Dense retrieval trends better than TF-IDF, not yet conclusively.** The
aggregate difference (+0.25, p=0.17) isn't significant at n=20, but one
item makes the quality difference concrete: for the offshore_structures
prompt "What is the main focus of the article?" (a coupled
mooring-viscous-flow solver paper), TF-IDF retrieved an entirely
unrelated chunk about "four-parameter semitoric systems" (a pure-math
dynamical-systems topic, likely a keyword collision), producing a correctly
on-topic-sounding but garbage-grounded answer. Dense retrieval found
nothing above its calibrated floor for the same query and the model fell
back to its own (also wrong, but at least on-topic) answer instead of
being misled by a bad retrieval.

**The 114 hand-authored pairs show no detectable benefit over the 2,181
synthetic-only pairs, on this eval.** `instruction_adapter_synthetic_only`
scored *slightly higher* (82) than the full `instruction_adapter` (79),
but the paired difference is not significant (p=0.33) -- a genuinely
surprising negative result rather than a confirmation that the hand-authored
seed set is pulling weight. This doesn't mean the hand-authored set is
useless (it was necessary to discover the 114-pair-only overfitting failure
that motivated the scale-up in the first place), only that on this
20-question sample its presence in the final training mix isn't
distinguishable from its absence.

## Round 3: validation checks

Six follow-up checks, addressing measurement validity, cost, and robustness
rather than adding new eval conditions to the main comparison table above.

### Automatic metric vs. human score

`correlate_auto_metric.py` computes bge-small-en-v1.5 cosine similarity
between each generation and its reference answer, and correlates it against
the existing hand-scored `total` for every generation already collected
(round 1 + round 2, n=220):

| Set | n | Pearson r | p |
|---|---|---|---|
| Round 1 | 80 | 0.342 | 0.0019 |
| Round 2 | 140 | 0.666 | <0.0001 |
| Combined | 220 | 0.547 | 1.4e-18 |

Moderate correlation -- useful for cheaply triaging large generation batches
(e.g. picking which items most need a human look), but not a substitute for
the manual rubric: embedding similarity to a reference answer doesn't capture
topicality drift or factual correctness the way the human rater's
accuracy/topicality/specificity breakdown does.

### Inference cost

`measure_inference.py` times a single forward pass through the base model
and the instruction adapter on the fixed hull-resistance prompt (greedy
decoding, warmup generation excluded from timing):

| Metric | Base model | + instruction adapter |
|---|---|---|
| Load time | 43.6 s | +0.65 s |
| Peak GPU memory (load) | 8.31 GB | 8.31 GB |
| Peak GPU memory (generation) | 8.22 GB | 8.22 GB |
| Throughput | -- | 17.6 tokens/sec |

The LoRA adapter adds well under a second of load time and no measurable
memory over the base model's own footprint -- a single-device, single-prompt
measurement, not a batched-throughput or multi-GPU benchmark.

### LoRA rank sweep (r=8 / 16 / 64)

Same hyperparameters as production (`--lora-alpha 32`, same data/schedule),
varying only rank, scored by best validation loss:

| Rank | Best val_loss | Step |
|---|---|---|
| 8 | 1.853 | 150 |
| 16 (production) | 1.853 | 150 |
| 64 | 1.850 | 250 |

All three ranks converge to essentially the same loss (differences are
within noise). No benefit from increasing capacity beyond r=8 at this
dataset size (2,295 QA pairs) -- the bottleneck is data, not adapter
capacity.

### Catastrophic forgetting (MMLU)

`mmlu_forgetting_check.py` scores the base model and instruction adapter on
the same random 200-question MMLU subset, using standard next-token-logit
comparison (not free-text generation + regex parsing, which gave spuriously
bad results on a first attempt -- see Errata below):

| | Correct / 200 | Accuracy |
|---|---|---|
| Base model | 117 | 0.585 |
| Instruction adapter | 126 | 0.630 |
| Delta | +9 | +0.045 |

No forgetting detected. The small positive delta is most plausibly noise
rather than a real improvement in general knowledge, but it rules out the
adapter trading general capability for domain specificity.

### External baseline: Qwen3-4B-Instruct-2507

OceanGPT-basic-4B-Instruct (the base model used throughout this project) is
itself built from Qwen3-4B-Instruct-2507. `generate_qwen3_comparison.py`
runs the *plain, un-pretrained* Qwen3-4B-Instruct-2507 through the same
harness and rubric against `fresh_test_set.jsonl`, isolating whether
OceanGPT's own ocean-domain pretraining adds value relative to a same-size
generic instruct model:

| Condition | Accuracy /40 | Topicality /40 | Specificity /40 | Total /120 |
|---|---|---|---|---|
| qwen3_instruct | 6 | 32 | 18 | 56 |
| qwen3_instruct + RAG (TF-IDF) | 3 | 29 | 23 | 55 |

For reference, from the round 2 table above: base (OceanGPT) = 41,
continuation_adapter = 51, instruction_adapter = 79.

Plain Qwen3-4B-Instruct **beats bare OceanGPT-basic-4B-Instruct** (56 vs.
41) and is comparable to the continuation-only adapter (51) and base+RAG
(55), but is **far below this project's own instruction adapter** (79).
Qualitatively, Qwen3 was noticeably more epistemically calibrated -- it
often explicitly declines to answer when a RAG-retrieved passage doesn't
actually address the question, rather than confidently hallucinating an
answer anyway (see `qwen3_readable.txt`). This complicates the assumption
that OceanGPT's domain pretraining was itself a clear improvement over a
strong generic base, while still cleanly confirming that this project's
instruction tuning contributes large, real value regardless of which base
model it starts from.

### Blind re-scoring (rater-bias check)

A stratified random sample of 40 generations (20 from round 1, 20 from
round 2, `random.seed(7)`) was re-scored blind -- condition label and
original score hidden -- and compared against the original scores:

| Metric | Value |
|---|---|
| Exact agreement | 34/40 (0.850) |
| Mean absolute difference | 0.150 |
| Max absolute difference | 1 (out of a 0-6 scale) |
| Pearson r (orig vs. blind) | 0.969 (p=1.0e-24) |
| Net bias (blind - orig) | -4 points total across 40 items (negligible) |

No meaningful rater bias: the original hand-scoring is highly reproducible
even without knowing which condition produced an answer.

### 4-bit quantization

`quantization_benchmark.py` compares the instruction adapter loaded in bf16
vs. 4-bit NF4 (`bitsandbytes`):

| | bf16 | 4-bit NF4 |
|---|---|---|
| Load time | 71.0 s | 74.6 s |
| Peak memory (load) | 8.31 GB | 3.12 GB |
| Peak memory (generation) | 8.22 GB | 3.03 GB |
| Throughput | 7.9 tok/s | 15.1 tok/s |

4-bit quantization cuts memory by 62% (8.3GB -> 3.1GB) with comparable
output quality on the sample prompt and *higher* throughput in this
single-request measurement -- the model fits comfortably on an 8GB consumer
GPU, which it would not in bf16.

### Errata: MMLU scoring bug

The first MMLU attempt generated free text (`max_new_tokens=5`) and
extracted an answer letter with a regex (`\b([ABCD])\b`), which gave the
base model 1/10 on a smoke test -- worse than random chance. Cause: models
often reason before stating a letter ("The correct answer is C
because...") or don't emit a bare letter within the token budget, so
first-match regex extraction was unreliable, not a real capability result.
Fixed by switching to the standard MMLU methodology (Hendrycks et al.):
compare next-token logits for the four answer-letter tokens directly via a
single forward pass, with no generation or text parsing.

## Round 4: an enlarged (120-question) test set, and a limit on the auto-metric

Round 2's fresh held-out set was only 20 questions. This round enlarges it
to 120 (12 per theme, 6x), re-running the 4 core conditions (base,
continuation_adapter, instruction_adapter, instruction_adapter + RAG
TF-IDF) and scoring with the automatic metric validated in Round 3, rather
than hand-scoring 480 generations -- the 3 ablation conditions added in
Round 2 (base+RAG, synthetic-only, dense RAG) already got clear null/negative
results at n=20 and were not re-run here, to keep cost proportionate to what
a larger set is actually for: more statistical power on the core claim.

**Building the set.** `select_fresh_test_sources_v2.py` picked 160
candidate passages (16/theme), excluding every `doc_id` already in
`qa_seed.jsonl` **and** in Round 2's `fresh_test_set.jsonl`, so this set has
zero document overlap with training, validation, checkpoint selection, or
the existing held-out set. `generate_qa_pairs.py` produced 287 raw
question-answer pairs from those passages. An automated filter dropped
bibliographic/metadata questions (title, authors, publication date -- these
test nothing about domain knowledge) and picked one question per unique
source document, down to 12/theme. A manual read-through of all 120 then
caught 10 the automated filter missed: keyword-collision false positives
(the same class of corpus noise already documented elsewhere in this
project) that happened to match a theme's anchor terms -- spacecraft
propulsion, automobile wind-tunnel testing, early machine-gun recoil
boosters, oarfish biology, an unrelated geopolitical conflict -- plus one
item with OCR-garbled math notation. Each was replaced with a manually
verified, on-topic question from the same theme's remaining candidates.
`fresh_test_set_v2.jsonl` is the resulting 120-question set.

### Results (automatic metric: bge-small cosine similarity to reference)

| Condition | Mean auto-score | 95% CI |
|---|---|---|
| base | 0.7324 | [0.7119, 0.7529] |
| continuation_adapter | 0.7000 | [0.6815, 0.7191] |
| instruction_adapter | 0.7596 | [0.7432, 0.7755] |
| instruction_adapter + RAG (TF-IDF) | 0.7461 | [0.7272, 0.7647] |

Paired Wilcoxon + bootstrap 95% CI on per-item auto-score differences
(n=120, 10,000 resamples):

| Comparison | Mean diff | 95% CI | Wilcoxon p |
|---|---|---|---|
| base vs. instruction_adapter | +0.0272 | [+0.0097, +0.0456] | 0.0118 |
| continuation_adapter vs. instruction_adapter | +0.0596 | [+0.0413, +0.0782] | 4.5e-08 |
| instruction_adapter vs. +RAG (TF-IDF) | -0.0135 | [-0.0282, +0.0005] | 0.2423 |
| base vs. continuation_adapter | -0.0323 | [-0.0483, -0.0160] | 4.4e-05 |

**The core result replicates at 6x the sample size.** Instruction tuning
beats both the base model and the continuation-only adapter by a wide,
highly significant margin -- the main claim this project rests on now has
far more statistical power behind it than the original n=20 rounds. RAG
still shows no significant benefit on top of the instruction adapter,
consistent with Round 2.

**But the base-vs-continuation_adapter row above is wrong, and that's worth
knowing.** The auto-metric says base (0.7324) beats continuation_adapter
(0.7000), p=4.4e-05 -- the *opposite* of Rounds 1-2, where
continuation_adapter clearly beat base on hand scoring (42 vs. 56 in Round
1, 41 vs. 51 in Round 2). A stratified 24-item hand-scored spot-check
(6 items per condition, real 0-6 rubric) confirms the auto-metric is wrong
here: the hand scores give continuation_adapter (3.00/6) over base
(2.17/6), the same direction as Rounds 1-2. Item-level correlation on this
spot-check is actually good -- Pearson r=0.619 (p=0.0012), Spearman
rho=0.680 (p=0.0003), n=24, slightly better than the combined r=0.547 from
Round 3 -- so the auto-metric isn't broken in general. The likely cause of
this one reversal: the reference answers were written by the base model
itself (documented since Round 1), so base's own phrasing naturally
resembles that reference more closely in embedding space regardless of
actual correctness, a self-similarity bias that only shows up when a
compared condition is the same model that wrote the references. Practical
takeaway: trust the auto-metric for ranking conditions in general and for
item-level triage, but not for a comparison against the specific model
that authored the reference answers -- that comparison needs hand scoring.

## Upload-RAG pipeline: a reranker floor, and a validation test

The app's "upload your own documents" tab (`upload_rag.py`) uses dense
retrieval + cross-encoder reranking + a 4-bit backbone, separate from the
main corpus pipeline above. Two follow-ups after it shipped:

**A relevance floor was missing, then miscalibrated, then fixed.**
`retrieve()` originally always returned its top-k reranked passages with
no floor, so an uploaded document that didn't cover a query would still
hand the model its "best" (but irrelevant) passages instead of saying so.
A first floor (0.01) was calibrated using reference-answer text standing
in for the uploaded document -- reference answers are far more lexically
similar to their own question than real document prose is, which inflated
scores and produced a floor that rejected 8 of 20 genuine matches in a
real test. Recalibrating against 20 genuine and 20 mismatched pairs built
from actual corpus passages (not reference answers) found the two score
distributions overlap substantially at the low end (genuine matches
0.00004-0.9999, mismatches 0.00004-0.0069) -- no threshold separates them
cleanly. `min_score=0.0005` keeps 16/20 genuine matches while accepting
3/20 mismatches as false positives; a real but weakly-matched document can
still fall below floor and generation falls back to ungrounded.

**Validation: upload-RAG vs. the main corpus pipeline on identical
questions.** 20 held-out questions (2/theme, from `fresh_test_set_v2.jsonl`)
were each run through `upload_rag.py` with that question's own real corpus
passage as the simulated upload (`generate_upload_rag_test.py`), and
compared hand-scored (real 0-6 rubric) against the existing
`instruction_adapter_rag_tfidf` generations for the same 20 questions
(from `comparison_outputs_v4.jsonl`).

| Condition | Total /120 | Mean |
|---|---|---|
| Main corpus RAG (TF-IDF, 119,762 chunks) | 72 | 3.60 |
| Upload RAG (single-document) | 91 | 4.55 |

Mean difference +0.95, 95% CI [+0.25, +1.75], Wilcoxon p=0.0284 (upload-RAG
wins 11/20, ties 5/20, loses 4/20). Full generations and per-item scores
are in `upload_vs_main_rag_readable.txt` and `upload_rag_vs_main_scores.csv`.

**This is not an apples-to-apples retrieval-difficulty comparison, and the
result should not be read as "upload-RAG is better."** Upload-RAG only had
to search the one document that actually contains the answer; the main
pipeline searched the full 119,762-chunk corpus and could retrieve an
unrelated chunk instead, which is exactly what happened on item 14 --
TF-IDF reproduced the same "semitoric systems" keyword-collision
contamination documented in Round 2, scoring 1/6, while upload-RAG's
single-document search couldn't make that mistake and scored 6/6. The
result supports the design's premise (constraining retrieval to a small,
verified-relevant context avoids this class of wrong-document failure),
but a fair comparison of retrieval difficulty would need the main
pipeline's task matched -- e.g. multiple candidate documents per upload --
which this test did not do.

Separately, 4 of the 20 items had every candidate rejected by the
reranker floor (0 hits) and fell back to ungrounded generation. Ungrounded
performance on those 4 was mixed (scores of 5, 2, 4, 6 vs. the matched
main-corpus condition's 3, 3, 4, 2) -- not catastrophic in this small
sample, but a reminder that the floor trades recall for precision, and a
real document can still be too weakly phrased relative to the question to
clear it.
