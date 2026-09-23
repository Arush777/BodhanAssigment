# Fine-tuning Bodhan `indic-translate` on Marathi

AI Research Engineer take-home, AI4Bharat / IIT Madras.
Task: fine-tune one Bodhan AI model on Marathi. This is the **Machine Translation** track.

---

## 1. The problem with the obvious approach

The assignment says "fine-tune on Marathi." The trap is that
[`bodhan-ai/indic-translate`](https://huggingface.co/bodhan-ai/indic-translate)
**already translates Marathi well** — `mar_Deva` is one of its 22 trained
languages, and the model card reports chrF++ 50.2 (en→mr) / 62.3 (mr→en) on
IN22-Gen.

So fine-tuning it on mined bitext with a LoRA learning rate of 1e-4 is a
textbook catastrophic-forgetting setup. "Raise the score" is a coin flip, and a
run that comes back 1.2 chrF++ lower with no explanation is not a result.

The assignment is explicit that it is **not scored on metrics** — it is scored
on getting a run working end to end, code structure, and judgement. So the
question I chose to answer is one that produces a finding either way:

> **How much adaptation can you buy from a strong instruction-tuned MT model,
> and what does it cost elsewhere?**

---

## 2. Experimental design

One fine-tune tells you nothing about *why* a score moved. Three explanations
fit any single negative result: the data was noisy, the data was the wrong
domain, or fine-tuning always hurts. Separating them requires changing one
variable at a time.

| Arm | Corpus | Quality | Domain |
|---|---|---|---|
| **A** | [`ai4bharat/samanantar`](https://huggingface.co/datasets/ai4bharat/samanantar) (structural filters only) | noisy | general / news / govt |
| **B** | [`SPRINGLab/BPCC_cleaned`](https://huggingface.co/datasets/SPRINGLab/BPCC_cleaned) (LaBSE ≥ 0.9) | clean | general / news / govt |
| **C** | [`SPRINGLab/shiksha`](https://huggingface.co/datasets/SPRINGLab/shiksha) (NPTEL lectures) | clean | **education** |

- **A vs B** holds domain constant → isolates **data quality**.
- **B vs C** holds quality constant → isolates **domain**.

All three are 25,000 pairs, identical hyperparameters, identical step count,
identical seed. Three independent LoRA adapters, each starting from the same
unmodified base — not sequential, not merged.

### Evaluation grid

Every model is scored on every test set, so no arm is judged only on its own
distribution:

| | IN22-Gen `mar` | `edu` holdout | `news` holdout | `hin`/`guj`/`tam` |
|---|---|---|---|---|
| base | 1024 | 500 | 500 | 300 ea |
| +A / +B / +C | 300 | 500 | 500 | 300 ea |

The cross-lingual columns are a **damage probe**: does Marathi-only fine-tuning
degrade Hindi (same script, same family), Gujarati (different script, same
family) or Tamil (different script, different family)? IN22-Gen is
multi-parallel, so the same source sentences across languages make this a
controlled comparison at no extra data cost.

---

## 3. Data pipeline

### Structural filtering is a hygiene pass, not a quality filter

`src/mt_marathi/filters.py` applies ten structural checks. Over Samanantar's
3,627,480 Marathi pairs it keeps **95.2%** — and that number is the diagnosis,
not the result. Structural checks cannot see meaning. From a random sample of 12:

```
eng: "Pakistan goalkeeper saved the shot from Akashdeep Singh."   [field hockey]
mar: "अफगाणिस्तानकडून फिरकीपटू राशिद खानने..."                      [cricket]
```

Well-formed on both sides, correct scripts, length ratio 1.5 — passes every
filter. **Semantic misalignment is the dominant failure mode in mined corpora**,
and it needs a meaning comparison, not a shape comparison.

### Blind validation of the corpora

Rather than assume the filtered corpora were better, I measured it. 48 pairs —
16 from each corpus — shuffled together with provenance stripped, judged
independently as CORRECT / PARTIAL / WRONG:

| Corpus | CORRECT | PARTIAL | WRONG | adequate % |
|---|---|---|---|---|
| BPCC_cleaned | 13 | 3 | **0** | 100% |
| shiksha | 9 | 7 | **0** | 100% |
| Samanantar (raw) | 9 | 3 | **4** | 75% |

All four `WRONG` pairs came from Samanantar. Honest statistics: Samanantar vs
BPCC alone is p = 0.051 (**not** significant at 0.05); pooled against both
filtered corpora it is p = 0.0094. Wilson 95% CIs overlap — Samanantar
[10%, 49%], filtered [0%, 19%]. The ordering is consistent; the precise rate is
not pinned down at n=16.

**The more useful finding is that the two corpora fail differently.** Samanantar
fails by *misalignment* (4 WRONG). Shiksha fails by *degraded fidelity within
correct alignment* (0 WRONG, but the most PARTIAL — "narrow skills" rendered as
अनंत, "infinite"). That predicts different damage profiles: Samanantar should
teach off-topic fluency, shiksha should teach on-topic imprecision.

### Test-set contamination, and a bug in the check for it

Training data is de-duplicated against IN22-Gen. The first implementation
reported 0 overlaps. A **positive control** — feeding it genuine IN22 sentences
with mangled case and punctuation — showed it caught **0 of 3**. Cause:
`normkey` stripped punctuation but never collapsed the whitespace left behind,
so `"hello, world"` became `"hello  world"` and never matched. A de-duplicator
that matches nothing is indistinguishable from a clean corpus.

---

## 4. Method

**LoRA, not full fine-tuning.** 7.94B params in bf16 with AdamW needs ~128 GB of
weights + gradients + optimizer state. A single 80 GB A100 cannot hold it.

**Verified trainable parameters: 100,999,168 of 8,042,100,000 — 1.2559%.**

`target_modules: "all-linear"` with the vision/audio towers and `lm_head`
excluded. Two reasons an explicit `["q_proj","v_proj"]` list would be wrong here:

1. **`num_kv_shared_layers: 18`** — layers 24–41 reuse KV from layer 23 and have
   **no `k_proj`/`v_proj` at all**. Counts: `q_proj`/`o_proj`/`gate`/`up`/`down`
   = 42 each, `k_proj`/`v_proj` = **24 each**. A `v_proj` target silently adapts
   57% of the network with no warning.
2. Gemma 4 carries `per_layer_input_gate` / `per_layer_projection` beside the
   usual projections, which a standard list misses.

The towers are excluded because translation is text-only — of 2,094 tensors only
678 are the language model; 744 are audio and 656 vision. They would receive
zero gradient. They also use `Gemma4ClippableLinear`, which is not an
`nn.Linear` subclass and makes PEFT raise.

**The prompt contract is not reimplemented.** `bodhan_genai.mt.data.render` owns
the instruction phrasing, the target-language naming and direction reversal. A
hand-written prompt still produces fluent output — just measurably worse output,
with nothing in the logs to say so.

### Memory budget (measured against the toolkit's own validated setting)

The dominant term is not the model. `vocab_size` is **262,144**, so the fp32
logits tensor plus its gradient is 16 GB at batch 8 × seq 1024 — larger than the
LoRA parameters, optimizer states and activations combined.

I changed `per_device_train_batch_size` from 1 to 8 and `max_seq_length` from
8192 to 1024. That is not a guess: the shipped config documents 1 × 8192 as
"memory-safe on an 80 GB card", and 8 × 1024 = 8,192 token-positions per step —
**identical**. Same logits tensor, same activation volume, inside a validated
envelope, but with the GPU actually busy on short sentence bitext.

---

## 5. Results

### Baseline reproduces the model card

| Direction | ours | card | 95% CI | BLEU |
|---|---|---|---|---|
| en→mar_Deva | **50.15** | 50.2 | [49.5, 51.0] | 14.35 |
| mar_Deva→en | **62.39** | 62.3 | [61.5, 63.2] | 37.67 |

An independently built pipeline landing within 0.1 chrF++ of the published
figure validates the prompt contract, IndicNLP pre-tokenisation, chrF++
`word_order=2` and greedy decoding simultaneously. I had predicted this would
*not* reproduce; being wrong here is the good outcome.

Cross-lingual baseline: en→hin 52.50, hin→en 61.88, en→guj 51.50, guj→en 63.12,
en→tam 48.87, tam→en 56.11. Degenerate outputs: 0–1 across all 8 directions.

### The noise floor is 13× what the toolkit documents

Bootstrap 95% CI at n=1024 is **±0.77 chrF++**. The toolkit's ±0.06 is
*decode-repeat* noise (same model twice, differing because batching reorders
float reductions) — not test-set sampling noise. Any difference under ~0.8
chrF++ between two models on this benchmark is not a result.

### Why chrF++ and not BLEU, argued from our own data

**en→tam: chrF++ 48.87 but BLEU 8.74.** Tamil is agglutinative, so word-level
BLEU collapses while character-level chrF++ stays informative. The same effect
appears within our target: Marathi BLEU 14.35 vs Hindi 24.74 despite nearly
identical chrF++ (50.2 vs 52.5) — Marathi is morphologically richer, so exact
word matching punishes it harder. This is why IndicTrans2 selects chrF++ as its
primary metric, and now it is evidenced rather than cited.

*(Fine-tuned arm results, degradation curves and the cross-lingual damage table
are appended as jobs complete.)*

---

## 6. What went wrong

Recorded in full in [DECISIONS.md](DECISIONS.md). The load-bearing ones:

1. **The toolkit's installer builds flash-attn from source by default.** Its
   `install.sh` defaults to `--extras all`, pulling TTS training dependencies.
   flash-attn ships as an sdist, so it compiles CUDA kernels for tens of minutes
   — at $2.50/hr, for a library this model cannot use (7 global layers at
   `head_dim 512`, above FA2's 256 cap; the card specifies `sdpa`).
   `--extras all-mt` cuts install from ~15 minutes to ~2.
2. **`early_stopping_patience: 999` does not disable early stopping.** It still
   constructs an `EarlyStoppingCallback`; large patience means it never *fires*,
   not that it is absent. Combined with dropping `metric_for_best_model`, that
   asserts at `on_train_begin`.
3. **`sacrebleu.CHRF().get_signature()` raises until the metric has scored
   something**, and `CHRF()` defaults to `word_order=0` — plain chrF, not chrF++.
4. **Artifact ordering saved the experiment.** Generations and metrics push after
   *every direction* rather than batching at the end, so when training crashed
   the baseline was already banked. A crash cost the probe, not the run.

---

## 7. Stated limitations

- **"Education domain" means university STEM lecture transcript**, not school
  textbook prose. No aligned NCERT/Balbharati parallel corpus exists publicly in
  any Indian language pair.
- **The shiksha en–mr slice has only 31 distinct courses.** Holding out 6 leaves
  25 for training, so topical coverage is narrow and the domain result should not
  be over-generalised.
- **Samanantar's licence is contradictory across sources** (CC0 on AI4Bharat's
  page, CC-BY-NC-4.0 on the HF card). The derived corpus is therefore kept in a
  private repo rather than redistributed.
- **n=16 per corpus** in the blind adequacy check. It establishes ordering, not
  precise error rates.
- **25k pairs per arm, 800 steps.** Compute-bound by choice: a clean 25k answers
  the question; 3.6M answers the same question for money that was not available.

---

## 8. Reproducing

```bash
uv venv .venv && . .venv/bin/activate && uv pip install -r requirements.txt
PYTHONPATH=src python -m mt_marathi.prepare_data     --target-rows 25000
PYTHONPATH=src python -m mt_marathi.prepare_bpcc     --target-rows 25000
PYTHONPATH=src python -m mt_marathi.prepare_shiksha  --target-rows 25000
```

Training and evaluation run on a single A100 80 GB via HF Jobs; see
[`jobs/`](../jobs). Artifacts: <https://huggingface.co/Arushhh/indic-translate-mr-lora>
