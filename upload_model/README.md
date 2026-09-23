---
license: other
license_name: indic-open-model-license-v1.0
license_link: https://huggingface.co/bodhan-ai/indic-translate/blob/main/indic-open-license.md
base_model: bodhan-ai/indic-translate
tags: [translation, marathi, lora, peft, indic]
language: [en, mr]
---

# indic-translate Marathi LoRA — experiment artifacts

LoRA adapters and evaluation artifacts from fine-tuning
[`bodhan-ai/indic-translate`](https://huggingface.co/bodhan-ai/indic-translate)
on English–Marathi data.

**This is a derivative work of `bodhan-ai/indic-translate` (Bodhan AI / AI4Bharat),
used and redistributed under the Indic Open Model License v1.0.** The adapters
here are not usable alone; they require the base model at load time.

## What is here

| path | contents |
|---|---|
| `generations/` | raw model outputs per direction — the primary artifact |
| `metrics/` | chrF++ (word_order=2) + BLEU with bootstrap CIs and sacrebleu signatures |
| `checkpoint-*/` | LoRA adapters, pushed during training |

Raw generations are kept deliberately: metrics can be recomputed from
generations, but generations cannot be recovered from metrics.

## Experiment

Three matched 25k-pair training arms isolating two axes:

| arm | corpus | quality | domain |
|---|---|---|---|
| A | Samanantar (structural filters only) | noisy | general/news/govt |
| B | BPCC_cleaned (LaBSE ≥ 0.9) | clean | general/news/govt |
| C | shiksha (NPTEL lectures) | clean-ish | education |

Training data was de-duplicated against IN22-Gen before use.

## Caveat on metrics

chrF++ differences below roughly ±0.7 are inside the test-set bootstrap CI at
n=1024 and should not be read as real. The ±0.06 figure quoted in the toolkit is
decode-repeat noise, a different and much smaller quantity.
