# Fine-tuning Bodhan `indic-translate` on Marathi

Take-home for the AI Research Engineer role (AI4Bharat, IIT Madras).
Task: fine-tune one Bodhan AI model on Marathi. This repo does the **Machine
Translation** track: a LoRA finetune of [`bodhan-ai/indic-translate`](https://huggingface.co/bodhan-ai/indic-translate)
on filtered English–Marathi bitext.

## The short version

`indic-translate` already supports Marathi and translates it well. So the
interesting question is not "can I raise the score" — it is **how much
adaptation you can buy from a strong instruction-tuned model before you start
damaging it.** That is what this repo measures.

## Layout

```
configs/           render + LoRA configs, each delta from the toolkit default annotated
src/mt_marathi/
  filters.py       quality filters for mined bitext, one filter per observed failure mode
  prepare_data.py  Samanantar -> filtered bitext JSONL
jobs/              HF Jobs launchers (single A100 80GB)
docs/              approach, decisions, results
```

## Running it

Data prep is local and free; training runs on a single A100 via HF Jobs.

```bash
uv venv .venv && . .venv/bin/activate && uv pip install -r requirements.txt
PYTHONPATH=src python -m mt_marathi.prepare_data --target-rows 25000
```

See [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md) and [docs/EXPLAINERS.md](docs/EXPLAINERS.md) for the full approach,
the reasoning behind each choice, and what went wrong.

## What this repo deliberately does not do

It does not reimplement the prompt contract. `bodhan_genai.mt.data.render`
owns the instruction phrasing, the target-language naming and the direction
reversal. A hand-written prompt still produces fluent output — just measurably
worse output, with nothing in the logs to say so.

## Explainers

The non-obvious parts of this model and toolkit — what LoRA actually attaches to,
why `all-linear` matters here, the two unrelated things called "checkpointing",
and why chrF++ over BLEU — are explained from first principles in
[docs/EXPLAINERS.md](docs/EXPLAINERS.md).
