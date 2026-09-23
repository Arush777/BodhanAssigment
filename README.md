# Marathi ↔ Dehwali Bhili translation

LoRA fine-tune of [`bodhan-ai/indic-translate`](https://huggingface.co/bodhan-ai/indic-translate)
for Marathi ↔ Bhili, addressing two reported failures: **numbers not translating**
and **short phrases translating badly**.

Bhili is not one of the base model's 22 supported languages, so this teaches the
model a language it has never seen rather than adapting one it knows.

## Result

The fine-tune beats a **COPY baseline** — echoing the input unchanged — in both
directions on all four test sets. That is the only comparison that means
anything here, because the *unmodified* model **loses** to copying on every
Marathi→Bhili set: Bhili is lexically close enough to Marathi that doing nothing
outperforms a model that has never seen the language.

| test set | direction | COPY | base | fine-tuned | FT − COPY |
|---|---|---|---|---|---|
| GENERAL | mar→bhb | 42.1 | 40.0 | **49.0** | **+6.8** |
| GENERAL | bhb→mar | 40.5 | 40.9 | **60.3** | **+19.8** |
| NUM_NAT | mar→bhb | 46.3 | 44.3 | **52.0** | **+5.7** |
| NUM_NAT | bhb→mar | 44.7 | 45.0 | **65.3** | **+20.6** |
| SHORT_NAT | mar→bhb | 42.6 | 39.3 | **47.1** | **+4.5** |
| SHORT_NAT | bhb→mar | 43.0 | 42.4 | **60.1** | **+17.1** |
| SHORT_HARD | mar→bhb | 33.5 | 31.5 | **40.5** | **+7.0** |
| SHORT_HARD | bhb→mar | 33.9 | 34.1 | **53.8** | **+19.9** |

chrF++ with `word_order=2`.

Spurious copy rate — of items whose reference differs from the source, how often
the model merely echoes it — falls from **41.9% → 8.5%** on general text and
**68.4% → 33.3%** on the hardest short-input cell.

Full analysis: **[docs/BHILI_REPORT.md](docs/BHILI_REPORT.md)**.

## Two findings that reframe the problem

**Part of the "number failure" is in the data.** 9.5% of references verbalise
numerals rather than preserving them (`20 → वीस`, `800 → आठहोव`) — a legitimate
Bhili prose convention. The human references themselves preserve source numerals
only **83.7%** of the time, so that is the ceiling, not 100%. And because the
prompt contract is frozen, there is no channel to request one convention at
inference: the training signal has to be made single-valued.

**The short-phrase failure is induced by the data.** Exact-copy rate is 33% at
1–2 source words versus 1.2% at 21+, so "echo short inputs" is a cheap rule that
fits the training distribution. Meanwhile the corpus contains **zero** instances
of धन्यवाद, नमस्कार or स्वागत across 21,622 rows.

## Layout

```
src/mt_marathi/
  bhili_data.py     corpus → training mixture; splits by paragraph, normalises
                    digit script, quarantines verbalised pairs, generates
                    numeral-substitution augmentation
  eval_bhili.py     evaluation with COPY baselines and per-failure diagnostics
  report.py         assembles the results table from metrics on the Hub
  curves.py         recovers training curves from a job log
  filters.py        structural quality filters for mined bitext
configs/            render + LoRA configs, each delta from the toolkit default annotated
jobs/               HF Jobs launcher (single A100)
tests/              pytest suite for the numeral pipeline
docs/               report, decision log, first-principles explainers
archive/en_mr/      superseded English↔Marathi track (see docs/DOCUMENTATION.md)
```

## Running it

```bash
uv venv .venv && . .venv/bin/activate && uv pip install -r requirements.txt
PYTHONPATH=src python -m mt_marathi.bhili_data     # build the mixture, local, free
python -m pytest tests/ -q
```

Training and evaluation run on a single A100 via HF Jobs; see
[jobs/train_and_eval.sh](jobs/train_and_eval.sh).

Artifacts — adapter, checkpoints, metrics, and **raw generations for every row** —
are at [`Arushhh/indic-translate-mar-bhili-lora`](https://huggingface.co/Arushhh/indic-translate-mar-bhili-lora).

## Notes on the toolkit

Four things that cost real GPU time to discover, documented in
[docs/DECISIONS.md](docs/DECISIONS.md):

- `install.sh` defaults to `--extras all`, which builds flash-attn from source —
  tens of minutes, for a library this model cannot use. `--extras all-mt` cuts
  install from ~15 minutes to ~2.
- `extra_languages` extends the **renderer** but not `IndicMTEngine`, which
  raises `ValueError: unsupported language 'Bhili'`. The engine's language table
  must be extended separately for the served prompt to match the trained one.
- `train_lora.sh` sets `WANDB_MODE=offline`; on ephemeral compute the run
  directory is destroyed on exit.
- `sacrebleu.CHRF()` defaults to `word_order=0` — that is chrF, not chrF++.
