# Marathi ↔ Dehwali Bhili translation

LoRA fine-tune of [`bodhan-ai/indic-translate`](https://huggingface.co/bodhan-ai/indic-translate)
for Marathi ↔ Dehwali Bhili, aimed at two reported failures: numbers that do not
translate, and short phrases that translate badly.

Bhili is not one of the base model's 22 supported languages. This is not
adapting a language the model already knows, it is teaching it one it has never
seen.

## Result

The fine-tune beats a COPY baseline (echo the source back unchanged) in both
directions on all four evaluation sets. COPY is the comparison that matters
here, because the unmodified model *loses* to it on every Marathi→Bhili set.
Bhili is lexically close enough to Marathi that doing nothing outscores a model
with no exposure to the target language.

| test set | direction | COPY | base | fine-tuned | FT over COPY |
|---|---|---|---|---|---|
| GENERAL | mar→bhb | 42.1 | 40.0 | **49.0** | **+6.9** |
| GENERAL | bhb→mar | 40.5 | 40.9 | **60.3** | **+19.8** |
| NUM_NAT | mar→bhb | 46.3 | 44.3 | **52.0** | **+5.7** |
| NUM_NAT | bhb→mar | 44.7 | 45.0 | **65.3** | **+20.6** |
| SHORT_NAT | mar→bhb | 42.6 | 39.3 | **47.1** | **+4.5** |
| SHORT_NAT | bhb→mar | 43.0 | 42.4 | **60.1** | **+17.1** |
| SHORT_HARD | mar→bhb | 33.5 | 31.5 | **40.5** | **+7.0** |
| SHORT_HARD | bhb→mar | 33.9 | 34.1 | **53.8** | **+19.9** |

chrF++ with `word_order=2`. The into-Marathi direction gains far more than the
into-Bhili direction, which is what you would expect when the model already
knows one side of the pair.

Spurious copy rate measures the failure directly: of items whose reference
differs from the source, how often does the model just echo the source back?

| test set | direction | base | fine-tuned |
|---|---|---|---|
| GENERAL | mar→bhb | 41.9% | 8.5% |
| GENERAL | bhb→mar | 48.8% | 2.9% |
| SHORT_NAT | mar→bhb | 68.4% | 33.3% |
| SHORT_NAT | bhb→mar | 57.9% | 8.8% |

Full analysis: [docs/BHILI_REPORT.md](docs/BHILI_REPORT.md).

## Two findings that reframe the problem

Part of the number failure lives in the data, not the model. Across the corpus,
83.6% of source numerals are preserved verbatim in the reference, 9.5% are
verbalised (20 → वीस, 800 → आठहोव) and 6.9% are altered outright. Verbalisation
is a legitimate Bhili prose convention, not an annotation error, which puts the
reference preservation ceiling at 83.7% rather than 100%. The prompt contract is
frozen, so there is no channel for asking the model which convention you want at
inference time. The training signal has to be made single-valued instead.

Number *words* were never broken. Bhili cardinals mostly match Marathi and
Hindi (चार → चार, आठ → आठ, बीस → बीस, दस → दह), so the model has little to
learn there.

The short-phrase failure is induced by the data too. Exact-copy rate falls
steeply with source length: 33.0% at 1-2 words, 18.3% at 3, 7.1% at 4-5, 1.9%
at 11-20, 1.2% at 21 and above. "Echo anything short" is a cheap rule that fits
the training distribution well. The corpus is also agricultural end to end, and
contains zero instances of धन्यवाद, नमस्कार or स्वागत across all 21,622 rows,
so the polite formulae people test first are exactly the ones nothing in the
data teaches.

## Layout

```
configs/            render + LoRA configs, every delta from the toolkit default
                    annotated in place
data/bhili_eval/    the four frozen evaluation sets
docs/               findings report, first-principles explainers
jobs/               HF Jobs launcher (single A100)
results/            results table, training curves
scripts/            pulls metrics and raw generations back down from the Hub
src/mt_marathi/
  bhili_data.py     corpus to training mixture: paragraph-level splits, digit
                    script normalisation, quarantine for verbalised pairs,
                    numeral-substitution augmentation
  eval_bhili.py     evaluation with COPY baselines and per-failure diagnostics
  report.py         assembles the results table from metrics on the Hub
  curves.py         recovers training curves from a job log
tests/              pytest suite for the numeral pipeline
```

## Running it

```bash
uv venv .venv && . .venv/bin/activate && uv pip install -r requirements.txt
PYTHONPATH=src python -m mt_marathi.bhili_data     # build the mixture, local, free
python -m pytest tests/ -q
```

Data prep runs on a laptop CPU. Training and evaluation go to a single A100 via
HF Jobs: see [jobs/train_and_eval.sh](jobs/train_and_eval.sh). The run is 900
steps over 52,994 rendered rows, about 52 minutes wall clock, final eval_loss
1.101.

Adapter, checkpoints, metrics and the raw generation for every evaluated row are
at [`Arushhh/indic-translate-mar-bhili-lora`](https://huggingface.co/Arushhh/indic-translate-mar-bhili-lora).

## Notes on the toolkit

Four things that cost real GPU time to find out.

- `install.sh` defaults to `--extras all`, which builds flash-attn from source.
  That is tens of minutes for a library this model cannot use. `--extras all-mt`
  cuts the install from roughly 15 minutes to 2.
- `extra_languages` extends the renderer but not `IndicMTEngine`, which raises
  `ValueError: unsupported language 'Bhili'`. The engine's language table has to
  be extended separately or the served prompt will not match the trained one.
- `train_lora.sh` sets `WANDB_MODE=offline`, and on ephemeral compute the run
  directory goes away when the job exits. Sync the metrics yourself.
- `sacrebleu.CHRF()` defaults to `word_order=0`. That is chrF, not chrF++.

Repo: [github.com/Arush777/BodhanAssigment](https://github.com/Arush777/BodhanAssigment)
