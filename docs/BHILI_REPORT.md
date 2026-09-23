# Marathi ↔ Dehwali Bhili — findings and results

Task: fine-tune Indic-Translate for Marathi↔Bhili, addressing two reported
failures — numbers not translating, and short phrases translating badly.

---

## 1. The measurement that reframes the problem

Before training anything, I measured what the **unmodified** model does, and
what simply **copying the input** scores. Bhili is lexically close to Marathi
and shares its script, so copying is not a strawman — it is a strong policy.

| test set | direction | COPY | base model |
|---|---|---|---|
| GENERAL | mar→bhb | **42.1** | 40.0 |
| NUM_NAT | mar→bhb | **46.3** | 44.3 |
| SHORT_NAT | mar→bhb | **42.6** | 39.3 |
| SHORT_HARD | mar→bhb | **33.5** | 31.5 |
| GENERAL | bhb→mar | 40.5 | 40.9 |
| NUM_NAT | bhb→mar | 44.7 | 45.0 |
| SHORT_NAT | bhb→mar | 43.0 | 42.4 |
| SHORT_HARD | bhb→mar | 33.9 | 34.1 |

chrF++, word_order=2, n=400/355/72/60.

**Copying beats the base model on every Marathi→Bhili set. Bhili→Marathi is a
tie.** An identity function outperforms a model that has never seen Bhili.

That asymmetry is two different failures under one label:

- **Generating Bhili** — the model cannot produce a language it never saw, so
  copying wins by exploiting real lexical overlap.
- **Generating Marathi** — the model is excellent at Marathi, so it does not
  lose to copying; but it cannot *understand* Bhili, so it cannot beat it
  either. Fluency is useless without comprehension.

**Consequence for evaluation.** Numeral-preservation scores 100% for any
copying model, by construction. Measured on the base model it reads 100% on
every set — the model "handles numbers perfectly" precisely because it is not
translating. Any report of this metric without a COPY row is misleading.

---

## 2. Part of the number problem is in the data

Of source sentences containing numerals, **9.5% of references do not preserve
them** — but they are not errors. They are verbalised:

```
20 → वीस            800 → आठहोव           5 ते 8 → पाच ते आठ
```

That is a legitimate convention in Bhili prose. The consequence is that the
**human references themselves preserve source numerals only 83.7% of the time**
(85.6% general, 78.6% short, 70.0% shortest). That is the ceiling, not 100%.

The corpus therefore teaches three incompatible policies — keep digits (83.6%),
verbalise (9.5%), alter (6.9%) — with no feature distinguishing them. And since
the prompt contract is frozen (one user turn, target language only, no system
message), **there is no channel to request one convention at inference.** The
model must guess. This is an information problem, not a modelling one; the only
fix is to make the training signal single-valued.

A second axis: digit script. The Marathi side is ~7,800 rows Devanagari vs
~1,900 ASCII; the Bhili side ~6,000 vs ~2,900. The auxiliary Hindi–Bhili corpus
(AdiBhasha) is almost entirely ASCII — the opposite convention — so mixing
corpora without normalising first makes the inconsistency worse.

**Open question for the team: which convention should Bhili output use?**

---

## 3. Number *words* are not the problem

Mined from the corpus and cross-checked against 20k independent Hindi–Bhili
pairs:

```
चार→चार   आठ→आठ   छह→छह   सात→सात   बीस→बीस   दस→दह/दाह   हजार→ओजार
```

**Bhili cardinals are largely the same words as Marathi and Hindi**, with
regular sound shifts. The base model already knows these tokens. No lexicon is
needed — which removes a whole workstream.

---

## 4. The short-phrase failure is induced by the data

```
exact-copy rate   1-2 words  33.0%      11-20 words  1.9%
                  3   words  18.3%      21+   words  1.2%
```

LoRA learns the cheapest rule that fits: **echo short inputs**. It is right a
third of the time on this distribution and catastrophically wrong on the
phrases being judged.

And the register is simply absent — across 21,622 rows, occurrences of
**धन्यवाद: 0, नमस्कार: 0, स्वागत: 0**. Krishi Darshini is agricultural extension
material; nobody greets anyone in it.

---

## 5. What was done

| Intervention | Rationale |
|---|---|
| Split by `para_id` **before** augmenting | 10,197 paragraphs → 400 held out; stops synthesised rows leaking into test |
| Digit script normalised to the source's | Removes one ambiguity axis without pre-empting the convention question |
| 891 verbalised pairs quarantined | Makes the numeral signal single-valued |
| 461 trivial short copies dropped | Starves the copy prior |
| **6,000 numeral-substitution rows** | Values swapped on *both* sides → correct by construction; covers magnitudes the corpus never had. Verified: 0 mismatches |
| 2,538 conversational rows from AdiBhasha | Repopulates the missing register |

27,329 pairs → 52,994 rendered rows, bidirectional. LoRA r=32, 900 steps,
100,999,168 trainable parameters (1.26%), single A100.

---

## 6. Results

### chrF++ against the COPY baseline

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

**The fine-tune beats copying in both directions on every test set** — so it
learned to translate rather than to echo. The asymmetry persists but inverts in
the model's favour: Bhili→Marathi gains ~3× more, because the model already had
Marathi fluency and only needed Bhili comprehension.

### Spurious copy rate (the short-phrase complaint, directly)

| test set | direction | base | fine-tuned |
|---|---|---|---|
| GENERAL | mar→bhb | 41.9% | **8.5%** |
| GENERAL | bhb→mar | 48.8% | **2.9%** |
| SHORT_NAT | mar→bhb | 68.4% | **33.3%** |
| SHORT_NAT | bhb→mar | 57.9% | **8.8%** |
| SHORT_HARD | mar→bhb | 68.4% | **33.3%** |
| SHORT_HARD | bhb→mar | 57.9% | **8.8%** |

Of items whose reference differs from the source, how often the model just
echoed the source. Short Marathi→Bhili remains the hardest cell at 33.3% — real
improvement, not a solved problem.

### Numerals

| test set | direction | ref. ceiling | base script | FT script |
|---|---|---|---|---|
| GENERAL | mar→bhb | 85.6% | 67.4 | **75.1** |
| NUM_NAT | mar→bhb | 83.7% | 70.7 | **76.3** |
| SHORT_NAT | bhb→mar | 91.7% | 58.3 | **83.3** |
| SHORT_HARD | bhb→mar | 87.5% | 50.0 | **75.0** |

Numeral-sequence match stays at ~100% — nothing broke. The gain is in **digit
script**, which is where the signal actually lives once you account for copying.

---

## 7. Limitations

- **SHORT_HARD is 60 items** (±13pp at 95%). It shows direction, not a precise rate.
- **One training run, no ablations.** The contribution of numeral augmentation
  versus conversational data versus the copy-stripping is not separated.
- **Held-out data is from the same corpus**, so this measures in-domain quality.
  Generalisation to other Bhili varieties is untested — the data is specifically
  *Dehwali* Bhili.
- **The digit-convention question is unresolved**, so training was normalised to
  "preserve the source's script" and the evaluation was made script-insensitive
  on the primary metric. If the intended convention differs, it is a
  post-processing change, not a retrain.
- **0.54 epochs.** More training may help; it was not explored.

---

## Artifacts

- Adapter, checkpoints, metrics, raw generations: `Arushhh/indic-translate-mar-bhili-lora`
- Code, configs, pipeline: `github.com/Arush777/BodhanAssigment`

Raw generations for every row are published, so any follow-up question can be
answered without re-running a job.
