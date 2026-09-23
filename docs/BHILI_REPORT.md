# Marathi to Dehwali Bhili

Fine-tuning Indic-Translate for Marathi/Bhili in both directions, with two
reported failures to address: numbers not translating, and short phrases
translating badly.

## 1. What the baseline actually is

Before training anything I measured the unmodified model, and also what simply
copying the input scores. Bhili shares Devanagari with Marathi and overlaps
heavily in vocabulary, so copying is not a strawman.

| test set | direction | COPY | base model |
|---|---|---|---|
| GENERAL | mar-bhb | 42.1 | 40.0 |
| NUM_NAT | mar-bhb | 46.3 | 44.3 |
| SHORT_NAT | mar-bhb | 42.6 | 39.3 |
| SHORT_HARD | mar-bhb | 33.5 | 31.5 |
| GENERAL | bhb-mar | 40.5 | 40.9 |
| NUM_NAT | bhb-mar | 44.7 | 45.0 |
| SHORT_NAT | bhb-mar | 43.0 | 42.4 |
| SHORT_HARD | bhb-mar | 33.9 | 34.1 |

chrF++ with word_order=2, n = 400 / 355 / 72 / 60.

Copying beats the base model on every Marathi to Bhili set. The reverse
direction is a tie. An identity function outperforms a model that has never
seen Bhili.

The asymmetry is two different failures sharing a label. Going into Bhili, the
model cannot generate a language it never saw, so copying wins by exploiting
real lexical overlap. Going into Marathi, the model is already fluent, so it
does not lose to copying; it cannot beat it either, because it does not
understand the Bhili input. Fluency without comprehension buys nothing.

This also rules out the obvious evaluation. Numeral preservation scores 100%
for any copying model by construction, and measured on the base model it reads
100% on every set. The model "handles numbers perfectly" precisely because it
is not translating.

## 2. Numbers: the metric was wrong, and the problem does not reproduce

The requirement is that a number comes across correctly in either numeral or
word form. Digit-sequence matching does not measure that. It scores 20 -> वीस as
a dropped numeral when the translation is correct.

Re-scoring the saved generations with a metric that extracts number values
rather than surface forms, so digits and cardinals collapse to the same thing:

| test set | direction | reference ceiling | base | fine-tuned | n |
|---|---|---|---|---|---|
| GENERAL | mar-bhb | 90.6% | 90.1% | 90.1% | 181 |
| GENERAL | bhb-mar | 90.0% | 90.6% | 89.4% | 170 |
| NUM_NAT | mar-bhb | 88.7% | 88.2% | 88.2% | 355 |
| NUM_NAT | bhb-mar | 88.8% | 89.7% | 90.0% | 331 |
| SHORT_NAT | mar-bhb | 85.7% | 85.7% | 78.6% | 14 |
| SHORT_HARD | mar-bhb | 80.0% | 80.0% | 70.0% | 10 |

Both models sit at the ceiling. The reference itself carries the source's number
values 88.7% of the time on the main set, and both models match that within a
point. On this test set there is no measurable number problem under the stated
criterion.

The two small cells where the fine-tune looks worse have n = 14 and n = 10, so
they are noise. They are listed rather than hidden.

Caveats. The cardinal lexicon covers Marathi, Hindi and the Bhili variants
attested in this corpus, but not every compound, so some word-form numbers are
missed. Building the metric surfaced four bugs of my own, worth recording
because each one moved the number: cardinals matched as substrings (तीस inside
चोवतीस), missing Bhili variants (पास for 5), missing compounds (चोवतीस for 34),
and list enumerators counted as content. Fixing them moved the reference ceiling
from 63.9% to 88.7%.

## 3. The digit convention question, resolved



Of source sentences containing numerals, 9.5% of references do not preserve
them. They are not errors. They are verbalised:

```
20 -> वीस          800 -> आठहोव          5 ते 8 -> पाच ते आठ
```

That is a normal convention in Bhili prose. The consequence is that the human
references themselves preserve source numerals only 83.7% of the time, falling
to 70.0% on the shortest sentences. That figure is the ceiling, not 100%.

So the corpus teaches three incompatible policies: keep the digits (83.6%),
verbalise them (9.5%), change them (6.9%). No feature distinguishes the cases.
The served prompt contract is frozen at one user turn naming only the target
language, with no system message, so there is no channel to request a
convention at inference. The model has to guess. This is an information
problem rather than a modelling one, and the only fix available is to make the
training signal single-valued.

There is a second axis. The Marathi side of the corpus runs roughly 7,800 rows
in Devanagari numerals against 1,900 in ASCII; the Bhili side is 6,000 against
2,900. The auxiliary Hindi/Bhili corpus is almost entirely ASCII, the opposite
convention, so mixing sources without normalising first makes the
inconsistency worse.

Since either form counts as correct, this ambiguity is not a problem to solve.
The model may pick either. The earlier argument that a frozen prompt contract
makes this unfixable was answering a question nobody asked.

## 4. Number words are not the problem

Mined from this corpus and cross-checked against 20k independent Hindi/Bhili
pairs:

```
चार -> चार    आठ -> आठ    छह -> छह    सात -> सात    बीस -> बीस
दस -> दह/दाह              हजार -> ओजार
```

Bhili cardinals are largely the same words as Marathi and Hindi, with regular
sound shifts. The base model already knows these tokens, so no lexicon was
needed. That removed a workstream.

## 5. The short-phrase failure is induced by the data

```
exact-copy rate    1-2 words  33.0%       11-20 words  1.9%
                   3   words  18.3%       21+   words  1.2%
```

LoRA learns the cheapest rule that fits, and "echo short inputs" is right a
third of the time on this distribution. It is wrong on exactly the phrases
being judged.

The register is also absent. Across 21,622 rows, धन्यवाद appears 0 times,
नमस्कार 0, स्वागत 0. Krishi Darshini is agricultural extension material. Nobody
greets anyone in it.

## 6. What was done

| Step | Reason |
|---|---|
| Split by para_id before augmenting | 10,197 paragraphs, 400 held out; stops synthesised rows leaking into test |
| Digit script normalised to the source's | Removes one ambiguity axis without pre-empting the convention question |
| 891 verbalised pairs quarantined | Makes the numeral signal single-valued |
| 461 trivial short copies dropped | Starves the copy prior |
| 6,000 numeral-substitution rows | Values swapped on both sides at once, so correct by construction; covers magnitudes the corpus never had. Verified at 0 mismatches |
| 2,538 conversational rows from AdiBhasha | Repopulates the missing register |

27,329 pairs became 52,994 rendered rows, bidirectional. LoRA r=32, all-linear
targets, 100,999,168 trainable parameters (1.26%), 900 steps, one A100, 52
minutes.

## 7. Results

chrF++ against the COPY baseline:

| test set | direction | COPY | base | fine-tuned | FT - COPY |
|---|---|---|---|---|---|
| GENERAL | mar-bhb | 42.1 | 40.0 | 49.0 | +6.9 |
| GENERAL | bhb-mar | 40.5 | 40.9 | 60.3 | +19.8 |
| NUM_NAT | mar-bhb | 46.3 | 44.3 | 52.0 | +5.7 |
| NUM_NAT | bhb-mar | 44.7 | 45.0 | 65.3 | +20.6 |
| SHORT_NAT | mar-bhb | 42.6 | 39.3 | 47.1 | +4.5 |
| SHORT_NAT | bhb-mar | 43.0 | 42.4 | 60.1 | +17.1 |
| SHORT_HARD | mar-bhb | 33.5 | 31.5 | 40.5 | +7.0 |
| SHORT_HARD | bhb-mar | 33.9 | 34.1 | 53.8 | +19.9 |

The fine-tune beats copying in both directions on every set, so it learned to
translate rather than to echo. The asymmetry persists but inverts in the
model's favour: Bhili to Marathi gains about three times more, because the
model already had the Marathi and only needed the comprehension.

Spurious copy rate, meaning the fraction of items whose reference differs from
the source where the model still echoed the source:

| test set | direction | base | fine-tuned |
|---|---|---|---|
| GENERAL | mar-bhb | 41.9% | 8.5% |
| GENERAL | bhb-mar | 48.8% | 2.9% |
| SHORT_NAT | mar-bhb | 68.4% | 33.3% |
| SHORT_NAT | bhb-mar | 57.9% | 8.8% |
| SHORT_HARD | mar-bhb | 68.4% | 33.3% |
| SHORT_HARD | bhb-mar | 57.9% | 8.8% |

This is the metric that names the short-phrase complaint directly. Short
Marathi to Bhili is still the hardest cell at 33.3%, so this is real
improvement and not a solved problem.

Numerals:

| test set | direction | ref. ceiling | base script | FT script |
|---|---|---|---|---|
| GENERAL | mar-bhb | 85.6% | 67.4 | 75.1 |
| NUM_NAT | mar-bhb | 83.7% | 70.7 | 76.3 |
| SHORT_NAT | bhb-mar | 91.7% | 58.3 | 83.3 |
| SHORT_HARD | bhb-mar | 87.5% | 50.0 | 75.0 |

Numeral-sequence match stays near 100%, so nothing broke. The gain is in digit
script, which is where the signal lives once copying is accounted for.

## 8. Limitations

SHORT_HARD is 60 items, giving roughly ±13pp at 95%. It shows direction, not a
precise rate.

One training run, no ablations. The separate contributions of numeral
augmentation, conversational data and copy-stripping are not isolated.

Held-out data comes from the same corpus, so this measures in-domain quality.
Generalisation to other Bhili varieties is untested, and the data is
specifically Dehwali.

The digit-convention question is unresolved, so training normalised to
"preserve the source's script" and the primary metric was made
script-insensitive. If the intended convention differs, that is a
post-processing change rather than a retrain.

Trained for 0.54 epochs. More may help; it was not explored.

## Artifacts

Adapter, checkpoints, metrics and raw generations for every row:
`Arushhh/indic-translate-mar-bhili-lora`. Code and configs:
`github.com/Arush777/BodhanAssigment`.

Raw generations are published so follow-up questions can be answered without
re-running a job.
