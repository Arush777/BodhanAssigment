# Decision log

Chronological, with the reasoning and what it cost. Entries are appended as
work proceeds; nothing here is rewritten after the fact.

## 1. MT, not ASR or TTS
Single A100, ~30h. MT is text-only, so no audio download or manifest build, and
the official toolkit ships a working LoRA recipe. Debug loops are seconds
(read a sentence) rather than minutes (inspect a spectrogram).

## 2. LoRA, not full fine-tuning
`indic-translate` is 7.94B params. Full FT in bf16 with AdamW needs roughly
128 GB of weights + gradients + optimizer state. A single 80 GB A100 cannot
hold it. LoRA at r=32 trains ~1.3% of parameters and fits comfortably.

## 3. The thesis: adaptation budget, not "beat the baseline"
Marathi is one of the model's 22 supported languages and it already translates
it well (card reports chrF++ 50.2 en->mr, 62.3 mr->en on IN22-Gen). Training on
mined bitext with a high LoRA LR is a textbook catastrophic-forgetting setup, so
"raise the score" is a coin flip. The question we can actually answer is: **how
much adaptation can you buy from a strong instruction-tuned model, and what does
it cost elsewhere.**

## 4. Corrected: the chrF++ noise floor
The toolkit documents +/-0.06 chrF++. That is **decode-repeat** noise (same model
twice, differing because batching reorders float reductions). It is NOT test-set
sampling noise, which for 1024 sentences is nearer +/-0.7-1.0 by bootstrap.
Conflating them would have let us report a 0.3 drop as a finding when it sits
inside the confidence interval. We report bootstrap CIs and state which noise
we measured.

## 5. Structural filtering is a hygiene pass, not a quality filter
`filters.py` over Samanantar-mr keeps **95.2%** of 3.6M rows. That number is the
diagnosis: structural checks (length, ratio, script, digits, captions) cannot
see meaning. Found in a random sample of 12:

    eng: "Pakistan goalkeeper saved the shot from Akashdeep Singh."   [hockey]
    mar: "अफगाणिस्तानकडून फिरकीपटू राशिद खानने..."                      [cricket]

Well-formed both sides, correct scripts, ratio 1.5 -- passes every filter.
Semantic misalignment is the dominant failure mode in mined corpora.

## 6. Rejected: computing LaBSE similarity ourselves
Implemented in `semantic_filter.py`, then measured: **5.5 hours** for a 200k
subsample on the M2's MPS backend. That is most of the remaining wall-clock to
produce a strictly worse version of a published artifact. Used
`SPRINGLab/BPCC_cleaned` (LaBSE >= 0.9 over all of BPCC) instead. The local
implementation is kept because the measurement is the justification.

## 7. Corrected: BPCC_cleaned is NOT education-domain
Claimed this on the strength of one sample row that looked like lecture
transcript. At n=8 the Marathi slice is river-cleanup schemes, Manipur politics,
hotel employment statistics, Muharram -- the same PIB/news register as
Samanantar, only cleaner. The education corpus is `SPRINGLab/shiksha`
(NPTEL lecture transcripts, CC-BY-4.0, ungated, 115,952 en-mr pairs).

## 8. Test-set contamination, and a bug in the check for it
De-duplicate training data against IN22-Gen before training; Samanantar and BPCC
both overlap public benchmarks. The first implementation reported 0 overlaps.
A **positive control** -- feeding it genuine IN22 sentences with mangled case and
punctuation -- showed it caught 0/3. Cause: `normkey` stripped punctuation but
never collapsed the whitespace left behind, so "hello, world" became
"hello  world" and never matched. After the fix, 2/3 (the third is an artifact of
the test mangling "B.C.E." into "b c e"). A dedup that matches nothing looks
exactly like a corpus with no contamination.

## 9. Three corpora, two clean axes
| Arm | Corpus | Quality | Domain |
|---|---|---|---|
| A | Samanantar, structural only | noisy | general/news/govt |
| B | BPCC_cleaned, LaBSE >= 0.9 | clean | general/news/govt |
| C | shiksha, NPTEL lectures | clean-ish | education |

A vs B isolates **quality**. B vs C isolates **domain**. All 25k, matched.

## 10. Course-level held-out split for shiksha
Sentences inside one lecture are highly redundant. A random split lets
near-duplicates straddle train/dev and reports memorisation as generalisation.
Whole `course_id`s are held out instead. Constraint to state plainly: only
**31 distinct courses** exist in the en-mr slice, so topical coverage is narrow.

## 11. Stated limitation: "education" here means university STEM lecture
shiksha is spoken NPTEL transcript, not school textbook prose. No aligned
NCERT/Balbharati parallel corpus exists publicly in any Indian language pair.
Domain-*adjacent* to Bodhan's use case, not domain-matched.

## 12. Baseline reproduces the model card exactly
Measured on IN22-Gen, greedy, chrF++ word_order=2:

| direction | ours | card | n |
|---|---|---|---|
| en->mar_Deva | **50.15** | 50.2 | 1024 |
| mar_Deva->en | **62.39** | 62.3 | 1024 |

I had predicted this would NOT reproduce, on the grounds that the card never
states its tokenisation. That was wrong, and being wrong is the good outcome: an
independently built pipeline landing within 0.1 chrF++ of the published figure
validates the prompt contract, the IndicNLP pre-tokenisation, the chrF++
word_order and the greedy decode settings all at once.

Bootstrap 95% CI at n=1024 is **+/-0.77 chrF++** -- 13x the +/-0.06 the toolkit
documents, confirming the correction in entry 4. Differences under ~0.8 are noise.

Cross-lingual baseline (for the damage probe): en->hin 52.50, hin->en 61.88,
en->guj 51.50, guj->en 63.12, en->tam 48.87, tam->en 56.11.

Metric choice justified by our own data, not by citation: **en->tam chrF++ 48.87
but BLEU 8.74**. Tamil is agglutinative, so word-level BLEU collapses while
chrF++ stays informative. Same effect within Marathi: BLEU 14.35 vs Hindi 24.74
despite near-identical chrF++ (50.2 vs 52.5).

## 13. LoRA trainable parameters: 100,999,168 (1.2559%)
`trainable params: 100,999,168 || all params: 8,042,100,000 || trainable%: 1.2559`
-- exactly the figure in the toolkit docs, confirming target/exclude resolution
behaved as its authors intended.

My hand estimate from config.json was 77,758,464, i.e. **23.2M short**. The
architecture model behind that estimate is therefore wrong somewhere; to be
settled from `adapter_config.json` on the first saved checkpoint rather than
guessed at.

## 14. early_stopping_patience does not disable early stopping
Setting `early_stopping_patience: 999` still CONSTRUCTS an
`EarlyStoppingCallback`; large patience means it never fires, not that it is
absent. Having also dropped `metric_for_best_model` when setting
`load_best_model_at_end: false`, training died at `on_train_begin`:

    AssertionError: EarlyStoppingCallback requires metric_for_best_model to be defined

Fixed by restoring `metric_for_best_model: eval_loss` + `greater_is_better:
false`. Cost ~$0.96 -- but the baseline eval had already completed and pushed,
because artifacts are uploaded per-direction rather than batched at the end.
That ordering is why a crash cost the probe and not the experiment.
