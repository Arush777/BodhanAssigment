# Explainers

The decisions in this project rest on a handful of things about this model and
this toolkit that are not obvious. This file explains them from the ground up,
so a reader who does not already know the concept can follow the reasoning
rather than take the conclusion on trust.

---

## 1. What LoRA attaches to, and what `exclude_modules` is for

### A model is a tree of named parts

A neural network is not one blob. It is a nested structure of components, and
every component has a name built from its position in that tree, exactly like a
file path:

```
model
├── language_model
│   └── layers
│       ├── 0
│       │   ├── self_attn
│       │   │   ├── q_proj    ← model.language_model.layers.0.self_attn.q_proj
│       │   │   └── o_proj
│       │   └── mlp
│       │       └── gate_proj
│       └── ... 41 more layers
├── vision_tower              ← the image half
│   └── encoder.layers.0.self_attn.q_proj
└── audio_tower               ← the speech half
    └── layers.0.self_attn.relative_k_proj
```

Each leaf is a **module**. The ones ending in `_proj` are matrix
multiplications — the things LoRA can attach to.

This checkpoint holds about 2,000 such tensors and **only a third belong to the
language model.** The rest are vision and audio, because Indic-Translate is
built on a multimodal base that can also see and hear.

### LoRA has to choose which ones to attach to

You say so with `target_modules`. Two styles:

- **By name** — `["q_proj", "v_proj"]`. "Attach to anything called q_proj or
  v_proj." This is the common tutorial recipe.
- **`"all-linear"`** — "attach to every matrix-multiply layer you find."

On this model the second is correct and the first is quietly wrong. See §2.

### That is what exclusion patterns are for

`"all-linear"` is too greedy here: it would sweep in the vision and audio
towers, which a text-only task never touches. `exclude_modules` is a list of
patterns meaning *skip anything whose name matches this*:

```
*vision_tower*   *audio_tower*   *visual*   *image_encoder*
*embed_vision*   *embed_audio*   *lm_head*
```

The `*` is a wildcard, the same idea as `*.jpg` matching any filename ending in
`.jpg`. So `*audio_tower*` means "any module path containing the text
`audio_tower` anywhere in it."

The intended logic is therefore: **take every linear layer, then discard the
ones whose path mentions vision or audio.** What survives is the language model.

### Why we cannot verify from the saved config that this worked

After training, PEFT records what it attached to in `adapter_config.json`. But
it does not store **full paths** — it stores the shortest suffix that is unique.
So instead of

```
model.audio_tower.layers.3.self_attn.relative_k_proj
```

the file contains only

```
relative_k_proj
```

The part of the name that would identify it as audio — `audio_tower` — is
stripped by the saving process, and that is exactly the part an exclusion
pattern needs to match against.

Our resolved adapter contains about six such entries: `relative_k_proj`,
`input_proj`, `output_proj`, `embedding_projection`, `input_proj_linear`,
`linear`. `relative_k_proj` is **Conformer** vocabulary, and Conformers are
speech encoders, so these look like audio modules. But a naming resemblance is
not confirmation, and we do not claim one.

### Why it cannot affect our results either way

LoRA initialises its two matrices asymmetrically: one random, the other **all
zeros**. At step zero the adapter contributes exactly nothing and the model
behaves identically to the base. Those weights move away from zero only where
gradient flows.

On a text-only translation task the audio tower is never invoked. No audio input
means no forward pass through it, which means no gradient, which means those
zero-initialised weights **stay at zero for the whole run.** They contribute
nothing during training and nothing at inference.

Worst case: a few megabytes of wasted optimiser state. It cannot change a single
output token, and since all three arms carry the identical configuration it
cannot bias the comparison even in principle.

---

## 2. Why `all-linear` and not a hand-written target list

Counting how often each projection appears in our resolved adapter:

| projection | count |
|---|---|
| q_proj, o_proj, gate_proj, up_proj, down_proj | **42** each |
| k_proj, v_proj | **24** each |

The model has 42 layers, so why do keys and values appear on only 24?

The answer is in the model's own config: `num_kv_shared_layers: 18`, and
42 − 18 = 24. Normally every layer computes its own keys and values from its
input. This model does not. **Layers 24 through 41 reuse the keys and values
computed at layer 23** rather than computing their own — a memory and speed
optimisation that also shrinks the KV cache at inference. Those 18 layers
therefore have no key or value weight matrices at all. There is physically
nothing for LoRA to attach to.

**The consequence.** The recipe in almost every LoRA tutorial is
`target_modules=["q_proj", "v_proj"]`. On this model that attaches to query on
all 42 layers but value on only 24 — and PEFT issues **no warning**. You would
quietly be adapting 18 layers with query alone and never know. `"all-linear"`
sidesteps this because it adapts whatever actually exists.

Verified trainable total: **100,999,168 of 8,042,100,000 — 1.2559%.**

---

## 3. Two unrelated things both called "checkpointing"

The names collide and the distinction matters.

**Gradient checkpointing** is a memory technique inside the backward pass.
Normally, computing gradients requires keeping every layer's intermediate
activations in VRAM. Gradient checkpointing throws most of them away and
**recomputes** them during backpropagation instead. It trades roughly 40% more
compute for a large memory saving. It writes nothing to disk, ever. It is a
decision made and discarded inside a single training step.

**Model checkpointing** (`save_steps`) is writing the adapter weights out so you
can resume, evaluate, or roll back. This is the only thing that governs
resumability.

They are independent. Turning gradient checkpointing off does not affect saving,
resuming, or inference — inference has no backward pass, so the setting is not
even read.

**Does gradient checkpointing change results?** Mathematically no: it recomputes
the same activations it would otherwise have stored. We measured this directly,
same seed and data, checkpointing on versus off:

| step | on | off |
|---|---|---|
| 10 | 1.886 | 1.886 |
| 30 | 1.690 | 1.690 |
| 70 | 1.733 | 1.732 |

Identical to the third decimal; the residual is floating-point non-determinism
from kernel selection during recomputation.

**We kept it on anyway.** Peak VRAM without it reaches ~70 GB of 80 at sequence
length 1024, and six rows of our 75,000 exceed 512 tokens — a small but real OOM
risk. On a fixed deadline with rented compute, certainty beat the speedup.

---

## 4. Why chrF++ rather than BLEU, argued from our own numbers

**BLEU** counts matching word n-grams between the model's output and the
reference. **chrF++** counts matching *character* n-grams (plus word bigrams).

For a morphologically rich language that distinction is decisive. Marathi
inflects heavily, so a translation can be correct and still use a different
surface form of the right word. BLEU sees a different word and scores zero for
that n-gram; chrF++ sees that most of the characters match and scores
accordingly.

Our baseline shows this starkly:

| direction | chrF++ | BLEU |
|---|---|---|
| en→Tamil | 48.87 | **8.74** |
| en→Hindi | 52.50 | 24.74 |
| en→Marathi | 50.15 | 14.35 |

Tamil is agglutinative — words are built by stacking morphemes — so word-level
BLEU collapses to 8.74 while character-level chrF++ stays at a sensible 48.87.
The same effect appears within our own target: Marathi's BLEU is 14.35 against
Hindi's 24.74 despite nearly identical chrF++, because Marathi is
morphologically richer than Hindi.

This is why IndicTrans2 selects chrF++ as its primary metric. We now have our
own evidence for it rather than a citation.

**One trap.** `sacrebleu.CHRF()` defaults to `word_order=0`, which is plain
chrF, **not** chrF++. You must pass `word_order=2` explicitly. And BLEU on a
Devanagari target is meaningless unless both prediction and reference are
IndicNLP-tokenised first — sacrebleu's default tokenizer is built for Latin
script.

---

## 5. Why three corpora rather than one

If you fine-tune on one dataset and the score moves, you cannot say why. Three
explanations fit any negative result equally well: the data was noisy, the data
was the wrong domain, or fine-tuning always damages this model.

One experiment cannot separate them, so we change one variable at a time:

| arm | corpus | quality | domain |
|---|---|---|---|
| A | Samanantar, structural filters only | noisy | general / news / govt |
| B | BPCC_cleaned, LaBSE ≥ 0.9 | **clean** | general / news / govt |
| C | shiksha, NPTEL lectures | clean | **education** |

**A → B** holds domain constant and varies quality, so any difference is a
quality effect. **B → C** holds quality constant and varies domain, so any
difference is a domain effect.

B is not "another dataset" — B is A's control. C is not "more data" — C is B's
control. Without B, comparing A to C would confound the two and tell you
nothing.

---

## 6. Two different noise floors, and why confusing them ruins a conclusion

The toolkit documents a chrF++ noise floor of **±0.06**. That is
**decode-repeat noise**: run the same model on the same test set twice and the
scores differ slightly, because batching reorders floating-point reductions.

That is not the number you need when comparing *two different models*. For that
you need **test-set sampling noise** — how much the score would move if you had
drawn a different 1,024 sentences. We measured it by bootstrap: **±0.77 chrF++**
at n=1024.

**Thirteen times larger.** Quoting a 0.3 chrF++ drop as a finding "well above
the ±0.06 noise floor" would be wrong: 0.3 sits comfortably inside the interval
that matters. Anything under roughly ±0.8 on this benchmark is not a result.
