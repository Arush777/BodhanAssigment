# Explainers

A few things about this model and this toolkit are not obvious from the docs,
and several decisions here only make sense once you know them. Each section
builds one from the ground up.

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

Each leaf is a module. The ones ending in `_proj` are matrix multiplications,
which is what LoRA can attach to.

The base here is Gemma 4 E4B, 7.94B parameters in total. Its language model is
decoder-only: 42 layers, hidden size 2560, a 262,144-token vocabulary. Of the
roughly two thousand tensors in the checkpoint only about a third belong to the
language model. The rest are the vision and audio towers, because the base can
also see and hear.

### LoRA has to choose which ones to attach to

`target_modules` says which. Two styles:

- By name, `["q_proj", "v_proj"]`: attach to anything called q_proj or v_proj.
  The usual tutorial recipe.
- `"all-linear"`: attach to every matrix-multiply layer you find.

On this model the second is correct and the first is quietly wrong. See §2.

### That is what exclusion patterns are for

`"all-linear"` is too greedy on a multimodal base. It sweeps in the vision and
audio towers, which a text-only task never touches. `exclude_modules` is a list
of patterns meaning *skip anything whose name matches this*:

```
*vision_tower*   *audio_tower*   *visual*   *image_encoder*
*embed_vision*   *embed_audio*   *lm_head*
```

`*` is a wildcard, the same idea as `*.jpg` matching any filename ending in
`.jpg`, so `*audio_tower*` means "any module path containing `audio_tower`
anywhere in it". Take every linear layer, discard the ones whose path mentions
vision or audio, and what survives is the language model.

### Why the saved config cannot confirm this worked

PEFT records what it attached to in `adapter_config.json`, but it does not store
full paths. It stores the shortest suffix that is unique. So instead of

```
model.audio_tower.layers.3.self_attn.relative_k_proj
```

the file contains only

```
relative_k_proj
```

The part of the name that would identify it as audio is stripped by the saving
process, and that is exactly the part an exclusion pattern needs to match
against.

Our resolved adapter contains about six such entries: `relative_k_proj`,
`input_proj`, `output_proj`, `embedding_projection`, `input_proj_linear`,
`linear`. `relative_k_proj` is Conformer vocabulary and Conformers are speech
encoders, so these look like audio modules. A naming resemblance is not
confirmation, and we do not claim one.

### Why it cannot affect the results either way

LoRA initialises its two matrices asymmetrically: one random, the other all
zeros. At step zero the adapter contributes exactly nothing and the model
behaves identically to the base. Those weights move away from zero only where
gradient flows.

A text-only translation task never invokes the audio tower. No audio input means
no forward pass through it, which means no gradient, which means those
zero-initialised weights sit at zero for the entire run, contributing nothing
during training and nothing at inference. Worst case is a few megabytes of
wasted optimiser state. It cannot change a single output token.

---

## 2. Why `all-linear` and not a hand-written target list

Count how often each projection appears in the resolved adapter:

| projection | count |
|---|---|
| q_proj, o_proj, gate_proj, up_proj, down_proj | **42** each |
| k_proj, v_proj | **24** each |

The model has 42 layers. So why do keys and values appear on only 24?

The config answers it: `num_kv_shared_layers: 18`, and 42 minus 18 is 24.
Normally every layer computes its own keys and values from its input. This model
does not. Layers 24 through 41 reuse the keys and values computed at layer 23,
a memory and speed optimisation that also shrinks the KV cache at inference.
Those 18 layers have no key or value weight matrices at all. There is physically
nothing for LoRA to attach to.

The consequence is a silent failure. `target_modules=["q_proj", "v_proj"]`
attaches to query on all 42 layers and value on only 24, and PEFT issues no
warning about it. You would be adapting 18 layers with query alone and never
find out. `"all-linear"` sidesteps the whole problem by adapting whatever
actually exists.

Trainable total, as reported by the run: 100,999,168 parameters, 1.2559% of the
checkpoint, with LoRA rank 32.

---

## 3. Two unrelated things both called "checkpointing"

The names collide and the distinction matters.

Gradient checkpointing is a memory technique inside the backward pass. Computing
gradients normally means keeping every layer's intermediate activations in VRAM.
Gradient checkpointing throws most of them away and recomputes them during
backpropagation instead, trading roughly 40% more compute for a large memory
saving. It writes nothing to disk, ever: it is a decision made and discarded
inside a single training step.

Model checkpointing (`save_steps`) is writing the adapter weights out so you can
resume, evaluate or roll back. It is the only one of the two that governs
resumability, and turning gradient checkpointing off does not affect it.
Inference has no backward pass, so there the flag is not even read.

It does not change results either. Recomputed activations are the same
activations, so paired runs at the same seed track each other to the third
decimal of the loss, and the residual is floating-point non-determinism from
kernel selection during recomputation.

We kept it on. The full run was 52 minutes on one A100, so the compute penalty
was cheap, while at sequence length 1024 the memory headroom without it is thin
enough that one long row could take the job down. Against a deadline on rented
compute, certainty beat the speedup.

---

## 4. Why chrF++ rather than BLEU

BLEU counts matching word n-grams between the model's output and the reference.
chrF++ counts matching *character* n-grams, plus word bigrams.

That difference decides whether the metric can see anything at all here. Marathi
inflects heavily, and so does Bhili: case, number and gender ride on suffixes,
so a translation can be correct and still land on a different surface form of
the right stem. BLEU compares whole tokens, and a different suffix is a
different token worth zero. chrF++ sees that most of the characters line up and
scores the near miss as a near miss.

Take a cardinal straight out of the corpus, Marathi दस against Bhili दह. One
character of two differs. BLEU treats them as unrelated words and awards
nothing; chrF++ credits the character they share. Multiply that across every
inflected form in 21,622 pairs of agricultural prose and word-level scoring is
mostly measuring noise.

There is a second reason, specific to a pair this close. Marathi and Bhili share
a script and a great deal of vocabulary, so a model that copies its input
verbatim already scores 42.1 chrF++ on general Marathi→Bhili text. Character
overlap is high before any translation happens. That is not an argument against
chrF++, it is an argument for never publishing a chrF++ number without the COPY
baseline beside it. Our 47.1 on SHORT_NAT looks better than our 40.5 on
SHORT_HARD, but the first is +4.5 over copying and the second is +7.0. The
absolute numbers rank them the wrong way round.

One trap. `sacrebleu.CHRF()` defaults to `word_order=0`, which is plain chrF and
not chrF++. Pass `word_order=2` explicitly. And BLEU on a Devanagari target is
meaningless unless prediction and reference are both IndicNLP-tokenised first,
since sacrebleu's default tokenizer is built for Latin script.

---

## 5. Two different noise floors, and why confusing them ruins a conclusion

The toolkit documents a chrF++ noise floor of ±0.06. That figure is
decode-repeat noise: run the same model on the same test set twice and the
scores differ slightly, because batching reorders floating-point reductions.

It is not the number you want when comparing two *different* models. For that
the relevant quantity is test-set sampling noise, meaning how far the score
would move if a different sample of sentences had landed in the evaluation set.
Bootstrapping the evaluation sets puts that an order of magnitude above the
decode-repeat floor.

So quoting a few tenths of a chrF++ point as a finding "well above the ±0.06
noise floor" is wrong. Tenths sit comfortably inside the interval that actually
governs a model-to-model comparison. Our margins over COPY run from +4.5 to
+20.6, which is the only reason we are willing to call them results.
