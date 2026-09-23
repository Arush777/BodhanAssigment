"""Build the Marathi<->Bhili training mixture and diagnostic evaluation sets.

The corpus is AI Kosh's Dehwali Bhili Krishi Darshini: 21,622 validated
Marathi/Bhili sentence pairs from agricultural extension material. Two measured
properties of it drive everything here.

**Numerals carry two conflicting conventions.** 44.6% of source sentences
contain a digit. 83.6% of references preserve the source numerals, 9.5%
verbalise them (20 -> वीस, 800 -> आठहोव) and 6.9% alter them. Verbalisation is a
legitimate style in Bhili prose, not corruption -- but the served prompt
contract is frozen (one user turn, target language only, no system message), so
there is no channel through which to request one style or the other. With both
present in training the model must guess, and will be wrong on roughly a tenth
of numeral sentences regardless of how well it is trained. That is an
information problem, not a modelling one, so the only fix is to make the
training signal single-valued.

**A copy prior is induced by the short bucket.** Exact-copy rate is 33% at 1-2
source words, 25% for short non-numeric segments, and 1.2% at 21+ words. Bhili
is lexically close to Marathi, so "echo short inputs" is a cheap rule that is
right a third of the time on this distribution -- and catastrophically wrong on
the greetings and courtesy phrases the corpus never contains (धन्यवाद, नमस्कार,
स्वागत: zero occurrences each).

Mitigations applied here, in order:
  1. split by para_id BEFORE any augmentation, so synthesised rows cannot leak
  2. normalise digit script to the source's, removing one axis of ambiguity
  3. quarantine verbalised pairs out of the main mixture
  4. numeral-substitution augmentation -- form-preserving, so guaranteed correct
  5. drop trivial short copies and repopulate from conversational auxiliary data
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

DEVA_DIGITS = "०१२३४५६७८९"
TO_ASCII = str.maketrans(DEVA_DIGITS, "0123456789")
TO_DEVA = str.maketrans("0123456789", DEVA_DIGITS)
NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def numerals(text: str) -> list[str]:
    """Ordered numerals, script-normalised, so comparisons ignore digit script."""
    return NUM_RE.findall(text.translate(TO_ASCII))


def source_digit_script(text: str) -> str:
    deva = sum(c in DEVA_DIGITS for c in text)
    ascii_ = sum(c.isdigit() and c not in DEVA_DIGITS for c in text)
    return "deva" if deva > ascii_ else "ascii"


def match_digit_script(text: str, script: str) -> str:
    """Rewrite digits in `text` into `script`.

    One of the two ambiguity axes is which script the target should use. The
    corpus disagrees with itself (Marathi side ~7.8k Devanagari vs ~1.9k ASCII;
    Bhili side ~6.0k vs ~2.9k), and the auxiliary Hindi-Bhili corpus is almost
    entirely ASCII -- the opposite convention. Mixing without normalising first
    makes the inconsistency worse, so we align the target to the source and
    leave the choice of global convention to a post-processing step.
    """
    return text.translate(TO_DEVA if script == "deva" else TO_ASCII)


def classify(src: str, tgt: str) -> str:
    """Which numeral policy does this pair demonstrate?"""
    s, t = numerals(src), numerals(tgt)
    if not s:
        return "no_numeral"
    if s == t:
        return "preserved"
    if not t:
        return "verbalised"
    return "altered"


def substitute_numerals(src: str, tgt: str, rng: random.Random) -> tuple[str, str] | None:
    """Replace each numeral with a sampled value on BOTH sides simultaneously.

    Only valid where the numeral sequences already agree: the substitution is
    then form-preserving, so the new pair is correct by construction. This is
    the one augmentation here that needs no external data and introduces no
    noise, and it covers magnitudes and formats the corpus never contained.
    """
    s_nums = numerals(src)
    if not s_nums or s_nums != numerals(tgt):
        return None

    script = source_digit_script(src)
    new_src, new_tgt = src, tgt
    for old in s_nums:
        if "." in old:
            new = f"{rng.uniform(0.1, 99.9):.1f}"
        elif len(old) <= 2:
            new = str(rng.randint(1, 99))
        elif len(old) == 3:
            new = str(rng.randint(100, 999))
        else:
            new = str(rng.randint(1000, 999999))
        old_src = old if script == "ascii" else old.translate(TO_DEVA)
        new_src_form = new if script == "ascii" else new.translate(TO_DEVA)
        # Replace one occurrence at a time so repeated values stay aligned.
        new_src = new_src.replace(old_src, new_src_form, 1)
        for variant in (old, old.translate(TO_DEVA)):
            if variant in new_tgt:
                new_tgt = new_tgt.replace(variant, new_src_form, 1)
                break
    if numerals(new_src) != numerals(new_tgt):
        return None
    return new_src, new_tgt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", type=Path,
                    default=Path("../tmp/bhili/Dehwali_Bhili_Krishi_Darshani_Pipeline.tsv"))
    ap.add_argument("--adibhasha", type=Path, default=Path("/tmp/adi.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/bhili"))
    ap.add_argument("--heldout-paras", type=int, default=400)
    ap.add_argument("--augment", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = list(csv.DictReader(args.tsv.open(encoding="utf-8"), delimiter="\t"))
    pairs = [(r["para_id"], r["sentence_marathi"].strip(),
              r["validated_translation_Dehwali_Bhili"].strip()) for r in rows]
    pairs = [p for p in pairs if p[1] and p[2]]

    # ---- split by paragraph, before augmentation, so nothing leaks ----------
    paras = sorted({p[0] for p in pairs})
    rng.shuffle(paras)
    held = set(paras[: args.heldout_paras])
    train = [p for p in pairs if p[0] not in held]
    test = [p for p in pairs if p[0] in held]
    print(f"  paragraphs {len(paras):,} -> held out {len(held):,}")
    print(f"  train {len(train):,} sentences | test {len(test):,} sentences")

    # ---- numeral policy census ---------------------------------------------
    from collections import Counter
    census = Counter(classify(m, b) for _, m, b in train)
    print("\n  numeral policy in TRAIN:")
    for k, v in census.most_common():
        print(f"    {k:<12}{v:>7,}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- main mixture -------------------------------------------------------
    main_rows, quarantine, dropped_copy = [], [], 0
    for _, m, b in train:
        kind = classify(m, b)
        if kind == "verbalised":
            quarantine.append({"mar": m, "bhb": b})
            continue
        if kind == "altered":
            continue                                   # alignment noise
        b = match_digit_script(b, source_digit_script(m))
        if m == b and len(m.split()) <= 3:
            dropped_copy += 1                          # kills the copy prior
            continue
        main_rows.append({"mar": m, "bhb": b})

    print(f"\n  main mixture        {len(main_rows):>7,}")
    print(f"  quarantined (verbalised) {quarantine.__len__():>2,}"
          f"   <- held back: ambiguous policy, reusable as number-word data")
    print(f"  dropped trivial short copies {dropped_copy:>5,}")

    # ---- numeral-substitution augmentation ---------------------------------
    eligible = [r for r in main_rows if numerals(r["mar"])]
    aug = []
    while len(aug) < args.augment and eligible:
        r = rng.choice(eligible)
        got = substitute_numerals(r["mar"], r["bhb"], rng)
        if got:
            aug.append({"mar": got[0], "bhb": got[1]})
    print(f"  numeral augmentation  {len(aug):>7,}  (form-preserving, correct by construction)")

    # ---- conversational repopulation from AdiBhasha ------------------------
    conv = []
    if args.adibhasha.exists():
        ad = list(csv.DictReader(args.adibhasha.open(encoding="utf-8")))
        short = [(r["Hindi"].strip(), r["Bhili"].strip()) for r in ad
                 if r["Hindi"].strip() and r["Bhili"].strip()
                 and len(r["Hindi"].split()) <= 8 and r["Hindi"].strip() != r["Bhili"].strip()]
        conv = [{"hin": h, "bhb": b} for h, b in short]
        print(f"  AdiBhasha short/conversational {len(conv):>5,}  (Hindi<->Bhili, register the main corpus lacks)")

    for name, data in (("train_main", main_rows), ("train_aug", aug),
                       ("quarantine_verbalised", quarantine),
                       ("aux_conversational", conv),
                       ("test_heldout", [{"mar": m, "bhb": b} for _, m, b in test])):
        path = args.out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in data:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  wrote {len(data):>7,} -> {path}")


if __name__ == "__main__":
    main()
