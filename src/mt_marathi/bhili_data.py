"""Build the Marathi/Bhili training mixture and the held-out evaluation sets.

Two properties of the corpus drive the choices here. Numerals carry two
conflicting conventions (83.6% preserved, 9.5% verbalised, 6.9% altered) and the
prompt contract has no channel to request one, so the signal has to be made
single-valued. And exact-copy rate runs 33% at one or two source words against
1.2% at twenty-plus, which is enough for LoRA to learn "echo short inputs".
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

DEVA_DIGITS = "०१२३४५६७८९"
TO_ASCII = str.maketrans(DEVA_DIGITS, "0123456789")
TO_DEVA = str.maketrans("0123456789", DEVA_DIGITS)
NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def numerals(text: str) -> list[str]:
    """Ordered numerals, script-normalised so comparisons ignore digit script."""
    return NUM_RE.findall(text.translate(TO_ASCII))


def source_digit_script(text: str) -> str:
    deva = sum(c in DEVA_DIGITS for c in text)
    ascii_ = sum(c.isdigit() and c not in DEVA_DIGITS for c in text)
    return "deva" if deva > ascii_ else "ascii"


def match_digit_script(text: str, script: str) -> str:
    return text.translate(TO_DEVA if script == "deva" else TO_ASCII)


def classify(src: str, tgt: str) -> str:
    """Which numeral policy a pair demonstrates."""
    s, t = numerals(src), numerals(tgt)
    if not s:
        return "no_numeral"
    if s == t:
        return "preserved"
    if not t:
        return "verbalised"
    return "altered"


def substitute_numerals(src: str, tgt: str, rng: random.Random) -> tuple[str, str] | None:
    """Swap every numeral for a sampled value on both sides at once.

    Only valid where the numeral sequences already agree; the substitution is
    then form-preserving and the new pair is correct by construction. Returns
    None rather than guessing when the sides disagree.
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
        old_form = old if script == "ascii" else old.translate(TO_DEVA)
        new_form = new if script == "ascii" else new.translate(TO_DEVA)
        new_src = new_src.replace(old_form, new_form, 1)
        for variant in (old, old.translate(TO_DEVA)):
            if variant in new_tgt:
                new_tgt = new_tgt.replace(variant, new_form, 1)
                break

    return (new_src, new_tgt) if numerals(new_src) == numerals(new_tgt) else None


def load_pairs(tsv: Path) -> list[tuple[str, str, str]]:
    rows = csv.DictReader(tsv.open(encoding="utf-8"), delimiter="\t")
    pairs = [(r["para_id"], r["sentence_marathi"].strip(),
              r["validated_translation_Dehwali_Bhili"].strip()) for r in rows]
    return [p for p in pairs if p[1] and p[2]]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  {len(rows):>7,} -> {path}")


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
    pairs = load_pairs(args.tsv)

    # Split by paragraph before augmenting, or substituted rows leak into test.
    paras = sorted({p[0] for p in pairs})
    rng.shuffle(paras)
    held = set(paras[: args.heldout_paras])
    train = [p for p in pairs if p[0] not in held]
    test = [p for p in pairs if p[0] in held]
    print(f"  {len(paras):,} paragraphs, {len(held):,} held out")
    print(f"  train {len(train):,} / test {len(test):,} sentences")

    census = Counter(classify(m, b) for _, m, b in train)
    print("\n  numeral policy:", dict(census.most_common()))

    main_rows, quarantine, dropped = [], [], 0
    for _, m, b in train:
        kind = classify(m, b)
        if kind == "verbalised":
            quarantine.append({"mar": m, "bhb": b})
            continue
        if kind == "altered":
            continue
        b = match_digit_script(b, source_digit_script(m))
        if m == b and len(m.split()) <= 3:
            dropped += 1
            continue
        main_rows.append({"mar": m, "bhb": b})
    print(f"  quarantined {len(quarantine):,} verbalised, dropped {dropped:,} short copies")

    eligible = [r for r in main_rows if numerals(r["mar"])]
    aug: list[dict] = []
    while len(aug) < args.augment and eligible:
        r = rng.choice(eligible)
        got = substitute_numerals(r["mar"], r["bhb"], rng)
        if got:
            aug.append({"mar": got[0], "bhb": got[1]})

    conv: list[dict] = []
    if args.adibhasha.exists():
        rows = csv.DictReader(args.adibhasha.open(encoding="utf-8"))
        conv = [{"hin": h, "bhb": b} for h, b in
                ((r["Hindi"].strip(), r["Bhili"].strip()) for r in rows)
                if h and b and len(h.split()) <= 8 and h != b]

    print()
    for name, rows in (("train_main", main_rows), ("train_aug", aug),
                       ("quarantine_verbalised", quarantine),
                       ("aux_conversational", conv),
                       ("test_heldout", [{"mar": m, "bhb": b} for _, m, b in test])):
        write_jsonl(args.out_dir / f"{name}.jsonl", rows)

    heldout = [{"mar": m, "bhb": b} for _, m, b in test]
    evals = {
        "GENERAL": heldout,
        "NUM_NAT": [r for r in heldout if numerals(r["mar"])],
        "SHORT_NAT": [r for r in heldout if len(r["mar"].split()) <= 4],
        # Copying scores zero here by construction, which is the point.
        "SHORT_HARD": [r for r in heldout
                       if len(r["mar"].split()) <= 4 and r["mar"] != r["bhb"]],
    }
    print()
    for name, rows in evals.items():
        write_jsonl(args.out_dir / f"eval_{name}.jsonl", rows)


if __name__ == "__main__":
    main()
