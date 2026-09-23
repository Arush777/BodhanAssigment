"""Build a blind adequacy-evaluation set across several corpora.

Spot-checking one corpus in isolation tells you almost nothing: "90% look fine"
is unfalsifiable without a comparison. Sampling from every corpus, shuffling
them together and stripping provenance turns a vibe-check into a measurement --
the number that matters is the GAP between corpora, not any single rate.

The judge sees only {id, eng, mar}. Corpus identity and quality scores are held
in key.json until verdicts are in.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

CORPORA = {
    "samanantar_raw": "data/bitext/samanantar_en_mr.jsonl",
    "bpcc_cleaned":   "data/bitext/bpcc_clean_en_mr.jsonl",
    "shiksha":        "data/bitext/shiksha_en_mr.jsonl",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-corpus", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out-dir", type=Path, default=Path("data/eval"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    picked = []
    for name, path in CORPORA.items():
        rows = [json.loads(l) for l in Path(path).open(encoding="utf-8")]
        sample = rng.sample(rows, args.per_corpus)
        picked += [{**r, "_corpus": name} for r in sample]
        print(f"  {name:<16} {len(rows):>7,} rows -> sampled {args.per_corpus}")

    rng.shuffle(picked)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    key = {}
    with (args.out_dir / "blind.jsonl").open("w", encoding="utf-8") as fh:
        for i, row in enumerate(picked, 1):
            key[str(i)] = row["_corpus"]
            fh.write(json.dumps({"id": i, "eng": row["eng"], "mar": row["mar"]},
                                ensure_ascii=False) + "\n")
    (args.out_dir / "key.json").write_text(json.dumps(key, indent=2))
    print(f"\n  wrote {len(picked)} blind items -> {args.out_dir/'blind.jsonl'}")


if __name__ == "__main__":
    main()
