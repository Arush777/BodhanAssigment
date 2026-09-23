"""Extract the English-Marathi slice of BPCC_cleaned, de-duplicated against IN22-Gen.

Why this corpus instead of LaBSE-filtering Samanantar ourselves: BPCC_cleaned is
BPCC with a LaBSE cosine >= 0.9 threshold already applied, over the full corpus.
We implemented the same filter (see semantic_filter.py) and measured it at ~5.5
hours for a 200k subsample on an M2's MPS backend -- i.e. most of the project's
remaining wall-clock to produce a strictly worse version of a free download.
Using the published artifact is the resource-aware call; the local
implementation is retained because it documents the reasoning.

Crucially this also de-duplicates against IN22-Gen. Both Samanantar and BPCC
overlap with public benchmarks, and training on your own test set makes every
number downstream meaningless.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

ENG, MAR = 0, 6          # ClassLabel indices in BPCC_cleaned
REPO = "SPRINGLab/BPCC_cleaned"
SHARD = "data/train-00000-of-00001.parquet"


def normkey(text: str) -> str:
    """Aggressive normalisation for overlap detection: case, punctuation, spacing."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    # Collapsing whitespace is not cosmetic: stripping punctuation leaves the
    # space that sat beside it, so "hello, world" becomes "hello  world" and
    # never matches "hello world". Without this the overlap check silently
    # matches nothing and reports a clean corpus. Caught by a positive control.
    return re.sub(r"\s+", " ", text).strip()


def in22_english(cache_dir: Path) -> set[str]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id="ai4bharat/IN22-Gen", filename="data/train-00000-of-00001.parquet",
                           repo_type="dataset", cache_dir=str(cache_dir))
    col = pq.read_table(path, columns=["eng_Latn"]).column("eng_Latn").to_pylist()
    print(f"  IN22-Gen: {len(col)} English references loaded for overlap removal")
    return {normkey(c) for c in col}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-rows", type=int, default=25_000)
    ap.add_argument("--min-score", type=float, default=0.90)
    ap.add_argument("--out", type=Path, default=Path("data/bitext/bpcc_clean_en_mr.jsonl"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/hf-cache"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    blocked = in22_english(args.cache_dir)

    print(f"  downloading {REPO} ({SHARD})")
    path = hf_hub_download(repo_id=REPO, filename=SHARD, repo_type="dataset",
                           cache_dir=str(args.cache_dir))

    rows, stats = [], {"scanned": 0, "en_mr": 0, "low_score": 0, "in22_overlap": 0, "dup": 0}
    seen: set[int] = set()
    table = pq.read_table(path, columns=["src_lang", "tgt_lang", "src_text", "tgt_text", "score"])
    for batch in table.to_batches(max_chunksize=100_000):
        c = batch.to_pydict()
        for sl, tl, st, tt, sc in zip(c["src_lang"], c["tgt_lang"], c["src_text"], c["tgt_text"], c["score"]):
            stats["scanned"] += 1
            if {sl, tl} != {ENG, MAR}:
                continue
            stats["en_mr"] += 1
            eng, mar = (st, tt) if sl == ENG else (tt, st)
            if sc is not None and sc < args.min_score:
                stats["low_score"] += 1
                continue
            if normkey(eng) in blocked:
                stats["in22_overlap"] += 1
                continue
            key = hash((eng, mar))
            if key in seen:
                stats["dup"] += 1
                continue
            seen.add(key)
            rows.append({"eng": eng.strip(), "mar": mar.strip(), "labse": round(float(sc or 0), 4)})

    print("\n  extraction report")
    for k, v in stats.items():
        print(f"    {k:<16}{v:>12,}")
    print(f"    {'usable':<16}{len(rows):>12,}")

    import random
    random.Random(args.seed).shuffle(rows)
    selected = rows[: args.target_rows]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in selected:
            fh.write(json.dumps({"eng": r["eng"], "mar": r["mar"]}, ensure_ascii=False) + "\n")
    print(f"\n  wrote {len(selected):,} pairs -> {args.out}")
    if selected:
        sc = [r["labse"] for r in selected]
        print(f"  LaBSE score of selected: min {min(sc):.3f}  mean {sum(sc)/len(sc):.3f}  max {max(sc):.3f}")


if __name__ == "__main__":
    main()
