"""Build a filtered en-mr bitext corpus from Samanantar.

Output is bitext JSONL -- one {"eng": ..., "mar": ...} per line -- which is
exactly what `bodhan_genai.mt.data.render` consumes. We deliberately stop
there: the prompt contract (instruction phrasing, target-language naming,
direction reversal) is owned by the Bodhan toolkit's renderer, and
reimplementing it here would be the single easiest way to silently break the
model. See docs/DOCUMENTATION.md.

    python -m mt_marathi.prepare_data --target-rows 80000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from mt_marathi.filters import FilterConfig, apply

REPO = "ai4bharat/samanantar"
SHARDS = ["mr/train-00000-of-00002.parquet", "mr/train-00001-of-00002.parquet"]


def load_rows(cache_dir: Path):
    """Stream Samanantar's Marathi shards as {'src','tgt'} dicts."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    for shard in SHARDS:
        path = hf_hub_download(
            repo_id=REPO, filename=shard, repo_type="dataset", cache_dir=str(cache_dir)
        )
        print(f"  reading {shard}")
        table = pq.read_table(path, columns=["src", "tgt"])
        for batch in table.to_batches(max_chunksize=50_000):
            cols = batch.to_pydict()
            for src, tgt in zip(cols["src"], cols["tgt"]):
                yield {"src": src, "tgt": tgt}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-rows", type=int, default=80_000,
                    help="how many pairs to keep; we are compute-bound, not data-bound")
    ap.add_argument("--out", type=Path, default=Path("data/bitext/samanantar_en_mr.jsonl"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/hf-cache"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = FilterConfig()
    kept, stats = apply(load_rows(args.cache_dir), cfg)

    print("\n  filter report")
    print(f"    {'read':<24}{stats['read']:>10,}")
    for key in sorted(k for k in stats if k.startswith("drop:")):
        print(f"    {key:<24}{stats[key]:>10,}")
    print(f"    {'kept':<24}{stats['kept']:>10,}"
          f"   ({100 * stats['kept'] / max(stats['read'], 1):.1f}% of input)")

    # Subsample rather than train on everything: 17 A100-hours is the budget,
    # and a clean 80k beats a noisy 2M for a domain-adaptation LoRA.
    random.Random(args.seed).shuffle(kept)
    selected = kept[: args.target_rows]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in selected:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n  wrote {len(selected):,} pairs -> {args.out}")


if __name__ == "__main__":
    main()
