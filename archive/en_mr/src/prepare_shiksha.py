"""Extract the English-Marathi slice of SPRINGLab/shiksha (NPTEL lecture transcripts).

This is the in-domain arm of the experiment. `shiksha` is bitext mined from
human-translated NPTEL university lecture transcriptions (arXiv:2412.09025),
CC-BY-4.0 and ungated -- the only verifiable, sizeable, downloadable
education-domain en-mr parallel corpus I could find. Note the register: spoken
university STEM lecture, NOT school textbook prose. No aligned NCERT/Balbharati
corpus exists publicly, in any Indian language pair.

Two things this does that a naive extraction would not:

1.  Splits held-out data by COURSE, not by row. Sentences from one lecture are
    highly redundant, so a random split lets near-duplicates straddle
    train/dev and reports memorisation as generalisation.
2.  Filters on the published LaBSE-style `score`. Unlike BPCC_cleaned (already
    thresholded at 0.9) shiksha ships its full range down to ~0.4, so the
    threshold here is load-bearing rather than decorative.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from mt_marathi.prepare_bpcc import normkey, in22_english

ENG, MAR = 0, 6
REPO = "SPRINGLab/shiksha"
SHARDS = [f"data/train-0000{i}-of-00004.parquet" for i in range(4)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-rows", type=int, default=25_000)
    ap.add_argument("--min-score", type=float, default=0.80)
    ap.add_argument("--held-out-courses", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("data/bitext/shiksha_en_mr.jsonl"))
    ap.add_argument("--heldout-out", type=Path, default=Path("data/bitext/shiksha_heldout_en_mr.jsonl"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/hf-cache"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    blocked = in22_english(args.cache_dir)
    rows: list[dict] = []
    stats: Counter = Counter()

    for shard in SHARDS:
        path = hf_hub_download(repo_id=REPO, filename=shard, repo_type="dataset",
                               cache_dir=str(args.cache_dir))
        print(f"  reading {shard}")
        table = pq.read_table(path, columns=["src_lang", "tgt_lang", "src_text",
                                             "tgt_text", "score", "course_id"])
        for batch in table.to_batches(max_chunksize=100_000):
            c = batch.to_pydict()
            for sl, tl, st, tt, sc, cid in zip(c["src_lang"], c["tgt_lang"], c["src_text"],
                                               c["tgt_text"], c["score"], c["course_id"]):
                stats["scanned"] += 1
                if {sl, tl} != {ENG, MAR}:
                    continue
                stats["en_mr"] += 1
                eng, mar = (st, tt) if sl == ENG else (tt, st)
                if sc is not None and sc < args.min_score:
                    stats["drop:low_score"] += 1
                    continue
                if normkey(eng) in blocked:
                    stats["drop:in22_overlap"] += 1
                    continue
                rows.append({"eng": eng.strip(), "mar": mar.strip(),
                             "score": round(float(sc or 0), 4), "course_id": cid})

    print("\n  extraction report")
    for k in ("scanned", "en_mr", "drop:low_score", "drop:in22_overlap"):
        print(f"    {k:<22}{stats[k]:>12,}")
    print(f"    {'usable':<22}{len(rows):>12,}")

    courses = sorted({r["course_id"] for r in rows})
    print(f"    {'distinct courses':<22}{len(courses):>12,}")

    # Hold out whole courses so no lecture straddles the split.
    rng = random.Random(args.seed)
    held = set(rng.sample(courses, min(args.held_out_courses, len(courses) // 5 or 1)))
    train = [r for r in rows if r["course_id"] not in held]
    heldout = [r for r in rows if r["course_id"] in held]
    print(f"    held-out courses      {len(held):>12,}  -> {len(heldout):,} pairs")

    rng.shuffle(train)
    selected = train[: args.target_rows]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for path, data in ((args.out, selected), (args.heldout_out, heldout[:2000])):
        with path.open("w", encoding="utf-8") as fh:
            for r in data:
                fh.write(json.dumps({"eng": r["eng"], "mar": r["mar"]}, ensure_ascii=False) + "\n")
        print(f"  wrote {len(data):,} -> {path}")

    if selected:
        sc = [r["score"] for r in selected]
        print(f"  score of selected: min {min(sc):.3f}  mean {sum(sc)/len(sc):.3f}  max {max(sc):.3f}")


if __name__ == "__main__":
    main()
