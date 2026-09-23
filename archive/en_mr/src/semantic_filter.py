"""Semantic alignment filter using LaBSE.

`filters.py` is a hygiene pass: it catches malformed pairs (wrong script, absurd
length ratio, figure captions). Running it over Samanantar's Marathi split keeps
95.2% of rows -- which is the tell that it is not measuring quality at all.

The dominant failure mode in a mined corpus is a pair that is perfectly
well-formed on both sides and simply does not mean the same thing:

    eng: "Pakistan goalkeeper saved the shot from Akashdeep Singh."   (hockey)
    mar: "अफगाणिस्तानकडून फिरकीपटू राशिद खानने..."                      (cricket)

Both sides are fluent, correctly scripted and similar in length, so every
structural filter passes it. Catching this needs a *meaning* comparison.

LaBSE embeds 109 languages into one space, so cosine(eng, mar) is a direct
estimate of translation equivalence. Note the honest caveat: Samanantar was
itself mined with LaBSE-based similarity, so we are raising an existing
threshold rather than applying a new signal. That bounds how much we can
expect to gain, and it belongs in the writeup.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, default=Path("data/bitext/pool.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/bitext/samanantar_en_mr_labse.jsonl"))
    ap.add_argument("--scores-out", type=Path, default=Path("data/bitext/labse_scores.json"))
    ap.add_argument("--target-rows", type=int, default=25_000)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  device: {device}")

    rows = [json.loads(l) for l in args.pool.open(encoding="utf-8")]
    print(f"  pool: {len(rows):,} pairs")

    model = SentenceTransformer("sentence-transformers/LaBSE", device=device)
    emb_en = model.encode([r["eng"] for r in rows], batch_size=args.batch_size,
                          convert_to_tensor=True, normalize_embeddings=True,
                          show_progress_bar=True)
    emb_mr = model.encode([r["mar"] for r in rows], batch_size=args.batch_size,
                          convert_to_tensor=True, normalize_embeddings=True,
                          show_progress_bar=True)

    # Both are unit-normalised, so the row-wise dot product is the cosine.
    sims = (emb_en * emb_mr).sum(dim=1).cpu().tolist()
    for row, sim in zip(rows, sims):
        row["labse"] = round(float(sim), 4)

    rows.sort(key=lambda r: -r["labse"])
    selected = rows[: args.target_rows]
    cutoff = selected[-1]["labse"]

    import statistics as st
    print(f"\n  LaBSE similarity over pool")
    print(f"    mean   {st.mean(sims):.3f}")
    print(f"    median {st.median(sims):.3f}")
    for q in (0.05, 0.25, 0.50, 0.75, 0.95):
        idx = int(q * (len(sims) - 1))
        print(f"    p{int(q*100):<5} {sorted(sims)[idx]:.3f}")
    print(f"\n  keeping top {len(selected):,} -> cutoff cosine >= {cutoff:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in selected:
            fh.write(json.dumps({"eng": row["eng"], "mar": row["mar"]}, ensure_ascii=False) + "\n")
    args.scores_out.write_text(json.dumps(
        {"cutoff": cutoff, "pool_size": len(rows), "kept": len(selected),
         "mean": st.mean(sims), "median": st.median(sims)}, indent=2))
    print(f"  wrote {len(selected):,} pairs -> {args.out}")


if __name__ == "__main__":
    main()
