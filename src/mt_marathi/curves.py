"""Extract training curves from a job log into CSV + a plot.

The Bodhan toolkit's `scripts/mt/train_lora.sh` sets `WANDB_MODE=offline` by
default -- sensible for air-gapped training nodes, but on an ephemeral HF Job it
means the run directory is destroyed on exit and nothing reaches the cloud.

Every metric W&B would have recorded is still printed to stdout, so the job log
is a complete record. This recovers it.

    python -m mt_marathi.curves data/job2.log --out results/
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path

ARM_RE = re.compile(r"=== ARM: (\w+) ===")
DICT_RE = re.compile(r"\{'(?:loss|eval_loss)'.*?\}")


def parse(log: Path) -> list[dict]:
    """Walk the log, tagging each metric dict with the arm it belongs to."""
    rows, arm = [], "unknown"
    for line in log.read_text(errors="ignore").splitlines():
        m = ARM_RE.search(line)
        if m:
            arm = m.group(1)
            continue
        for d in DICT_RE.findall(line):
            try:
                rec = ast.literal_eval(d)
            except (ValueError, SyntaxError):
                continue
            # The trainer prints numbers as strings; coerce what we can.
            out = {"arm": arm, "split": "eval" if "eval_loss" in rec else "train"}
            for k, v in rec.items():
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    out[k] = v
            rows.append(out)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    rows = parse(args.log)
    if not rows:
        print("  no metric lines found")
        return
    args.out.mkdir(parents=True, exist_ok=True)

    cols = sorted({k for r in rows for k in r})
    csv_path = args.out / "training_curves.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    arms = sorted({r["arm"] for r in rows})
    print(f"  {len(rows)} metric rows across arms: {arms}")
    print(f"  wrote {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib absent; CSV written, skipping plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm in arms:
        tr = [r for r in rows if r["arm"] == arm and r["split"] == "train" and "epoch" in r]
        ev = [r for r in rows if r["arm"] == arm and r["split"] == "eval" and "epoch" in r]
        if tr:
            axes[0].plot([r["epoch"] for r in tr], [r["loss"] for r in tr], label=arm, lw=1)
        if ev:
            axes[1].plot([r["epoch"] for r in ev], [r["eval_loss"] for r in ev],
                         marker="o", ms=3, label=arm, lw=1)
    axes[0].set(title="train loss", xlabel="epoch", ylabel="loss")
    axes[1].set(title="eval loss", xlabel="epoch", ylabel="eval_loss")
    for a in axes:
        a.legend(fontsize=8)
        a.grid(alpha=.3)
    fig.tight_layout()
    png = args.out / "training_curves.png"
    fig.savefig(png, dpi=130)
    print(f"  wrote {png}")


if __name__ == "__main__":
    main()
