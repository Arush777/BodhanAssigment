"""Recover training curves from a job log into CSV + a plot.

The toolkit's train_lora.sh sets WANDB_MODE=offline and HF Jobs disk is
ephemeral, so the W&B run directory dies with the job. Everything W&B would have
logged is on stdout, so the job log is the record.

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
    """Metric dicts from the log, each tagged with the arm it was printed under."""
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
            out = {"arm": arm, "split": "eval" if "eval_loss" in rec else "train"}
            # trainer prints some numbers as strings
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
