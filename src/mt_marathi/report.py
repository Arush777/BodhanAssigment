"""Assemble the Marathi<->Bhili results table from the metrics on the Hub.

Every row is reported against a COPY baseline. That is not decoration: Bhili is
lexically close enough to Marathi that echoing the input is a strong policy, and
on the unmodified model it actually WINS on every Marathi->Bhili test set. A
table without the COPY row would let a model that learned to copy look like a
model that learned to translate.
"""

from __future__ import annotations

import argparse, json, os, urllib.request
from pathlib import Path

REPO = "Arushhh/indic-translate-mar-bhili-lora"
SETS = ["GENERAL", "NUM_NAT", "SHORT_NAT", "SHORT_HARD"]
DIRS = ["mar-bhb", "bhb-mar"]


def fetch(tag: str) -> dict:
    url = f"https://huggingface.co/{REPO}/resolve/main/metrics/{tag}.metrics.json"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=40))
    except Exception as exc:
        print(f"  {tag}: not available ({exc})")
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/bhili_results.md"))
    args = ap.parse_args()

    base, ft = fetch("baseline"), fetch("finetuned")
    if not base:
        print("  no baseline metrics; nothing to assemble")
        return

    lines = ["# Marathi ↔ Dehwali Bhili — results", "",
             "chrF++ uses `word_order=2`. COPY = echo the source unchanged.",
             "`spurious_copy` = of the items whose reference differs from the source,",
             "the fraction where the model simply echoed the source back.", ""]

    lines += ["## chrF++ against the COPY baseline", "",
              "| test set | direction | COPY | base | fine-tuned | FT − COPY |",
              "|---|---|---|---|---|---|"]
    for s in SETS:
        for d in DIRS:
            k = f"{s}.{d}"
            c = base.get(f"{k}.COPY", {}).get("chrf2")
            b = base.get(k, {}).get("chrf2")
            f = ft.get(k, {}).get("chrf2")
            if c is None:
                continue
            delta = f"{f - c:+.1f}" if f is not None else "—"
            lines.append(f"| {s} | {d} | {c:.1f} | {b:.1f} | "
                         f"{f'{f:.1f}' if f is not None else '—'} | **{delta}** |")

    lines += ["", "## Spurious copy rate (lower is better)", "",
              "| test set | direction | base | fine-tuned |", "|---|---|---|---|"]
    for s in SETS:
        for d in DIRS:
            k = f"{s}.{d}"
            b = base.get(k, {}).get("spurious_copy")
            f = ft.get(k, {}).get("spurious_copy")
            if b is None:
                continue
            lines.append(f"| {s} | {d} | {b:.1f}% | "
                         f"{f'{f:.1f}%' if f is not None else '—'} |")

    lines += ["", "## Numerals", "",
              "`nsem` is numeral-sequence exact match against the SOURCE. A copying model",
              "scores 100% by construction, so this metric cannot show improvement — it",
              "only shows breakage. The signal is in digit-script match, and the reference's",
              "own preservation rate is the ceiling.", "",
              "| test set | direction | ref. ceiling | base nsem | FT nsem | base script | FT script |",
              "|---|---|---|---|---|---|---|"]
    for s in SETS:
        for d in DIRS:
            k = f"{s}.{d}"
            b, f = base.get(k, {}), ft.get(k, {})
            if "ref_preserves" not in b:
                continue
            g = lambda src, key: f"{src[key]:.1f}" if key in src else "—"
            lines.append(f"| {s} | {d} | {b['ref_preserves']:.1f}% | {g(b,'nsem')} | {g(f,'nsem')} "
                         f"| {g(b,'digit_script_match')} | {g(f,'digit_script_match')} |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
