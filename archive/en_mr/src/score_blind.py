"""Un-blind judge verdicts and report per-corpus adequacy."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    verdicts = json.loads(Path(sys.argv[1]).read_text())
    key = json.loads(Path("data/eval/key.json").read_text())

    tally: dict[str, Counter] = {}
    for item_id, verdict in verdicts.items():
        corpus = key[str(item_id)]
        tally.setdefault(corpus, Counter())[verdict.upper()] += 1

    print(f"  {'corpus':<18}{'CORRECT':>9}{'PARTIAL':>9}{'WRONG':>7}{'n':>5}{'correct%':>10}{'adequate%':>11}")
    for corpus in sorted(tally):
        c = tally[corpus]
        n = sum(c.values())
        print(f"  {corpus:<18}{c['CORRECT']:>9}{c['PARTIAL']:>9}{c['WRONG']:>7}{n:>5}"
              f"{c['CORRECT']/n*100:>9.0f}%{(c['CORRECT']+c['PARTIAL'])/n*100:>10.0f}%")
    print("\n  n per corpus is small -- this is a sanity check on relative ordering,")
    print("  not a precise estimate of any single corpus's error rate.")


if __name__ == "__main__":
    main()
