"""Quality filters for mined en-mr bitext.

Samanantar is a mined corpus: sentence pairs were aligned automatically, so a
meaningful fraction are not translations of each other at all. The canonical
example, row 2 of the Marathi split:

    src: "We need to put faith in his reminders."
    tgt: "[ ७ पानांवरील चित्र]"          # "[Picture on page 7]"

Training on that teaches the model to hallucinate captions. Each filter below
targets one observed failure mode and records why it fired, so the drop counts
are auditable rather than a single opaque "cleaned" number.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

# Devanagari block. Marathi is written in Devanagari; so are Hindi and Sanskrit,
# so this is a script check, not a language check -- it catches English-on-both-
# sides and mojibake, not Hindi mislabelled as Marathi.
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
LATIN = re.compile(r"[A-Za-z]")

# Caption/figure/table residue from PDF-derived text on either side.
CAPTION_LIKE = re.compile(r"^\s*[\[\(]?\s*(fig|figure|table|image|photo|चित्र|आकृती|तक्ता)\b", re.I)
BRACKETED_ONLY = re.compile(r"^\s*[\[\(].{0,60}[\]\)]\s*$")

# A URL or an email on one side and prose on the other is never a translation.
URLISH = re.compile(r"(https?://|www\.|\S+@\S+\.\w+)")


@dataclass(frozen=True)
class FilterConfig:
    min_chars: int = 12
    max_chars: int = 400          # sentence-level corpus; documents are a separate job
    min_words_src: int = 3
    max_length_ratio: float = 2.5  # chars, longer side / shorter side
    min_script_purity: float = 0.5  # fraction of *letters* in the expected script
    max_digit_ratio: float = 0.30


def _script_purity(text: str, pattern: re.Pattern[str]) -> float:
    """Fraction of alphabetic characters that belong to the expected script."""
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return 0.0
    return sum(bool(pattern.match(c)) for c in letters) / len(letters)


def _digit_ratio(text: str) -> float:
    stripped = [c for c in text if not c.isspace()]
    if not stripped:
        return 1.0
    return sum(c.isdigit() for c in stripped) / len(stripped)


def reject_reason(src: str, tgt: str, cfg: FilterConfig) -> str | None:
    """Return the name of the first filter that rejects this pair, else None.

    Order is deliberate: cheap structural checks first, script analysis last.
    """
    src = (src or "").strip()
    tgt = (tgt or "").strip()

    if not src or not tgt:
        return "empty"
    if src == tgt:
        return "identical"

    if not (cfg.min_chars <= len(src) <= cfg.max_chars):
        return "src_length"
    if not (cfg.min_chars <= len(tgt) <= cfg.max_chars):
        return "tgt_length"
    if len(src.split()) < cfg.min_words_src:
        return "src_too_few_words"

    longer, shorter = max(len(src), len(tgt)), min(len(src), len(tgt))
    if longer / shorter > cfg.max_length_ratio:
        return "length_ratio"

    if BRACKETED_ONLY.match(src) or BRACKETED_ONLY.match(tgt):
        return "bracketed_only"
    if CAPTION_LIKE.match(src) or CAPTION_LIKE.match(tgt):
        return "caption_like"
    if bool(URLISH.search(src)) != bool(URLISH.search(tgt)):
        return "url_mismatch"

    if _digit_ratio(src) > cfg.max_digit_ratio or _digit_ratio(tgt) > cfg.max_digit_ratio:
        return "digit_heavy"

    if _script_purity(src, LATIN) < cfg.min_script_purity:
        return "src_not_latin"
    if _script_purity(tgt, DEVANAGARI) < cfg.min_script_purity:
        return "tgt_not_devanagari"

    return None


def apply(rows, cfg: FilterConfig) -> tuple[list[dict], Counter]:
    """Filter an iterable of {'src','tgt'} dicts, deduplicating as we go."""
    kept: list[dict] = []
    stats: Counter = Counter()
    seen: set[int] = set()

    for row in rows:
        stats["read"] += 1
        src, tgt = (row.get("src") or "").strip(), (row.get("tgt") or "").strip()

        reason = reject_reason(src, tgt, cfg)
        if reason is not None:
            stats[f"drop:{reason}"] += 1
            continue

        key = hash((src, tgt))
        if key in seen:
            stats["drop:duplicate"] += 1
            continue
        seen.add(key)

        kept.append({"eng": src, "mar": tgt})
        stats["kept"] += 1

    return kept, stats
