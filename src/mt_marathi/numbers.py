"""Number accuracy that accepts digits or words.

The reported requirement is that a number comes across correctly, in either
numeral or word form. Digit-sequence matching fails that: it scores 20 -> वीस
as a dropped numeral when it is a correct translation. This extracts values
rather than surface forms, so both spellings collapse to the same thing.

The cardinal lexicon covers Marathi, Hindi and the Bhili variants attested in
the corpus. It is not exhaustive above 100, so a value expressed in words the
lexicon does not know will be missed. Coverage is reported alongside the score.
"""

from __future__ import annotations

import re
from collections import Counter

TO_ASCII = str.maketrans("०१२३४५६७८९", "0123456789")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")

CARDINALS: dict[str, float] = {}
for words, value in [
    (["शून्य"], 0),
    # Bhili variants seen in the corpus
    (["पास"], 5), (["चोवतीस"], 34), (["आठहोव"], 800),
    (["एक", "एकुज"], 1), (["दोन", "दो", "बेन"], 2), (["तीन"], 3), (["चार"], 4),
    (["पाच", "पांच", "पाँच", "पॉंच"], 5), (["सहा", "छह", "छः"], 6), (["सात"], 7),
    (["आठ"], 8), (["नऊ", "नौ"], 9), (["दहा", "दस", "दह", "दाह"], 10),
    (["अकरा", "ग्यारह"], 11), (["बारा", "बारह"], 12), (["तेरा", "तेरह"], 13),
    (["चौदा", "चौदह"], 14), (["पंधरा", "पंद्रह", "पोंदरा"], 15),
    (["सोळा", "सोलह"], 16), (["सतरा", "सत्रह"], 17), (["अठरा", "अठारह"], 18),
    (["एकोणीस", "उन्नीस"], 19), (["वीस", "बीस"], 20),
    (["पंचवीस", "पच्चीस", "पोचीस"], 25), (["तीस"], 30), (["चाळीस", "चालीस"], 40),
    (["पन्नास", "पचास"], 50), (["साठ"], 60), (["सत्तर"], 70),
    (["ऐंशी", "अस्सी"], 80), (["नव्वद", "नब्बे"], 90),
    (["शंभर", "सौ"], 100), (["हजार", "ओजार"], 1000), (["लाख", "लॉख"], 100000),
]:
    for w in words:
        CARDINALS[w] = value

# Devanagari has no \b, so guard with "not followed/preceded by another
# Devanagari letter". Without this, तीस (30) matches inside चोवतीस (34).
_ALT = "|".join(sorted(map(re.escape, CARDINALS), key=len, reverse=True))
_WORD_RE = re.compile(rf"(?<![\u0900-\u097F])(?:{_ALT})(?![\u0900-\u097F])")

# List enumerators: "1)" or "१)" at the start of a segment are formatting, not
# content, and references routinely drop them.
_ENUM_RE = re.compile(r"^\s*[\d\u0966-\u096F]+[).]\s*")


def values(text: str) -> Counter:
    """Multiset of numeric values in `text`, from digits and from cardinals."""
    text = _ENUM_RE.sub("", text)
    found = Counter(float(x) for x in NUM_RE.findall(text.translate(TO_ASCII)))
    found.update(CARDINALS[w] for w in _WORD_RE.findall(text))
    return found


def has_number(text: str) -> bool:
    return bool(values(text))


def accuracy(preds: list[str], refs: list[str]) -> dict:
    """Fraction of number-bearing references whose values the prediction reproduces."""
    pairs = [(p, r) for p, r in zip(preds, refs) if has_number(r)]
    if not pairs:
        return {}
    exact = sum(values(p) == values(r) for p, r in pairs)
    recall_num = sum(len(values(p) & values(r)) for p, r in pairs)
    recall_den = sum(sum(values(r).values()) for p, r in pairs)
    return {
        "n_with_number": len(pairs),
        "value_match": round(exact / len(pairs) * 100, 1),
        "value_recall": round(recall_num / recall_den * 100, 1) if recall_den else 0.0,
    }
