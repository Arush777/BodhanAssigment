"""Tests for the pieces of the pipeline where a silent bug would corrupt training.

Numeral handling is the whole point of this task, so the substitution
augmentation gets the most attention: it generates ~6k training rows, and if it
ever desynchronised the two sides it would teach the model to hallucinate
numbers — the exact failure we were asked to fix.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mt_marathi.bhili_data import (  # noqa: E402
    classify, match_digit_script, numerals, source_digit_script, substitute_numerals,
)


class TestNumerals:
    def test_devanagari_and_ascii_compare_equal(self):
        assert numerals("१० ते १२ किलो") == numerals("10 ते 12 किलो") == ["10", "12"]

    def test_decimals_survive(self):
        assert numerals("3.0 मि.ली.") == ["3.0"]

    def test_no_digits_is_empty(self):
        assert numerals("सेंद्रिय खते") == []


class TestDigitScript:
    def test_detects_dominant_script(self):
        assert source_digit_script("१० किलो") == "deva"
        assert source_digit_script("10 किलो") == "ascii"

    def test_round_trip_preserves_value(self):
        text = "8329432097 वर संपर्क"
        deva = match_digit_script(text, "deva")
        assert deva != text and numerals(deva) == numerals(text)
        assert match_digit_script(deva, "ascii") == text


class TestClassify:
    def test_preserved(self):
        assert classify("१० किलो", "10 किलो") == "preserved"

    def test_verbalised_is_not_corruption(self):
        # 20 -> वीस. A Bhili prose convention, not a dropped numeral; it is
        # quarantined rather than discarded.
        assert classify("20 किलो प्रति हेक्टर", "वीस किलो एक हेक्टर") == "verbalised"

    def test_altered(self):
        assert classify("10 किलो", "12 किलो") == "altered"

    def test_no_numeral(self):
        assert classify("सेंद्रिय खते", "वासडा खत") == "no_numeral"


class TestSubstitution:
    def test_both_sides_stay_aligned(self):
        rng = random.Random(0)
        src, tgt = "बियाणे 10 ते 12 किलो पुरेसे होते.", "बियारु 10 ते 12 किलो ओत जाताहा."
        for _ in range(200):
            got = substitute_numerals(src, tgt, rng)
            assert got is not None
            assert numerals(got[0]) == numerals(got[1]), got

    def test_refuses_when_sides_disagree(self):
        # Never fabricate alignment: if the numerals differ, decline.
        assert substitute_numerals("10 किलो", "12 किलो", random.Random(0)) is None

    def test_refuses_when_no_numerals(self):
        assert substitute_numerals("सेंद्रिय खते", "वासडा खत", random.Random(0)) is None

    def test_preserves_source_digit_script(self):
        rng = random.Random(1)
        got = substitute_numerals("१० किलो लागते.", "१० किलो जोवे.", rng)
        assert got is not None
        assert source_digit_script(got[0]) == "deva"
