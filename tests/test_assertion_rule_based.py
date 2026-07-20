"""Tests for rule-based assertion classification."""

from __future__ import annotations

import unittest

from src.Assertion.rule_based import classify_assertion


class AssertionRuleBasedTest(unittest.TestCase):
    def test_entity_alone_is_uncertain(self) -> None:
        self.assertEqual(classify_assertion("[ENTITY]."), {"tag": "isUncertain"})

    def test_entity_with_unicode_and_ascii_punctuation_is_uncertain(self) -> None:
        sentences = (
            " [ENTITY] ",
            "[ENTITY]?!",
            "([ENTITY]).",
            "“[ENTITY]…”",
            "— [ENTITY];",
        )

        for sentence in sentences:
            with self.subTest(sentence=sentence):
                self.assertEqual(
                    classify_assertion(sentence),
                    {"tag": "isUncertain"},
                )

    def test_entity_with_other_text_does_not_use_entity_only_rule(self) -> None:
        self.assertEqual(classify_assertion("Đã ghi nhận [ENTITY]."), {"tag": None})


if __name__ == "__main__":
    unittest.main()
