"""Tests for rule-based THUỐC extraction."""

from __future__ import annotations

import unittest

from drug_extractor import build_drug_name_matcher, extract_drug_mentions


class DrugExtractorTest(unittest.TestCase):
    """Unit tests for THUỐC span extraction."""

    def test_single_drug_with_dose_route_frequency(self) -> None:
        """Extract one full drug phrase with dose, route, and frequency."""

        matcher = build_drug_name_matcher({"amlodipine"})
        text = "amlodipine 10 mg po daily"

        self.assertEqual(
            extract_drug_mentions(text, matcher),
            [
                {
                    "text": "amlodipine 10 mg po daily",
                    "type": "THUỐC",
                    "position": [0, 25],
                }
            ],
        )

    def test_two_drugs_joined_by_and_are_separate(self) -> None:
        """Do not let the first drug span consume the second drug."""

        matcher = build_drug_name_matcher({"amlodipine", "aspirin"})
        text = "amlodipine 10 mg po daily and aspirin 81 mg po daily"

        self.assertEqual(
            extract_drug_mentions(text, matcher),
            [
                {
                    "text": "amlodipine 10 mg po daily",
                    "type": "THUỐC",
                    "position": [0, 25],
                },
                {
                    "text": "aspirin 81 mg po daily",
                    "type": "THUỐC",
                    "position": [30, 52],
                },
            ],
        )

    def test_standalone_drug_name(self) -> None:
        """Extract a standalone drug name without dose, route, or frequency."""

        matcher = build_drug_name_matcher({"aspirin"})
        text = "continue aspirin."

        self.assertEqual(
            extract_drug_mentions(text, matcher),
            [
                {
                    "text": "aspirin",
                    "type": "THUỐC",
                    "position": [9, 16],
                }
            ],
        )

    def test_no_drug(self) -> None:
        """Return an empty list when no drug name matches."""

        matcher = build_drug_name_matcher({"aspirin"})

        self.assertEqual(extract_drug_mentions("no medication listed", matcher), [])

    def test_longest_overlapping_drug_name_wins(self) -> None:
        """Prefer the longest drug-name match when one contains another."""

        matcher = build_drug_name_matcher({"aspirin", "baby aspirin"})
        text = "take baby aspirin 81 mg po daily"

        self.assertEqual(
            extract_drug_mentions(text, matcher),
            [
                {
                    "text": "baby aspirin 81 mg po daily",
                    "type": "THUỐC",
                    "position": [5, 32],
                }
            ],
        )

    def test_lab_like_terms_are_filtered_but_caffeine_is_kept(self) -> None:
        """Filter common lab/chemistry terms while keeping caffeine configurable."""

        matcher = build_drug_name_matcher(
            {
                "creatinine": [("1", "IN")],
                "kali": [("2", "IN")],
                "caffeine": [("3", "BN")],
            }
        )
        text = "Creatinine and kali were high, caffeine was documented."

        self.assertEqual(
            extract_drug_mentions(text, matcher),
            [
                {
                    "text": "caffeine",
                    "type": "THUỐC",
                    "position": [31, 39],
                }
            ],
        )

    def test_ingredient_standalone_without_medication_context_is_filtered(self) -> None:
        """An IN/PIN standalone mention without dose/route/frequency is filtered."""

        matcher = build_drug_name_matcher({"guaiac": [("1", "IN")]})

        self.assertEqual(extract_drug_mentions("guaiac negative", matcher), [])

    def test_brand_name_standalone_is_kept(self) -> None:
        """A BN standalone mention is kept even without dose/route/frequency."""

        matcher = build_drug_name_matcher({"vicodin": [("1", "BN")]})

        self.assertEqual(
            extract_drug_mentions("vicodin", matcher),
            [
                {
                    "text": "vicodin",
                    "type": "THUỐC",
                    "position": [0, 7],
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
