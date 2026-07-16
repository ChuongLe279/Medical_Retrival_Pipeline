"""Tests for merging entity outputs from separate sources."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ner.merge_entities import (
    load_entities_from_source,
    merge_entities_for_file,
    process_all_files,
    validate_merged_output,
)


class MergeEntitiesTest(unittest.TestCase):
    """Unit tests for entity merge behavior."""

    def test_no_overlap_merges_and_sorts_by_position(self) -> None:
        """Three non-overlapping sources are merged and sorted by span."""

        merged = merge_entities_for_file(
            [{"text": "đau bụng", "type": "TRIỆU_CHỨNG", "position": [20, 28]}],
            [{"text": "aspirin", "type": "THUỐC", "position": [0, 7]}],
            [{"text": "creatinin", "type": "TÊN_XÉT_NGHIỆM", "position": [10, 19]}],
        )

        self.assertEqual(
            merged,
            [
                {"text": "aspirin", "type": "THUỐC", "position": [0, 7]},
                {"text": "creatinin", "type": "TÊN_XÉT_NGHIỆM", "position": [10, 19]},
                {"text": "đau bụng", "type": "TRIỆU_CHỨNG", "position": [20, 28]},
            ],
        )

    def test_exact_duplicate_same_position_and_type_keeps_one(self) -> None:
        """Duplicate position/type from two sources collapses to one entity."""

        merged = merge_entities_for_file(
            [{"text": "aspirin", "type": "THUỐC", "position": [0, 7]}],
            [{"text": "aspirin", "type": "THUỐC", "position": [0, 7]}],
            [],
        )

        self.assertEqual(
            merged,
            [{"text": "aspirin", "type": "THUỐC", "position": [0, 7]}],
        )

    def test_same_position_different_type_prefers_rule_based_and_logs(self) -> None:
        """Rule-based entities win over NER for same-span type conflicts."""

        conflicts = []
        merged = merge_entities_for_file(
            [{"text": "CRP", "type": "TRIỆU_CHỨNG", "position": [0, 3]}],
            [],
            [{"text": "CRP", "type": "TÊN_XÉT_NGHIỆM", "position": [0, 3]}],
            source_file="1.json",
            conflict_log=conflicts,
        )

        self.assertEqual(
            merged,
            [{"text": "CRP", "type": "TÊN_XÉT_NGHIỆM", "position": [0, 3]}],
        )
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["file"], "1.json")
        self.assertEqual(conflicts[0]["conflict_type"], "same_position_different_type")
        self.assertEqual(conflicts[0]["dropped_entity"]["source"], "ner")

    def test_partial_overlap_keeps_both_and_logs_marker(self) -> None:
        """Partial overlaps are logged but left untouched."""

        conflicts = []
        merged = merge_entities_for_file(
            [{"text": "đau bụng nhiều", "type": "TRIỆU_CHỨNG", "position": [0, 14]}],
            [],
            [{"text": "bụng nhiều", "type": "TÊN_XÉT_NGHIỆM", "position": [4, 14]}],
            conflict_log=conflicts,
        )

        self.assertEqual(len(merged), 2)
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["conflict_type"], "partial_overlap")
        self.assertTrue(conflicts[0]["_partial_overlap"])
        self.assertNotIn("_partial_overlap", merged[0])
        self.assertNotIn("_partial_overlap", merged[1])

    def test_missing_sources_processes_single_available_source(self) -> None:
        """A file present in only one source still gets merged output."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ner_dir = root / "ner"
            drug_dir = root / "drug"
            lab_dir = root / "lab"
            output_dir = root / "merged"
            test_dir = root / "test"
            for directory in (ner_dir, drug_dir, lab_dir, test_dir):
                directory.mkdir()

            (drug_dir / "1.json").write_text(
                json.dumps([{"text": "aspirin", "type": "THUỐC", "position": [0, 7]}], ensure_ascii=False),
                encoding="utf-8",
            )
            (test_dir / "1.txt").write_text("aspirin", encoding="utf-8")

            summary = process_all_files(ner_dir, drug_dir, lab_dir, output_dir, test_dir)
            merged = json.loads((output_dir / "1.json").read_text(encoding="utf-8"))

            self.assertEqual(summary["total_files_processed"], 1)
            self.assertEqual(summary["empty_files"], 0)
            self.assertEqual(merged, [{"text": "aspirin", "type": "THUỐC", "position": [0, 7]}])

    def test_all_three_sources_missing_for_any_name_does_not_crash(self) -> None:
        """Empty source directories simply produce an empty batch summary."""

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ner_dir = root / "ner"
            drug_dir = root / "drug"
            lab_dir = root / "lab"
            output_dir = root / "merged"
            for directory in (ner_dir, drug_dir, lab_dir):
                directory.mkdir()

            summary = process_all_files(ner_dir, drug_dir, lab_dir, output_dir)

            self.assertEqual(summary["total_files_processed"], 0)
            self.assertEqual(summary["empty_files"], 0)
            self.assertTrue((output_dir / "merge_summary.json").exists())

    def test_validate_merged_output_reports_offset_mismatch(self) -> None:
        """Validation catches text that does not match the source span."""

        errors = validate_merged_output(
            [{"text": "aspirin", "type": "THUỐC", "position": [0, 7]}],
            "warfarin",
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("text mismatch", errors[0])

    def test_invalid_json_source_returns_empty_list(self) -> None:
        """Invalid JSON is treated as an empty source."""

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.json"
            path.write_text("{not json", encoding="utf-8")

            self.assertEqual(load_entities_from_source(path), [])


if __name__ == "__main__":
    unittest.main()
