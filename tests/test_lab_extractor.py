"""Tests for rule-based lab/vital extraction."""

from __future__ import annotations

import unittest

from lab_extractor import (
    ExtractionStats,
    LabSeedEntry,
    build_lab_matcher,
    detect_unparseable_vital_blob,
    extract_lab_pairs,
)


def build_test_matcher():
    """Build a compact matcher for tests."""

    entries = [
        LabSeedEntry("CREATININE", "lab_test", ("creatinin",)),
        LabSeedEntry("ALT", "lab_test", ("alt",)),
        LabSeedEntry("AST", "lab_test", ("ast",)),
        LabSeedEntry("CBC", "lab_test", ("công thức máu", "cbc")),
        LabSeedEntry("TROPONIN", "lab_test", ("troponin",)),
        LabSeedEntry("HEMOGLOBIN", "lab_test", ("huyết cầu tố",)),
        LabSeedEntry("BLOOD_PRESSURE", "vital_sign", ("huyết áp",)),
        LabSeedEntry("HEART_RATE", "vital_sign", ("mạch", "nhịp tim")),
        LabSeedEntry("RESPIRATORY_RATE", "vital_sign", ("nhịp thở",)),
        LabSeedEntry("SPO2", "vital_sign", ("spo2",)),
        LabSeedEntry("ULTRASOUND_HEPATOBILIARY", "imaging", ("Siêu âm gan mật",)),
    ]
    return build_lab_matcher(entries)


class LabExtractorTest(unittest.TestCase):
    """Unit tests for TÊN_XÉT_NGHIỆM and KẾT_QUẢ_XÉT_NGHIỆM extraction."""

    def setUp(self) -> None:
        """Create matcher."""

        self.matcher = build_test_matcher()

    def test_creatinine_connector(self) -> None:
        """creatinin là 3.0 -> name + result."""

        self.assertEqual(
            extract_lab_pairs("creatinin là 3.0", self.matcher),
            [
                {"text": "creatinin", "type": "TÊN_XÉT_NGHIỆM", "position": [0, 9]},
                {"text": "3.0", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "position": [13, 16]},
            ],
        )

    def test_alt_ast_two_pairs(self) -> None:
        """alt là 176 và ast là 287 -> two independent pairs."""

        self.assertEqual(
            extract_lab_pairs("alt là 176 và ast là 287", self.matcher),
            [
                {"text": "alt", "type": "TÊN_XÉT_NGHIỆM", "position": [0, 3]},
                {"text": "176", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "position": [7, 10]},
                {"text": "ast", "type": "TÊN_XÉT_NGHIỆM", "position": [14, 17]},
                {"text": "287", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "position": [21, 24]},
            ],
        )

    def test_full_name_and_abbreviation_in_parentheses(self) -> None:
        """Merge full name and abbreviation in parentheses."""

        self.assertEqual(
            extract_lab_pairs("công thức máu (cbc) nâng cao lên 11.3", self.matcher),
            [
                {"text": "công thức máu (cbc)", "type": "TÊN_XÉT_NGHIỆM", "position": [0, 19]},
                {"text": "11.3", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "position": [33, 37]},
            ],
        )

    def test_name_without_numeric_result(self) -> None:
        """Name match without numeric result only emits test name."""

        self.assertEqual(
            extract_lab_pairs("công thức máu bình thường", self.matcher),
            [{"text": "công thức máu", "type": "TÊN_XÉT_NGHIỆM", "position": [0, 13]}],
        )

    def test_no_lab(self) -> None:
        """No lab mention returns empty list."""

        self.assertEqual(extract_lab_pairs("bệnh nhân đau bụng", self.matcher), [])

    def test_adjacent_troponin(self) -> None:
        """troponin 0.01 -> adjacent numeric result."""

        self.assertEqual(
            extract_lab_pairs("troponin 0.01", self.matcher),
            [
                {"text": "troponin", "type": "TÊN_XÉT_NGHIỆM", "position": [0, 8]},
                {"text": "0.01", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "position": [9, 13]},
            ],
        )

    def test_hemoglobin_synonym(self) -> None:
        """huyết cầu tố 26.1 -> synonym match."""

        self.assertEqual(
            extract_lab_pairs("huyết cầu tố 26.1", self.matcher),
            [
                {"text": "huyết cầu tố", "type": "TÊN_XÉT_NGHIỆM", "position": [0, 12]},
                {"text": "26.1", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "position": [13, 17]},
            ],
        )

    def test_dimension_exclude(self) -> None:
        """Image dimension-like measurement is not a lab result."""

        stats = ExtractionStats()
        self.assertEqual(extract_lab_pairs("Tử cung đo 14,2 x 8,8 x 7,5 cm", self.matcher, stats), [])
        self.assertEqual(stats.dimension_excludes, 0)

    def test_blood_pressure_fraction(self) -> None:
        """Blood pressure result keeps fraction as one result."""

        self.assertEqual(
            extract_lab_pairs("huyết áp 159/72", self.matcher),
            [
                {"text": "huyết áp", "type": "TÊN_XÉT_NGHIỆM", "position": [0, 8]},
                {"text": "159/72", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "position": [9, 15]},
            ],
        )

    def test_spo2_range_excludes_suffix(self) -> None:
        """SpO2 range keeps number range only, excluding RA."""

        self.assertEqual(
            extract_lab_pairs("spo2 94-95 RA", self.matcher),
            [
                {"text": "spo2", "type": "TÊN_XÉT_NGHIỆM", "position": [0, 4]},
                {"text": "94-95", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "position": [5, 10]},
            ],
        )

    def test_unparseable_vital_blob(self) -> None:
        """Compact VS blob is logged/detected, but not parsed into entities."""

        text = "VS98.3 12987 56 18 99RA"
        self.assertEqual(extract_lab_pairs(text, self.matcher), [])
        self.assertEqual(len(detect_unparseable_vital_blob(text)), 1)

    def test_cardiology_compound_does_not_emit_pulse(self) -> None:
        """bác sĩ tim mạch is not a heart-rate measurement."""

        self.assertEqual(extract_lab_pairs("hẹn tái khám với bác sĩ tim mạch", self.matcher), [])

    def test_portal_vein_compound_does_not_emit_pulse(self) -> None:
        """tĩnh mạch cửa is not a heart-rate measurement."""

        self.assertEqual(extract_lab_pairs("đặt shunt dẫn lưu tĩnh mạch cửa", self.matcher), [])

    def test_artery_compound_does_not_emit_pulse(self) -> None:
        """xơ vữa động mạch is not a heart-rate measurement."""

        self.assertEqual(extract_lab_pairs("bệnh tim mạch do xơ vữa động mạch", self.matcher), [])

    def test_pulse_with_number_is_kept(self) -> None:
        """Mạch: 88 -> name + result."""

        self.assertEqual(
            extract_lab_pairs("Mạch: 88", self.matcher),
            [
                {"text": "Mạch", "type": "TÊN_XÉT_NGHIỆM", "position": [0, 4]},
                {"text": "88", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "position": [6, 8]},
            ],
        )

    def test_hypertension_does_not_emit_blood_pressure(self) -> None:
        """Tăng huyết áp without a numeric value is a condition, not a measurement."""

        self.assertEqual(extract_lab_pairs("Tăng huyết áp", self.matcher), [])

    def test_blood_pressure_with_fraction_is_kept(self) -> None:
        """huyết áp 120/70 -> name + result."""

        self.assertEqual(
            extract_lab_pairs("huyết áp 120/70", self.matcher),
            [
                {"text": "huyết áp", "type": "TÊN_XÉT_NGHIỆM", "position": [0, 8]},
                {"text": "120/70", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "position": [9, 15]},
            ],
        )

    def test_fast_heart_rate_without_number_is_rejected(self) -> None:
        """nhịp tim nhanh without a number is not emitted."""

        self.assertEqual(extract_lab_pairs("nhịp tim nhanh", self.matcher), [])

    def test_fast_heart_rate_with_number_is_kept(self) -> None:
        """nhịp tim nhanh 130 lần/phút keeps the concrete numeric result."""

        self.assertEqual(
            extract_lab_pairs("nhịp tim nhanh 130 lần/phút", self.matcher),
            [
                {"text": "nhịp tim", "type": "TÊN_XÉT_NGHIỆM", "position": [0, 8]},
                {"text": "130", "type": "KẾT_QUẢ_XÉT_NGHIỆM", "position": [15, 18]},
            ],
        )

    def test_fast_respiratory_rate_without_number_is_rejected(self) -> None:
        """Real file-84 style symptom text should not become a vital measurement."""

        self.assertEqual(extract_lab_pairs("nhịp thở nhanh và thiếu oxy", self.matcher), [])

    def test_imaging_name_without_number_is_kept(self) -> None:
        """Imaging names are kept without numeric results."""

        self.assertEqual(
            extract_lab_pairs("Siêu âm gan mật ghi nhận tình trạng giãn đường mật", self.matcher),
            [{"text": "Siêu âm gan mật", "type": "TÊN_XÉT_NGHIỆM", "position": [0, 15]}],
        )


if __name__ == "__main__":
    unittest.main()
