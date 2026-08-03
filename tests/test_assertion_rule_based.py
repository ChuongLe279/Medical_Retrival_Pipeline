"""Tests for rule-based assertion classification."""

from __future__ import annotations

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from src.Assertion.rule_based import classify_assertion, classify_entity, process_file


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

    def test_first_person_em_is_not_family(self) -> None:
        sentence = "Em bị [ENTITY] đã lâu."
        self.assertEqual(classify_assertion(sentence), {"tag": None})

    def test_negation_does_not_leak_past_treatment_verb(self) -> None:
        sentence = (
            "Không có cơn rét run, dùng 1 viên [ENTITY] viên sủi, "
            "không thấy đỡ."
        )
        self.assertEqual(classify_assertion(sentence), {"tag": None})

    def test_diagnostic_modifier_is_not_assertion_negation(self) -> None:
        self.assertEqual(
            classify_assertion("Chẩn đoán: Cơn đau thắt ngực không [ENTITY]."),
            {"tag": None},
        )

    def test_explicit_relative_with_condition_is_family(self) -> None:
        sentence = "Mẹ bệnh nhân từng bị [ENTITY]."
        self.assertEqual(classify_assertion(sentence), {"tag": "isFamily"})

    def test_possessive_relative_with_extra_words_is_family(self) -> None:
        sentence = "Ông em năm nay 70 tuổi, bị [ENTITY] hai năm nay."
        self.assertEqual(classify_assertion(sentence), {"tag": "isFamily"})

    def test_relative_caregiver_context_is_not_family(self) -> None:
        sentence = "Cha mẹ cần đưa trẻ đi khám khi trẻ bị [ENTITY]."
        self.assertEqual(classify_assertion(sentence), {"tag": None})

    def test_tung_inside_diagnosis_name_is_not_history(self) -> None:
        sentence = "Tiêm thuốc cho rụng tóc từng [ENTITY]."
        self.assertEqual(classify_assertion(sentence), {"tag": None})

    def test_ocr_chua_is_not_negation(self) -> None:
        sentence = "Bổ sung thực phẩm có chưa [ENTITY]."
        self.assertEqual(classify_assertion(sentence), {"tag": None})

    def test_bare_entity_in_raw_history_section_is_historical(self) -> None:
        raw_text = "1. Tiền sử bệnh\nThuốc trước khi nhập viện\n- metoprolol"
        start = raw_text.index("metoprolol")
        entity = {
            "text": "metoprolol",
            "type": "THUỐC",
            "position": [start, start + len("metoprolol")],
            "sentence": "[ENTITY].",
        }

        self.assertEqual(
            classify_entity(entity, raw_text),
            {"tag": "isHistorical"},
        )

    def test_preadmission_phrase_is_not_mistaken_for_current_context(self) -> None:
        raw_text = "Thuốc trước khi nhập viện: gleevec"
        start = raw_text.index("gleevec")
        entity = {
            "text": "gleevec",
            "type": "THUỐC",
            "position": [start, start + len("gleevec")],
            "sentence": "Thuốc trước khi nhập viện: [ENTITY].",
        }

        self.assertEqual(classify_entity(entity, raw_text), {"tag": "isHistorical"})

    def test_bare_entity_in_current_section_remains_empty(self) -> None:
        raw_text = "Triệu chứng hiện tại\n- đau đầu"
        start = raw_text.index("đau đầu")
        entity = {
            "text": "đau đầu",
            "type": "TRIỆU_CHỨNG",
            "position": [start, start + len("đau đầu")],
            "sentence": "[ENTITY].",
        }

        self.assertEqual(
            classify_entity(entity, raw_text),
            {"tag": "isUncertain"},
        )

    def test_history_heading_does_not_leak_into_long_advice(self) -> None:
        raw_text = (
            "Tiền sử phẫu thuật / thủ thuật\n"
            "• Uống trà gừng và mật ong để loại bỏ vi khuẩn gây hại, "
            "khử mùi hôi do viêm, làm dịu niêm mạc, giảm sưng đau"
        )
        start = raw_text.index("mùi hôi")
        entity = {
            "text": "mùi hôi",
            "type": "TRIỆU_CHỨNG",
            "position": [start, start + len("mùi hôi")],
            "sentence": "Khử [ENTITY] do viêm.",
        }

        self.assertEqual(classify_entity(entity, raw_text), {"tag": None})

    def test_history_heading_stops_at_blank_fragment_boundary(self) -> None:
        raw_text = "Các bệnh lý mạn tính\n- tăng huyết áp\n\n- đau ngực"
        start = raw_text.index("đau ngực")
        entity = {
            "text": "đau ngực",
            "type": "TRIỆU_CHỨNG",
            "position": [start, start + len("đau ngực")],
            "sentence": "[ENTITY].",
        }

        self.assertEqual(classify_entity(entity, raw_text), {"tag": "isUncertain"})

    def test_history_word_inside_bullet_is_not_a_new_section(self) -> None:
        raw_text = (
            "Bệnh sử hiện tại\n"
            "Diễn biến bệnh\n"
            "- Tiền sử 2 tháng đau bụng liên tục\n"
            "- Diễn biến trong viện: tăng men gan"
        )
        start = raw_text.index("men gan")
        entity = {
            "text": "men gan",
            "type": "TRIỆU_CHỨNG",
            "position": [start, start + len("men gan")],
            "sentence": "Tăng [ENTITY].",
        }

        self.assertEqual(classify_entity(entity, raw_text), {"tag": None})

    def test_presenting_symptom_before_admission_is_current(self) -> None:
        sentence = (
            "Khoảng 5 ngày trước nhập viện, bệnh nhân xuất hiện cảm giác "
            "[ENTITY] kèm sốt."
        )
        self.assertEqual(classify_assertion(sentence), {"tag": None})

    def test_current_duration_written_as_tien_su_is_not_historical(self) -> None:
        sentence = "Tiền sử 2 ngày [ENTITY] ngày càng nặng."
        self.assertEqual(classify_assertion(sentence), {"tag": None})

    def test_current_symptom_with_onset_duration_is_not_historical(self) -> None:
        sentence = (
            "Bệnh nhân vào viện vì [ENTITY], bệnh khởi phát cách đây 1 tháng."
        )
        self.assertEqual(classify_assertion(sentence), {"tag": None})

    def test_prior_diagnosis_before_admission_remains_historical(self) -> None:
        sentence = (
            "Khoảng 1 tháng trước nhập viện, bệnh nhân bị ngã và được chẩn đoán "
            "[ENTITY]."
        )
        self.assertEqual(classify_assertion(sentence), {"tag": "isHistorical"})

    def test_resolved_symptom_is_historical_not_negated(self) -> None:
        for sentence in (
            "Hiện tại bệnh nhân hết [ENTITY].",
            "Sau đó không còn [ENTITY] nhưng vẫn đau ngực.",
            "Dường như không còn [ENTITY].",
            "[ENTITY] đã khỏi.",
        ):
            with self.subTest(sentence=sentence):
                self.assertEqual(
                    classify_assertion(sentence), {"tag": "isHistorical"}
                )

    def test_plain_absence_remains_negated(self) -> None:
        self.assertEqual(
            classify_assertion("Bệnh nhân không có [ENTITY]."),
            {"tag": "isNegated"},
        )

    def test_low_probability_diagnosis_is_uncertain(self) -> None:
        sentence = (
            "Không ghi nhận huyết khối, xạ hình cho thấy xác suất thấp "
            "[ENTITY]."
        )
        self.assertEqual(
            classify_assertion(sentence), {"tag": "isUncertain"}
        )

    def test_process_file_changes_only_supported_assertion_result(self) -> None:
        data = [
            {
                "text": "sốt",
                "type": "TRIỆU_CHỨNG",
                "position": [10, 13],
                "sentence": "Không có [ENTITY].",
                "candidate": {"code": "x"},
                "custom_progress": {"keep": True},
                "assertion_result": {"tag": None},
            },
            {
                "text": "WBC",
                "type": "TÊN_XÉT_NGHIỆM",
                "position": [20, 23],
                "assertion_result": {"tag": "keep-me"},
                "custom_progress": [1, 2, 3],
            },
        ]

        with TemporaryDirectory() as directory:
            path = Path(directory) / "1.json"
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            process_file(path)
            result = json.loads(path.read_text(encoding="utf-8"))

        assert result[0] == {
            **data[0],
            "assertion_result": {"tag": "isNegated"},
        }
        assert result[1] == data[1]


if __name__ == "__main__":
    unittest.main()
