"""Test cho phần trích cụm ứng viên hình ảnh."""

from __future__ import annotations

import unittest

from find_imaging_gaps import extract_imaging_candidate_phrases


class FindImagingGapsTest(unittest.TestCase):
    """Các test nhỏ cho heuristic trích cụm hình ảnh."""

    def test_ultrasound_hepatobiliary_stops_before_verb(self) -> None:
        """Dừng trước cụm ghi nhận."""

        text = "Siêu âm gan mật ghi nhận tình trạng giãn đường mật"
        self.assertEqual(extract_imaging_candidate_phrases(text), [("Siêu âm gan mật", 0, 15)])

    def test_doppler_stops_before_date_and_result_verb(self) -> None:
        """Dừng trước cụm vào [Date] cho thấy."""

        text = "siêu âm Doppler động mạch vào [Date] cho thấy"
        self.assertEqual(extract_imaging_candidate_phrases(text), [("siêu âm Doppler động mạch", 0, 25)])

    def test_no_trigger_returns_empty(self) -> None:
        """Câu không có trigger hình ảnh thì trả về rỗng."""

        self.assertEqual(extract_imaging_candidate_phrases("bệnh nhân đau bụng và sốt"), [])

    def test_max_six_words_from_trigger(self) -> None:
        """Cụm quá dài bị chặn ở giới hạn sáu token dạng từ."""

        text = "Siêu âm một hai ba bốn năm sáu bảy cho thấy"
        self.assertEqual(extract_imaging_candidate_phrases(text), [("Siêu âm một hai ba bốn", 0, 22)])


if __name__ == "__main__":
    unittest.main()
