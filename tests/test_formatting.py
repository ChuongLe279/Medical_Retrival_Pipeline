import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "Final_validation" / "formatting.py"
SPEC = importlib.util.spec_from_file_location("formatting", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
formatting = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(formatting)


def test_calculate_position_uses_occurrence_and_tolerates_whitespace():
    source = "đau bụng âm ỉ; Đau bụng  âm ỉ"

    assert formatting.calculate_position("Đau bụng âm ỉ", source) == [15, 29]


@pytest.mark.parametrize(
    ("primary", "qwen", "expected"),
    [
        ("isNegated", "isHistorical", ["isNegated"]),
        ("isUncertain", "isNegated", []),
        (None, "isFamily", []),
        (None, "isUncertain", []),
        (None, None, []),
    ],
)
def test_assertion_precedence(primary, qwen, expected):
    entity = {"assertion_result": {"tag": primary}, "qwen_assertion": qwen}

    assert formatting._format_assertions(entity) == expected


def test_format_file_uses_raw_offsets_and_formats_assertions():
    source = "1.  Tiền sử\n- ho\n- ho"
    entities = [
        {
            "text": "ho",
            "type": "TRIỆU_CHỨNG",
            "position": [14, 16],
            "assertion_result": {"tag": None},
            "qwen_assertion": "isNegated",
            "candidate": None,
        },
        {
            "text": "ho",
            "type": "TRIỆU_CHỨNG",
            "position": [19, 21],
            "assertion_result": {"tag": "isUncertain"},
            "qwen_assertion": "isFamily",
            "candidate": None,
        },
    ]

    assert formatting._format_file(entities, source) == [
        {
            "text": "ho",
            "type": "TRIỆU_CHỨNG",
            "assertions": [],
            "position": [14, 16],
        },
        {
            "text": "ho",
            "type": "TRIỆU_CHỨNG",
            "assertions": [],
            "position": [19, 21],
        },
    ]


def test_candidate_code_is_stringified():
    assert formatting._format_candidates({"candidate": {"code": 123}}) == ["123"]


def test_create_submission_zip_uses_required_output_directory(tmp_path):
    second = tmp_path / "2.json"
    tenth = tmp_path / "10.json"
    second.write_text(json.dumps([]), encoding="utf-8")
    tenth.write_text(json.dumps([]), encoding="utf-8")

    destination = formatting.create_submission_zip(
        [tenth, second], tmp_path / "submission.zip"
    )

    with zipfile.ZipFile(destination) as archive:
        assert archive.namelist() == ["output/2.json", "output/10.json"]
