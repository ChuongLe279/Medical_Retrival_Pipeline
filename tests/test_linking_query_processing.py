import json

from src.Linking import query_processing


def test_query_expansion_uses_manual_map_and_skips_unsupported(tmp_path):
    file_path = tmp_path / "1.json"
    file_path.write_text(
        json.dumps(
            [
                {"text": "Tylenol", "type": "THUỐC"},
                {"text": "tylenol", "type": "THUỐC"},
                {"text": "đau đầu", "type": "TRIỆU_CHỨNG", "eng_translation": ["headache"]},
                {"text": "viêm phổi", "type": "CHẨN_ĐOÁN"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    mapping_path = tmp_path / "manual_queries.json"
    mapping_path.write_text(
        json.dumps(
            {
                "THUỐC": {"tylenol": ["Tylenol"]},
                "CHẨN_ĐOÁN": {"viêm phổi": ["Pneumonia"]},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    query_processing.query_expansion([file_path], mapping_path)
    entities = json.loads(file_path.read_text(encoding="utf-8"))

    assert entities[0]["eng_translation"] == ["Tylenol"]
    assert entities[1]["eng_translation"] == ["Tylenol"]
    assert "eng_translation" not in entities[2]
    assert entities[3]["eng_translation"] == ["Pneumonia"]


def test_missing_manual_query_fails_loudly(tmp_path):
    mapping_path = tmp_path / "manual_queries.json"
    mapping_path.write_text(
        json.dumps({"THUỐC": {}, "CHẨN_ĐOÁN": {}}, ensure_ascii=False),
        encoding="utf-8",
    )
    file_path = tmp_path / "1.json"
    file_path.write_text(
        json.dumps([{"text": "Tylenol", "type": "THUỐC"}], ensure_ascii=False),
        encoding="utf-8",
    )

    try:
        query_processing.query_expansion([file_path], mapping_path)
    except KeyError as exc:
        assert "Tylenol" in str(exc)
    else:
        raise AssertionError("Expected a missing manual query to raise KeyError")
