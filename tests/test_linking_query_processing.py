import json

from src.Linking import query_processing


def test_query_expansion_is_type_specific_cached_and_skips_unsupported(tmp_path, monkeypatch):
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
    calls = []

    def fake_queries(text, entity_type):
        calls.append((text, entity_type))
        return ["Acetaminophen"] if entity_type == "THUỐC" else ["Pneumonia"]

    monkeypatch.setattr(query_processing, "qwen_linking_queries", fake_queries)
    query_processing.query_expansion([file_path])
    entities = json.loads(file_path.read_text(encoding="utf-8"))

    assert calls == [("Tylenol", "THUỐC"), ("viêm phổi", "CHẨN_ĐOÁN")]
    assert entities[0]["eng_translation"] == ["Acetaminophen"]
    assert entities[1]["eng_translation"] == ["Acetaminophen"]
    assert "eng_translation" not in entities[2]
    assert entities[3]["eng_translation"] == ["Pneumonia"]
