import json

from src.Assertion.preprocessing import (
    add_sentence_to_entities,
    calculating_assertion_pos,
    calculating_raw_pos,
    find_position,
)


def test_find_position_prefers_earliest_case_insensitive_occurrence():
    text = "THIẾU MEN G6PD trước, thiếu men G6PD sau"

    assert find_position("thiếu men G6PD", text) == (0, len("THIẾU MEN G6PD"))


def test_raw_position_is_preserved_but_assertion_position_is_normalized(tmp_path):
    raw_text = "-  cơn  đau đầu nhiều hơn\n- Theo dõi thêm"
    normalized_text = "cơn đau đầu nhiều hơn. Theo dõi thêm."
    raw_entity_text = "cơn  đau đầu"
    raw_start = raw_text.index(raw_entity_text)
    normalized_start = normalized_text.index("cơn đau đầu")

    raw_path = tmp_path / "raw.txt"
    normalized_path = tmp_path / "normalized.txt"
    json_path = tmp_path / "entities.json"
    raw_path.write_text(raw_text, encoding="utf-8")
    normalized_path.write_text(normalized_text, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            [{
                "text": "cơn đau đầu",
                "type": "TRIỆU_CHỨNG",
                "position": [0, len("cơn đau đầu")],
            }],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    calculating_raw_pos(json_path, raw_path)
    calculating_assertion_pos(json_path, normalized_path)
    add_sentence_to_entities(normalized_path, json_path)
    entity = json.loads(json_path.read_text(encoding="utf-8"))[0]

    assert entity["position"] == [raw_start, raw_start + len(raw_entity_text)]
    assert entity["assertion_position"] == [
        normalized_start,
        normalized_start + len("cơn đau đầu"),
    ]
    assert entity["sentence"] == "[ENTITY] nhiều hơn."


def test_normalized_assertion_position_handles_terminal_punctuation(tmp_path):
    normalized_text = "Khả năng Viêm mô tế bào. Theo dõi thêm."
    entity_text = "Viêm mô tế bào."
    normalized_start = normalized_text.index(entity_text)

    normalized_path = tmp_path / "normalized.txt"
    json_path = tmp_path / "entities.json"
    normalized_path.write_text(normalized_text, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            [{
                "text": entity_text,
                "type": "CHẨN_ĐOÁN",
                "position": [100, 114],
            }],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    entities = calculating_assertion_pos(json_path, normalized_path)

    assert entities[0]["position"] == [100, 114]
    assert entities[0]["assertion_position"] == [
        normalized_start,
        normalized_start + len(entity_text),
    ]
