import json

from src.Assertion.preprocessing import add_sentence_to_entities, calculating_assertion_pos


def test_assertion_position_uses_raw_test_input_offsets(tmp_path):
    raw_text = "Tiêu đề\n\nBệnh nhân không sốt."
    entity_text = "sốt"
    raw_start = raw_text.index(entity_text)

    text_path = tmp_path / "1.txt"
    json_path = tmp_path / "1.json"
    text_path.write_text(raw_text, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            [
                {
                    "text": entity_text,
                    "type": "TRIỆU_CHỨNG",
                    "position": [raw_start, raw_start + len(entity_text)],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    entities = calculating_assertion_pos(json_path, text_path)

    assert entities[0]["assertion_position"] == [
        raw_start,
        raw_start + len(entity_text),
    ]


def test_assertion_position_allows_ner_terminal_period_missing_from_raw_input(tmp_path):
    raw_text = "- Khả năng Viêm mô tế bào\n- Theo dõi thêm"
    raw_entity_text = "Viêm mô tế bào"
    raw_start = raw_text.index(raw_entity_text)

    text_path = tmp_path / "12.txt"
    json_path = tmp_path / "12.json"
    text_path.write_text(raw_text, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            [
                {
                    "text": raw_entity_text + ".",
                    "type": "CHẨN_ĐOÁN",
                    "position": [raw_start, raw_start + len(raw_entity_text)],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    entities = calculating_assertion_pos(json_path, text_path)

    assert entities[0]["text"] == raw_entity_text
    assert entities[0]["assertion_position"] == [
        raw_start,
        raw_start + len(raw_entity_text),
    ]


def test_assertion_position_matches_collapsed_ner_whitespace_to_raw_input(tmp_path):
    raw_text = "-  cơn  đau đầu nhiều hơn"
    raw_entity_text = "cơn  đau đầu"
    raw_start = raw_text.index(raw_entity_text)

    text_path = tmp_path / "23.txt"
    json_path = tmp_path / "23.json"
    text_path.write_text(raw_text, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            [
                {
                    "text": "cơn đau đầu",
                    "type": "TRIỆU_CHỨNG",
                    "position": [raw_start, raw_start + len(raw_entity_text)],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    entities = calculating_assertion_pos(json_path, text_path)

    assert entities[0]["text"] == raw_entity_text
    assert entities[0]["assertion_position"] == [
        raw_start,
        raw_start + len(raw_entity_text),
    ]


def test_sentence_comes_from_normalized_text_while_position_remains_raw(tmp_path):
    raw_text = "-  cơn  đau đầu nhiều hơn\n- Theo dõi thêm"
    normalized_text = "cơn đau đầu nhiều hơn. Theo dõi thêm."
    raw_entity_text = "cơn  đau đầu"
    raw_start = raw_text.index(raw_entity_text)

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
                "position": [raw_start, raw_start + len(raw_entity_text)],
            }],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    calculating_assertion_pos(json_path, raw_path)
    add_sentence_to_entities(normalized_path, json_path)
    entity = json.loads(json_path.read_text(encoding="utf-8"))[0]

    assert entity["assertion_position"] == [
        raw_start,
        raw_start + len(raw_entity_text),
    ]
    assert entity["sentence"] == "[ENTITY] nhiều hơn."
