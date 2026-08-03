from src.Ner.cleanup_predictions import cleanup_entities


def test_longest_nested_entity_wins_and_metadata_is_preserved():
    raw = "Thiếu men G6PD là bệnh di truyền."
    entities = [
        {"text": "Thiếu men", "type": "CHẨN_ĐOÁN", "position": [0, 9]},
        {
            "text": "Thiếu men G6PD",
            "type": "CHẨN_ĐOÁN",
            "position": [0, 14],
            "candidate": {"code": "D75A"},
            "assertion_result": {"tag": None},
        },
        {"text": "G6PD", "type": "TÊN_XÉT_NGHIỆM", "position": [10, 14]},
    ]

    cleaned, counters = cleanup_entities(entities, raw)

    assert cleaned == [entities[1]]
    assert counters["overlap_removed"] == 2


def test_truncated_decomposed_vietnamese_word_is_removed():
    raw = "Ca\u0301c tổn thương"
    entities = [{"text": "Ca", "type": "TÊN_XÉT_NGHIỆM", "position": [0, 2]}]

    cleaned, counters = cleanup_entities(entities, raw)

    assert cleaned == []
    assert counters["truncated_word"] == 1


def test_clear_type_errors_are_corrected():
    raw = "BC 5,38 G/l và COPD"
    entities = [
        {"text": "BC", "type": "THUỐC", "position": [0, 2], "candidate": None},
        {"text": "COPD", "type": "THUỐC", "position": [15, 19], "candidate": None},
    ]

    cleaned, counters = cleanup_entities(entities, raw)

    assert [entity["type"] for entity in cleaned] == [
        "TÊN_XÉT_NGHIỆM",
        "CHẨN_ĐOÁN",
    ]
    assert "candidate" not in cleaned[0]
    assert cleaned[1]["eng_translation"] == ["Chronic obstructive pulmonary disease"]
    assert counters["type:THUỐC->TÊN_XÉT_NGHIỆM"] == 1


def test_same_span_manual_type_override_is_used():
    raw = "ho"
    entities = [
        {"text": "ho", "type": "CHẨN_ĐOÁN", "position": [0, 2]},
        {"text": "ho", "type": "TRIỆU_CHỨNG", "position": [0, 2]},
    ]

    cleaned, _ = cleanup_entities(entities, raw)

    assert len(cleaned) == 1
    assert cleaned[0]["type"] == "TRIỆU_CHỨNG"


def test_partial_medical_phrase_is_expanded_from_raw_source():
    raw = "Bệnh nhân có tiền sử viêm tụy."
    start = raw.index("tụy")
    entities = [
        {
            "text": "tụy",
            "type": "CHẨN_ĐOÁN",
            "position": [start, start + 3],
            "sentence": "Bệnh nhân có tiền sử viêm [ENTITY].",
            "candidate": {"code": "wrong"},
        }
    ]

    cleaned, counters = cleanup_entities(entities, raw)

    assert cleaned[0]["text"] == "viêm tụy"
    assert cleaned[0]["position"] == [raw.index("viêm tụy"), len(raw) - 1]
    assert cleaned[0]["eng_translation"] == ["Pancreatitis"]
    assert cleaned[0]["candidate"] is None
    assert counters["span_repaired"] == 1


def test_noise_anatomy_word_is_removed_but_real_short_symptom_remains():
    raw = "Ban đầu bệnh nhân sốt."
    entities = [
        {"text": "đầu", "type": "TRIỆU_CHỨNG", "position": [4, 7]},
        {"text": "sốt", "type": "TRIỆU_CHỨNG", "position": [18, 21]},
    ]

    cleaned, _ = cleanup_entities(entities, raw)

    assert [entity["text"] for entity in cleaned] == ["sốt"]
