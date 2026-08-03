from src.Linking.fusing import fuse_entity_result


def entity(entity_type, sapbert, rapidfuzz):
    return {
        "text": "example",
        "type": entity_type,
        "retriever": {"sapbert": sapbert, "rapidfuzz": rapidfuzz},
        "assertion_result": {"tag": None},
    }


def test_agreement_beats_conflicting_single_retriever_results():
    record = entity(
        "CHẨN_ĐOÁN",
        {"AGREED": 0.82, "SAP_ONLY": 0.88},
        {"AGREED": 0.90, "FUZZ_ONLY": 0.95},
    )

    fuse_entity_result(record)

    assert record["candidate"]["code"] == "AGREED"


def test_unique_exact_rapidfuzz_match_can_override_sapbert():
    record = entity(
        "THUỐC",
        {"SEMANTIC": 0.91, "EXACT": 0.60},
        {"EXACT": 1.0, "OTHER": 0.90},
    )

    fuse_entity_result(record)

    assert record["candidate"]["code"] == "EXACT"
    assert record["candidate"]["confidence"] == 1.0
    assert record["candidate"]["ranked_candidates"][0] == {
        "code": "EXACT",
        "confidence": 1.0,
    }


def test_tied_exact_rapidfuzz_results_do_not_override_sapbert():
    record = entity(
        "CHẨN_ĐOÁN",
        {"SEMANTIC": 0.90},
        {"LEXICAL_A": 1.0, "LEXICAL_B": 1.0},
    )

    fuse_entity_result(record)

    assert record["candidate"]["code"] == "SEMANTIC"


def test_fuzzy_specific_diagnosis_does_not_replace_strong_general_match():
    record = entity(
        "CHẨN_ĐOÁN",
        {
            "GENERAL_UTI": 0.938,
            "NEONATAL_UTI": 0.730,
        },
        {
            "NEONATAL_UTI": 0.95,
            "OTHER": 0.90,
        },
    )

    fuse_entity_result(record)

    assert record["candidate"]["code"] == "GENERAL_UTI"


def test_equal_scores_have_deterministic_code_tie_break():
    record = entity(
        "CHẨN_ĐOÁN",
        {"B_CODE": 0.8, "A_CODE": 0.8},
        {},
    )

    fuse_entity_result(record)

    assert record["candidate"]["code"] == "A_CODE"


def test_empty_or_unsupported_entity_has_no_candidate():
    empty = entity("CHẨN_ĐOÁN", {}, {})
    unsupported = entity("TRIỆU_CHỨNG", {"WRONG": 1.0}, {"WRONG": 1.0})

    fuse_entity_result(empty)
    fuse_entity_result(unsupported)

    assert empty["candidate"] is None
    assert unsupported["candidate"] is None


def test_fusion_keeps_three_ranked_candidates():
    record = entity(
        "CHẨN_ĐOÁN",
        {"A": 0.90, "B": 0.80, "C": 0.70, "D": 0.60},
        {},
    )

    fuse_entity_result(record)

    assert [
        item["code"] for item in record["candidate"]["ranked_candidates"]
    ] == ["A", "B", "C"]
