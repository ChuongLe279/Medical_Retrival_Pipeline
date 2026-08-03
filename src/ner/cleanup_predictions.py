"""Clean merged NER predictions against the raw test documents.

The model/rule union contains nested duplicates, truncated Unicode fragments,
and a few recurring type errors. This pass is deterministic and source based:
it never changes a span to text that is not present in the raw input, and it
preserves all metadata on entities that survive unchanged.
"""

from __future__ import annotations

import argparse
import copy
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON_FOLDER = PROJECT_ROOT / "data" / "new_ner_output"
DEFAULT_TEXT_FOLDER = PROJECT_ROOT / "data" / "new_test_input"

LINKABLE_TYPES = {"CHẨN_ĐOÁN", "THUỐC"}


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


# These are recurring non-entities emitted by the NER model. Real short
# medical mentions such as ho, sốt, đau, phù, nôn, and G6PD are intentionally
# not filtered by length.
INVALID_BY_TYPE = {
    "CHẨN_ĐOÁN": {
        "bình", "bệnh", "bệnh dạ", "bệnh dạibao", "bệnh dạithường",
        "c", "châ", "chậ", "chư", "chứng", "có", "cảm", "dày", "dị",
        "gan", "giãn động", "hohl", "hoại", "huyết", "hội", "iii",
        "kinh", "loạn", "miễn", "màng cứng", "mô", "mô tế bào", "nhiễm",
        "phế nang", "quai ruột", "rụng", "suy", "sơ sinh", "thấ", "tim",
        "tiêu", "tri", "trị", "trị kiê", "tính", "tăng", "tăng số lươ",
        "u châ", "u chậ", "u trị phu", "ung bướu", "vi", "xuất", "đông",
        "̉", "̉n giâ", "̣",
    },
    "THUỐC": {"ad", "acne", "cam", "tham"},
    "TÊN_XÉT_NGHIỆM": {"đồng", "vitamin"},
    "TRIỆU_CHỨNG": {
        "ảnh", "bên", "bụng", "bã", "chân", "cơ", "da", "dạ", "gia",
        "hạ", "kinh", "lưng", "miệng", "nhảy", "ngủ", "răng", "sức",
        "sờ", "thở", "tình", "tim", "trán", "tóc", "vùng", "đầu",
    },
}

# Repairs are applied only when the full phrase literally contains the old
# raw span. The optional English query is used when a repaired mention becomes
# linkable. This recovers recurrent word-level truncations without inventing
# text that is absent from the source document.
PHRASE_REPAIRS: dict[str, list[tuple[str, str, str | None]]] = {
    "sâu": [("sâu răng", "CHẨN_ĐOÁN", "Dental caries")],
    "nhồi": [("nhồi máu", "CHẨN_ĐOÁN", "Myocardial infarction")],
    "giật": [("co giật", "CHẨN_ĐOÁN", "Seizure")],
    "viêm": [("viêm cơ tim", "CHẨN_ĐOÁN", "Myocarditis")],
    "sỏi": [
        ("sỏi đoạn cuối ống mật chủ", "CHẨN_ĐOÁN", "Choledocholithiasis")
    ],
    "tụy": [("viêm tụy", "CHẨN_ĐOÁN", "Pancreatitis")],
    "hẹp": [("hẹp ống sống", "CHẨN_ĐOÁN", "Spinal stenosis")],
    "dị": [("dị ứng", "CHẨN_ĐOÁN", "Allergy")],
    "âu": [("rối loạn lo âu", "CHẨN_ĐOÁN", "Anxiety disorder")],
    "trầm": [("trầm cảm", "CHẨN_ĐOÁN", "Depressive disorder")],
    "teo": [("teo niêm mạc tử cung", "CHẨN_ĐOÁN", "Endometrial atrophy")],
    "đay": [("mày đay mạn", "CHẨN_ĐOÁN", "Chronic urticaria")],
    "vùng": [("rụng tóc từng vùng", "CHẨN_ĐOÁN", "Alopecia areata")],
    "táo": [("táo bón mãn tính", "CHẨN_ĐOÁN", "Chronic constipation")],
    "ngừng thở khi": [("ngừng thở khi ngủ", "CHẨN_ĐOÁN", "Sleep apnea")],
    "dịch": [
        ("suy giảm miễn dịch", "CHẨN_ĐOÁN", "Immunodeficiency"),
        ("dịch rỉ huyết thanh từ vết mổ", "TRIỆU_CHỨNG", None),
        ("dịch từ vết mổ", "TRIỆU_CHỨNG", None),
    ],
    "máu": [
        ("dịch thanh dịch lẫn máu từ vết mổ", "TRIỆU_CHỨNG", None),
        ("chảy máu", "TRIỆU_CHỨNG", None),
    ],
    "lẫn": [("lú lẫn", "TRIỆU_CHỨNG", None)],
    "dài": [("đau đầu kéo dài", "TRIỆU_CHỨNG", None)],
    "niệu": [("thiểu niệu", "TRIỆU_CHỨNG", None)],
    "lạ": [("ớn lạnh", "TRIỆU_CHỨNG", None)],
    "lạ": [("ớn lạnh", "TRIỆU_CHỨNG", None)],
    "suy": [("suy kiệt", "TRIỆU_CHỨNG", None)],
    "bục": [("bục chỉ khâu bụng", "TRIỆU_CHỨNG", None)],
    "tự": [("tiểu tiện không tự chủ", "TRIỆU_CHỨNG", None)],
    "chảy": [("tiêu chảy", "TRIỆU_CHỨNG", None)],
    "sờ": [("đau khi sờ nắn vùng mổ", "TRIỆU_CHỨNG", None)],
    "nhịp": [("nhịp tim", "TRIỆU_CHỨNG", None)],
    "ăn": [("ăn uống kém", "TRIỆU_CHỨNG", None)],
    "cơn": [("cơn nhịp tim chậm nặng", "TRIỆU_CHỨNG", None)],
    "cơn đau dữ": [("cơn đau dữ dội", "TRIỆU_CHỨNG", None)],
    "đau bụng liên": [("đau bụng liên tục", "TRIỆU_CHỨNG", None)],
    "chịu vùng ngực": [("khó chịu vùng ngực", "TRIỆU_CHỨNG", None)],
    "khó thở khi": [("khó thở khi gắng sức", "TRIỆU_CHỨNG", None)],
    "đau buốt khi": [("đau buốt khi đi tiểu", "TRIỆU_CHỨNG", None)],
    "dịch rò rỉ quanh": [("dịch rò rỉ quanh ống thông", "TRIỆU_CHỨNG", None)],
    "cảm giác thắt chặt ngực vùng": [
        ("cảm giác thắt chặt ngực vùng trước tim", "TRIỆU_CHỨNG", None)
    ],
    "đau kéo": [("đau kéo dài", "TRIỆU_CHỨNG", None)],
    "đau": [("đau nửa đầu dữ dội", "TRIỆU_CHỨNG", None)],
}

# Same-span model disagreements that can be resolved from the actual mention.
TYPE_OVERRIDES = {
    "đau nửa đầu": "CHẨN_ĐOÁN",
    "ho": "TRIỆU_CHỨNG",
    "paracetamol": "THUỐC",
    "rụng tóc": "TRIỆU_CHỨNG",
    "rung nhĩ": "CHẨN_ĐOÁN",
    "sốt": "TRIỆU_CHỨNG",
    "tiêu chảy": "TRIỆU_CHỨNG",
    "vitamin b12": "THUỐC",
}

# Clear recurrent type errors. Metadata incompatible with the new type is
# removed; assertion metadata remains usable when converting drug -> symptom.
TYPE_CORRECTIONS = {
    ("THUỐC", "bc"): "TÊN_XÉT_NGHIỆM",
    ("THUỐC", "neut"): "TÊN_XÉT_NGHIỆM",
    ("THUỐC", "copd"): "CHẨN_ĐOÁN",
    ("THUỐC", "stress"): "TRIỆU_CHỨNG",
}

LINKING_FIELDS = {"eng_translation", "retriever", "candidate"}
ASSERTION_FIELDS = {"assertion_position", "sentence", "assertion_result"}


def _position(entity: dict[str, Any]) -> tuple[int, int] | None:
    value = entity.get("position")
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    ):
        return None
    return value[0], value[1]


def _normalized_raw_with_map(raw_text: str) -> tuple[str, list[tuple[int, int]]]:
    """NFC/casefold raw text while retaining output-char -> raw-span maps."""
    normalized: list[str] = []
    raw_spans: list[tuple[int, int]] = []
    index = 0
    while index < len(raw_text):
        cluster_start = index
        index += 1
        while index < len(raw_text) and unicodedata.combining(raw_text[index]):
            index += 1
        cluster = unicodedata.normalize("NFC", raw_text[cluster_start:index]).casefold()
        normalized.extend(cluster)
        raw_spans.extend([(cluster_start, index)] * len(cluster))
    return "".join(normalized), raw_spans


def _masked_line(raw_text: str, start: int, end: int) -> str:
    line_start = raw_text.rfind("\n", 0, start) + 1
    line_end = raw_text.find("\n", end)
    if line_end < 0:
        line_end = len(raw_text)
    return (
        raw_text[line_start:start]
        + "[ENTITY]"
        + raw_text[end:line_end]
    ).strip()


def _repair_span(
    entity: dict[str, Any],
    raw_text: str,
    normalized_raw: str,
    raw_spans: list[tuple[int, int]],
    counters: Counter[str],
) -> bool:
    key = normalize_text(str(entity.get("text", "")))
    options = PHRASE_REPAIRS.get(key)
    position = _position(entity)
    if not options or position is None:
        return False
    old_start, old_end = position

    matches: list[tuple[int, int, str, str | None]] = []
    for phrase, new_type, query in options:
        needle = unicodedata.normalize("NFC", phrase).casefold()
        offset = 0
        while True:
            found = normalized_raw.find(needle, offset)
            if found < 0:
                break
            raw_start = raw_spans[found][0]
            raw_end = raw_spans[found + len(needle) - 1][1]
            if raw_start <= old_start and raw_end >= old_end:
                matches.append((raw_start, raw_end, new_type, query))
            offset = found + 1
    if not matches:
        return False

    # Prefer the longest containing medical phrase.
    start, end, new_type, query = max(matches, key=lambda item: item[1] - item[0])
    old_type = entity.get("type")
    entity["text"] = raw_text[start:end]
    entity["position"] = [start, end]
    entity["type"] = new_type
    entity["sentence"] = _masked_line(raw_text, start, end)
    entity.pop("assertion_position", None)
    entity["assertion_result"] = {"tag": None}

    if new_type in LINKABLE_TYPES:
        entity["eng_translation"] = [query] if query else []
        entity["retriever"] = {"sapbert": {}, "rapidfuzz": {}}
        entity["candidate"] = None
    else:
        for field in LINKING_FIELDS:
            entity.pop(field, None)
    counters["span_repaired"] += 1
    if old_type != new_type:
        counters[f"repair_type:{old_type}->{new_type}"] += 1
    return True


def _is_contextual_garbage(entity: dict[str, Any], raw_text: str) -> bool:
    """Reject recurring OCR/model prefixes while keeping real short mentions."""
    if entity.get("type") != "TRIỆU_CHỨNG":
        return False
    if normalize_text(str(entity.get("text", ""))) != "ho":
        return False
    start, end = _position(entity) or (0, 0)
    line_end = raw_text.find("\n", end)
    if line_end < 0:
        line_end = len(raw_text)
    following = normalize_text(raw_text[end:line_end])
    return following.startswith(("bệnh ", "đái tháo đường"))


def _is_truncated_word(raw_text: str, start: int, end: int) -> bool:
    """Detect spans cut through a word or before a combining accent."""
    if start < 0 or end <= start or end > len(raw_text):
        return True
    if unicodedata.combining(raw_text[start]):
        return True
    if start and (
        raw_text[start - 1].isalnum() or unicodedata.combining(raw_text[start - 1])
    ) and (raw_text[start].isalnum() or unicodedata.combining(raw_text[start])):
        return True
    if end < len(raw_text):
        next_character = raw_text[end]
        if unicodedata.combining(next_character):
            return True
        if (
            raw_text[end - 1].isalnum()
            or unicodedata.combining(raw_text[end - 1])
        ) and next_character.isalnum():
            return True
    return False


def _correct_type(entity: dict[str, Any], counters: Counter[str]) -> None:
    old_type = entity.get("type")
    key = normalize_text(str(entity.get("text", "")))
    new_type = TYPE_CORRECTIONS.get((old_type, key))
    if new_type is None or new_type == old_type:
        return

    entity["type"] = new_type
    counters[f"type:{old_type}->{new_type}"] += 1
    if new_type not in LINKABLE_TYPES:
        for field in LINKING_FIELDS:
            entity.pop(field, None)
    if new_type not in {"CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"}:
        for field in ASSERTION_FIELDS:
            entity.pop(field, None)
    if new_type == "CHẨN_ĐOÁN" and key == "copd":
        entity["eng_translation"] = ["Chronic obstructive pulmonary disease"]
        entity["retriever"] = {"sapbert": {}, "rapidfuzz": {}}
        entity["candidate"] = None


def _quality(entity: dict[str, Any]) -> tuple[int, int, int, str]:
    """Prefer requested types, useful metadata, then the longest span."""
    text_key = normalize_text(str(entity.get("text", "")))
    requested_type = TYPE_OVERRIDES.get(text_key)
    type_match = int(requested_type is not None and entity.get("type") == requested_type)
    linked = int(bool((entity.get("candidate") or {}).get("code")))
    start, end = _position(entity) or (0, 0)
    return type_match, linked, end - start, str(entity.get("type", ""))


def cleanup_entities(
    entities: list[dict[str, Any]], raw_text: str
) -> tuple[list[dict[str, Any]], Counter[str]]:
    counters: Counter[str] = Counter()
    valid: list[dict[str, Any]] = []
    normalized_raw, raw_spans = _normalized_raw_with_map(raw_text)

    for original in entities:
        entity = copy.deepcopy(original)
        position = _position(entity)
        if position is None:
            counters["invalid_position"] += 1
            continue
        start, end = position
        if start < 0 or end > len(raw_text) or end <= start:
            counters["invalid_position"] += 1
            continue

        # The raw source is authoritative for both submitted text and boundary
        # validation. This also safely preserves decomposed Vietnamese Unicode.
        raw_span = raw_text[start:end]
        if normalize_text(raw_span) != normalize_text(str(entity.get("text", ""))):
            counters["text_mismatch"] += 1
            continue
        entity["text"] = raw_span

        repaired = _repair_span(
            entity, raw_text, normalized_raw, raw_spans, counters
        )
        position = _position(entity)
        if position is None:
            counters["invalid_position"] += 1
            continue
        start, end = position
        raw_span = raw_text[start:end]

        if not repaired and _is_truncated_word(raw_text, start, end):
            counters["truncated_word"] += 1
            continue

        _correct_type(entity, counters)
        entity_type = str(entity.get("type", ""))
        key = normalize_text(raw_span)
        if key in INVALID_BY_TYPE.get(entity_type, set()):
            counters[f"invalid_mention:{entity_type}"] += 1
            continue
        if _is_contextual_garbage(entity, raw_text):
            counters["contextual_garbage"] += 1
            continue
        valid.append(entity)

    # Resolve exact spans first, using manual type decisions where available.
    by_span: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for entity in valid:
        by_span.setdefault(_position(entity), []).append(entity)  # type: ignore[arg-type]
    exact_resolved: list[dict[str, Any]] = []
    for group in by_span.values():
        if len(group) > 1:
            counters["exact_duplicate_removed"] += len(group) - 1
        exact_resolved.append(max(group, key=_quality))

    # Longest-first suppression removes both same-type fragments and nested
    # cross-type fragments (e.g. G6PD inside "thiếu men G6PD").
    selected: list[dict[str, Any]] = []
    ordered = sorted(
        exact_resolved,
        key=lambda entity: (
            -(_position(entity)[1] - _position(entity)[0]),  # type: ignore[index]
            -_quality(entity)[0],
            _position(entity)[0],  # type: ignore[index]
            str(entity.get("type", "")),
        ),
    )
    for entity in ordered:
        start, end = _position(entity)  # type: ignore[misc]
        if any(max(start, kept_start) < min(end, kept_end) for kept_start, kept_end in map(_position, selected)):
            counters["overlap_removed"] += 1
            continue
        selected.append(entity)

    selected.sort(key=lambda entity: (_position(entity), str(entity.get("type", ""))))
    counters["kept"] = len(selected)
    counters["removed"] = len(entities) - len(selected)
    return selected, counters


def process_folder(json_folder: Path, text_folder: Path, *, apply: bool) -> Counter[str]:
    totals: Counter[str] = Counter()
    json_files = sorted(json_folder.glob("*.json"), key=lambda path: int(path.stem))
    for json_path in json_files:
        text_path = text_folder / f"{json_path.stem}.txt"
        if not text_path.is_file():
            raise FileNotFoundError(text_path)
        entities = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(entities, list) or not all(isinstance(item, dict) for item in entities):
            raise ValueError(f"{json_path} must contain a list of objects")
        raw_text = text_path.read_text(encoding="utf-8-sig")
        cleaned, counters = cleanup_entities(entities, raw_text)
        totals.update(counters)
        if apply:
            temporary = json_path.with_name(f".{json_path.name}.tmp")
            temporary.write_text(
                json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(json_path)
    totals["files"] = len(json_files)
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-folder", type=Path, default=DEFAULT_JSON_FOLDER)
    parser.add_argument("--text-folder", type=Path, default=DEFAULT_TEXT_FOLDER)
    parser.add_argument("--apply", action="store_true", help="Write cleaned JSON files")
    args = parser.parse_args()
    totals = process_folder(args.json_folder, args.text_folder, apply=args.apply)
    print(json.dumps(dict(totals), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
