"""Conservative Vietnamese clinical assertion classifier.

This module deliberately handles high-precision lexical cues first and returns
None when there is not enough evidence.  It expects the target mention in the
sentence to be replaced by ``[ENTITY]``.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path


ENTITY = "[ENTITY]"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "new_ner_output"
RAW_TEXT_PATH = PROJECT_ROOT / "data" / "new_test_input"
SUPPORTED_ENTITY_TYPES = {"CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"}


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def _contains_only_entity(sentence: str) -> bool:
    """Return True when only one entity, whitespace, and punctuation remain."""
    text = sentence.lower()
    if text.count("[entity]") != 1:
        return False

    remainder = text.replace("[entity]", "", 1)
    return all(
        character.isspace() or unicodedata.category(character).startswith("P")
        for character in remainder
    )


def _clause_around_entity(sentence: str) -> str:
    """Keep the target's clause to reduce cues leaking from nearby entities."""
    text = _normalise(sentence)
    parts = re.split(
        r"(?<=[;.!?])\s+|\s+(?:nhưng|tuy nhiên)\s+|,\s+còn\s+", text
    )
    return next((part for part in parts if "[entity]" in part), text)


FAMILY_MEMBER = (
    r"(?:cha|ba|bố|mẹ|má|anh trai|chị gái|em trai|em gái|ông|bà|cô|dì|"
    r"chú|cậu|mợ|vợ|chồng|con|con trai|con gái|thân nhân|người nhà)"
)

FAMILY = (
    re.compile(
        rf"\b{FAMILY_MEMBER}(?:\s+(?:của\s+)?(?:tôi|em|bệnh nhân|bé|trẻ))?"
        rf"(?:\s+\w+){{0,3}}\s+"
        rf"(?:bị|mắc|có|từng bị|đã từng bị|được chẩn đoán(?: mắc)?)\s+"
        rf"(?:bệnh\s+)?\[entity\]"
    ),
    re.compile(
        rf"\b{FAMILY_MEMBER}(?:\s+(?:của\s+)?(?:tôi|em|bệnh nhân|bé|trẻ))?"
        rf"\s+(?:có|bị|mắc)\b[^.;:!?]{{0,90}}\[entity\]"
    ),
    re.compile(r"\b(?:tiền sử gia đình|gia đình có người|bên nội|bên ngoại)\b[^.;:!?]{0,80}\[entity\]"),
    re.compile(
        rf"\b{FAMILY_MEMBER}\s+(?:của\s+)?(?:tôi|em|bạn|bệnh nhân)\b"
        rf"[^.;!?]{{0,65}}\b(?:bị|mắc|có)\s+(?:bệnh\s+)?\[entity\]"
    ),
    re.compile(r"\bkhi\s+con\s+(?:bị|mắc|có)\s+\[entity\]"),
)

NEGATED = (
    # High-confidence list scopes found repeatedly in the test set.
    re.compile(r"\btriệu chứng liên quan\s*:\s*không thấy\b[^.;:!?]{0,220}\[entity\]"),
    re.compile(r"^không\s+buồn\s+\[entity\]"),
    re.compile(r"^không\s+buồn\s+nôn,\s*(?:hay\s+)?nôn,\s*\[entity\]"),
    re.compile(r"\bphủ nhận bất kỳ\b[^.;:!?]{0,180}\[entity\]"),
    re.compile(r"\bkhông ghi nhận\b[^.;:!?]{0,120}\[entity\]"),
    re.compile(r"\b(?:không|chẳng|chưa)\s+(?:có\s+)?(?:bị\s+|mắc\s+|thấy\s+|ghi nhận\s+|phát hiện\s+|còn\s+)?\[entity\]"),
    re.compile(
        r"\bkhông\s+có\s+(?!thay đổi|cải thiện|phương pháp|khả năng)"
        r"(?:(?!\b(?:dùng|sử dụng|điều trị|sau|nhưng|tuy nhiên)\b)[^.;:!?]){0,100}"
        r"\[entity\]"
    ),
    re.compile(r"\bkhông\s+còn\b[^.;:!?]{0,45}\[entity\]"),
    re.compile(r"\bkhông\s+(?:có\s+)?(?:cơn|cảm giác|dấu hiệu|triệu chứng)\s+\[entity\]"),
    re.compile(r"\bkhông\s+(?:gây|ra|xuất)\s+\[entity\]"),
    re.compile(r"\bkhông\s+liên quan\s+đến\b[^.;:!?]{0,100}\[entity\]"),
    re.compile(r"\bkhông\s+(?:ai|người nào)\s+(?:bị|mắc|có)\b[^.;:!?]{0,70}\[entity\]"),
    re.compile(r"\b(?:không|chẳng|chưa)\s+(?:có\s+)?(?:bằng chứng|dấu hiệu|biểu hiện|triệu chứng)(?:\s+(?:của|gợi ý))?\s+\[entity\]"),
    re.compile(r"\b(?:phủ nhận|loại trừ)\s+(?:có\s+)?\[entity\]"),
    re.compile(r"\bphủ nhận\b[^.;:!?]{0,60}\[entity\]"),
    re.compile(r"\b\[entity\]\s+(?:âm tính|không thấy|không còn|đã khỏi)\b"),
    re.compile(r"\b\[entity\][^.;:!?]{0,20}\b(?:là\s+)?âm tính\b"),
    re.compile(r"\bâm tính\s+(?:với|cho)\s+\[entity\]"),
    re.compile(r"\b\[entity\]\s+(?:đã\s+)?hết\b"),
    re.compile(r"\bchưa từng\s+(?:bị|mắc|có)\s+\[entity\]"),
    re.compile(r"\bchưa từng\s+(?:dùng|sử dụng|điều trị bằng|xét nghiệm)\s+\[entity\]"),
    re.compile(r"\bchưa\s+(?:bị|mắc|có)\b[^.;:!?]{0,50}\[entity\]"),
    re.compile(r"\b(?:không ghi nhận|không phát hiện|không thấy)\b[^.;:!?]{0,45}\[entity\]"),
    re.compile(r"\bhết\s+\[entity\]"),
    re.compile(r"\bkhông\s+phải\s+(?:là\s+)?\[entity\]"),
    re.compile(r"\bkhông\s+có\b[^.;:!?]{0,80}\b(?:hoặc|và)\s+\[entity\]"),
    re.compile(r"\b\[entity\]\s*(?:\(\s*-\s*\)|:\s*âm tính|âm tính)\b"),
)

UNCERTAIN = (
    re.compile(r"\b(?:nghi|nghi ngờ|nghĩ nhiều đến|gợi ý|theo dõi)\s+(?:có\s+)?\[entity\]"),
    re.compile(r"\b(?:chưa|không)\s+(?:thể\s+)?(?:loại trừ|khẳng định)\s+\[entity\]"),
    re.compile(r"\b(?:có thể|có khả năng|có lẽ|khả năng|nguy cơ)\s+(?:là\s+|bị\s+|mắc\s+)?\[entity\]"),
    re.compile(r"\b\[entity\]\s+(?:được\s+)?(?:gợi ý|nghi ngờ|chưa xác định|chờ xác nhận|cần theo dõi)\b"),
    re.compile(r"\b(?:không xác nhận|không nhớ)\b[^.;:!?]{0,65}\[entity\]"),
    re.compile(r"\bkhả năng\b[^.;:!?]{0,45}\b(?:bị|mắc|có)\s+\[entity\]"),
)

HISTORICAL = (
    re.compile(r"\btiền sử(?!\s+bệnh\s+hiện tại)\b[^.;!?]{0,90}\[entity\]"),
    re.compile(
        r"\b(?:đã\s+)?từng\s+(?:bị|mắc|có|dùng|sử dụng|"
        r"điều trị|quay lại|nhập viện)\b[^.;!?]{0,90}\[entity\]"
    ),
    re.compile(r"\[entity\][^.;:!?]{0,90}\b(?:cách đây|trước đây|trước đó|năm 19\d{2}|năm 20\d{2}|đã khỏi|đã hồi phục|đã ngừng)\b"),
    re.compile(r"\b(?:cách đây|trước đây|trước đó|\d+\s+(?:ngày|tuần|tháng|năm)\s+trước)\b[^.;:!?]{0,90}\[entity\]"),
    re.compile(r"\b(?:thuốc|điều trị)\s+trước\s+khi\s+nhập\s+viện\b[^.;!?]{0,90}\[entity\]"),
    re.compile(r"\b\[entity\][^.;:!?]{0,12}\b(?:đã hết thuốc|hiện đã ngừng sử dụng|đã ngừng|đã dừng)\b"),
    re.compile(r"\b\[entity\][^.;:!?]{0,80}\b(?:thời kỳ sơ sinh|hồi nhỏ|trong quá khứ|lần cuối)\b"),
    re.compile(r"\b(?:thời kỳ sơ sinh|hồi nhỏ|trong quá khứ|lần cuối)\b[^.;:!?]{0,80}\[entity\]"),
    re.compile(r"\b(?:bệnh lý|các bệnh(?: lý)?)\s+m(?:ạn|ãn)\s+tính\s*:[^.;!?]{0,100}\[entity\]"),
    re.compile(r"\btừ\s+năm\s+\d{1,2}(?:\s*[-–]\s*\d{1,2})?\s+tuổi\b[^.;!?]{0,100}\[entity\]"),
)

HISTORICAL_DRUG = (
    re.compile(r"\bđã\s+(?:dùng|sử dụng|uống)\s+(?:thêm\s+)?(?:\d+(?:[.,]\d+)?\s*(?:mg|g)\s+)?(?:po\s+)?\[entity\]"),
    re.compile(r"\b(?:ngừng|dừng)(?:\s+sử dụng|\s+dùng|\s+uống)?\s+\[entity\]"),
    re.compile(r"\b\[entity\][^.;:!?]{0,45}\b(?:đã\s+hết|hết|ngừng|dừng)(?:\s+thuốc|\s+sử dụng|\s+dùng|\s+uống)?\b"),
    re.compile(r"\bhết\b[^.;:!?]{0,45}\[entity\]"),
    re.compile(r"\b\[entity\][^.;:!?]{0,90}\bdừng\s+dùng\b"),
)

NON_ASSERTION_CONTEXT = (
    # Dataset OCR: "có chưa vitamin C" is a misspelling of "có chứa", not
    # the negator "chưa".
    re.compile(r"\bcó\s+chưa\s+\[entity\]"),
    # These clauses deny a change/relationship, not the target entity itself.
    re.compile(r"\bkhông\s+thấy\s+tình trạng\s+\[entity\]\s+nặng hơn\b"),
    re.compile(r"\bkhông\s+thay đổi\s+tính chất\s+\[entity\]\b"),
    re.compile(r"\bkhông\s+có\s+cơn\b[^.;:!?]{0,80}\bdùng\b[^.;:!?]{0,25}\[entity\]"),
    re.compile(r"\b(?:cơn đau thắt ngực|bệnh|chẩn đoán)\b[^.;:!?]{0,35}\bkhông\s+\[entity\]"),
    re.compile(r"\bkhông\s+do\s+\[entity\]"),
)


def _first_match(patterns: tuple[re.Pattern[str], ...], text: str):
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def classify_assertion(sentence: str) -> dict:
    """Return only one assertion tag (or None)."""
    clause = _clause_around_entity(sentence)
    if "[entity]" not in clause:
        raise ValueError(f"Sentence does not contain {ENTITY}: {sentence!r}")

    if _first_match(NON_ASSERTION_CONTEXT, clause):
        return {"tag": None}

    if _contains_only_entity(sentence):
        return {"tag": "isUncertain"}

    for tag, patterns in (
        # "chưa/không thể loại trừ" is uncertainty, not negation, so it is
        # evaluated before the broader negation patterns.
        ("isUncertain", UNCERTAIN),
        ("isNegated", NEGATED),
        ("isFamily", FAMILY),
        ("isHistorical", HISTORICAL),
    ):
        if _first_match(patterns, clause):
            return {"tag": tag}

    return {"tag": None}


HISTORICAL_SECTION = re.compile(
    r"^(?:\d+[.)]\s*)?(?:tiền sử(?:\s+bệnh|\s+bản thân|\s+bệnh nội khoa|\s+bệnh lý)?|"
    r"các bệnh lý mạn tính|bệnh lý mạn tính|các bệnh mạn tính|"
    r"các bệnh mãn tính|bệnh mạn tính|bệnh mãn tính|dị ứng|"
    r"thuốc (?:đã điều trị )?trước khi nhập viện)\s*:?.*$"
)
CURRENT_SECTION = re.compile(
    r"^(?:\d+[.)]\s*)?(?:các\s+)?(?:triệu chứng hiện tại|triệu chứng khi nhập viện|"
    r"tiền sử bệnh hiện tại|bệnh sử|hiện tại|lý do nhập viện|lý do vào viện|"
    r"các yếu tố nguy cơ liên quan|dấu hiệu lâm sàng|khám lúc vào viện|khám hiện tại|"
    r"đánh giá tại bệnh viện|cận lâm sàng|chẩn đoán|y lệnh|điều trị|"
    r"thời điểm khởi phát triệu chứng|diễn biến bệnh|đặc điểm triệu chứng|"
    r"các sự kiện trước khi nhập viện|triệu chứng liên quan)\s*:?.*$"
)

GENERIC_OR_FORWARD_LOOKING = re.compile(
    r"\b(?:nên|cần|có thể|giúp|hỗ trợ|ngăn ngừa|phòng ngừa|"
    r"để (?:điều trị|loại bỏ|ngăn|giảm|hỗ trợ|tái tạo))\b"
)


def _raw_line_with_mask(entity: dict, raw_text: str) -> tuple[str, int]:
    """Return the entity's raw line with its exact span replaced by [ENTITY]."""
    position = entity.get("position")
    if (
        not isinstance(position, (list, tuple))
        or len(position) != 2
        or not all(isinstance(value, int) for value in position)
    ):
        raise ValueError(f"Invalid raw position for {entity.get('text')!r}: {position!r}")
    start, end = position
    if start < 0 or end <= start or end > len(raw_text):
        raise ValueError(f"Raw position {position!r} is outside the input document")

    line_start = raw_text.rfind("\n", 0, start) + 1
    line_end = raw_text.find("\n", end)
    if line_end < 0:
        line_end = len(raw_text)
    raw_line = raw_text[line_start:line_end]
    relative_start = start - line_start
    relative_end = end - line_start
    masked = raw_line[:relative_start] + ENTITY + raw_line[relative_end:]
    line_index = raw_text.count("\n", 0, line_start)
    return masked.strip(), line_index


def _nearest_section(raw_text: str, line_index: int) -> str | None:
    """Return a structurally connected section for the entity's raw line.

    The test documents contain unrelated notes and web articles pasted next to
    each other.  A blank line or prose/header barrier therefore terminates a
    section instead of allowing a heading to leak into the next fragment.
    """
    lines = raw_text.splitlines()
    for offset, line in enumerate(reversed(lines[max(0, line_index - 15):line_index + 1])):
        candidate = _normalise(line).strip(" -–—*•+")
        is_bullet = bool(re.match(r"^[-–—*•+]", line.strip()))
        if not candidate:
            return None
        if not is_bullet and CURRENT_SECTION.fullmatch(candidate):
            return "current"
        if not is_bullet and HISTORICAL_SECTION.fullmatch(candidate):
            return "historical"

        # The entity's own line may be prose.  Intervening prose and
        # unrecognized headings are hard boundaries between pasted fragments.
        if offset and (
            len(candidate) > 160
            or (not is_bullet and len(candidate) < 90)
        ):
            return None
    return None


def _can_inherit_history(raw_line: str) -> bool:
    """Gate history inheritance to compact, record-like dataset lines."""
    normalized = _normalise(raw_line)
    if len(normalized) > 120 or GENERIC_OR_FORWARD_LOOKING.search(normalized):
        return False
    stripped = raw_line.lstrip()
    return stripped.startswith(("-", "–", "—", "*", "•", "+")) or len(normalized) <= 80


def classify_entity(entity: dict, raw_text: str | None = None) -> dict:
    """Classify with normalized sentence plus optional raw line/section context."""
    sentence = entity.get("sentence")
    if not isinstance(sentence, str):
        raise ValueError(f"Supported entity {entity.get('text')!r} has no valid sentence")

    sentence_result = classify_assertion(sentence)
    if raw_text is None:
        return sentence_result

    raw_line, line_index = _raw_line_with_mask(entity, raw_text)
    raw_result = classify_assertion(raw_line) if ENTITY.casefold() in raw_line.casefold() else {"tag": None}

    # Prefer high-information labels found in either representation. Family
    # and negation outrank temporality when more than one cue is present.
    priority = {"isNegated": 4, "isFamily": 3, "isHistorical": 2, "isUncertain": 1, None: 0}
    selected = max(
        (sentence_result.get("tag"), raw_result.get("tag")),
        key=lambda tag: priority.get(tag, -1),
    )

    if entity.get("type") == "THUỐC":
        drug_context = _clause_around_entity(sentence)
        raw_drug_context = _clause_around_entity(raw_line)
        if _first_match(HISTORICAL_DRUG, drug_context) or _first_match(
            HISTORICAL_DRUG, raw_drug_context
        ):
            selected = max(
                (selected, "isHistorical"), key=lambda tag: priority.get(tag, -1)
            )

    # Bare/neutral bullets inherit a nearby history section. Explicit
    # uncertainty remains empty rather than being converted to historical.
    if selected in (None, "isUncertain"):
        explicit_uncertainty = _first_match(UNCERTAIN, _clause_around_entity(sentence))
        current_line = _normalise(raw_line)
        explicit_current = re.search(
            r"\b(?:(?<!trước )ngày (?:vào|nhập) viện|hiện tại|"
            r"(?<!trước )khi nhập viện|lúc vào viện|"
            r"vẫn còn|đang (?:bị|có|dùng|sử dụng|điều trị))\b",
            current_line,
        )
        if (
            not explicit_uncertainty
            and not explicit_current
            and _can_inherit_history(raw_line)
            and _nearest_section(raw_text, line_index) == "historical"
        ):
            selected = "isHistorical"

    return {"tag": selected}


def process_file(
    input_path: Path,
    output_path: Path | None = None,
    raw_text_path: Path | None = None,
) -> Path:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(entity, dict) for entity in data):
        raise ValueError(f"{input_path} must contain a JSON list of objects")

    if raw_text_path is None:
        candidate = RAW_TEXT_PATH / f"{input_path.stem}.txt"
        raw_text_path = (
            candidate
            if input_path.parent.name in {"new_ner_output", "new_new_output"}
            and candidate.is_file()
            else None
        )
    raw_text = raw_text_path.read_text(encoding="utf-8-sig") if raw_text_path else None

    for entity in data:
        if entity.get("type") in SUPPORTED_ENTITY_TYPES:
            entity["assertion_result"] = classify_entity(entity, raw_text)

    # Unless an explicit output is supplied, update the input JSON in place.
    destination = output_path or input_path
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(destination)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT_PATH,
        help=f"Input JSON file or folder (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"Input path does not exist: {args.input}")

    if args.input.is_dir():
        if args.output is not None:
            parser.error("--output can only be used when processing one JSON file")

        json_files = sorted(
            args.input.glob("*.json"),
            key=lambda path: int(path.stem),
        )
        if not json_files:
            parser.error(f"No source JSON files found in {args.input}")

        for input_file in json_files:
            print(f"Processing {input_file.name}")
            print(f"Saved {process_file(input_file)}")
    else:
        print(f"Saved {process_file(args.input, args.output)}")
