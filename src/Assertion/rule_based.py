"""Conservative Vietnamese clinical assertion classifier.

This module deliberately handles high-precision lexical cues first and returns
None when there is not enough evidence.  It expects the target mention in the
sentence to be replaced by ``[ENTITY]``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ENTITY = "[ENTITY]"
DEFAULT_INPUT_PATH = Path(__file__).resolve().parent / "data" / "mock_ner"


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _clause_around_entity(sentence: str) -> str:
    """Keep the target's clause to reduce cues leaking from nearby entities."""
    text = _normalise(sentence)
    parts = re.split(
        r"(?<=[;:.!?])\s+|\s+(?:nhưng|tuy nhiên)\s+|,\s+còn\s+", text
    )
    return next((part for part in parts if "[entity]" in part), text)


FAMILY = re.compile(
    r"\b(?:cha|ba|bố|mẹ|má|anh|chị|em|ông|bà|cô|dì|chú|cậu|mợ|"
    r"vợ|chồng|con|thân nhân|gia đình|bên nội|bên ngoại|bác(?!\s+sĩ))\b"
)

NEGATED = (
    re.compile(r"\b(?:không|chẳng|chưa)\s+(?:có\s+)?(?:bị\s+|mắc\s+|thấy\s+|ghi nhận\s+|phát hiện\s+|còn\s+)?\[entity\]"),
    re.compile(r"\b(?:không|chẳng|chưa)\s+(?:có\s+)?(?:bằng chứng|dấu hiệu|biểu hiện|triệu chứng)(?:\s+(?:của|gợi ý))?\s+\[entity\]"),
    re.compile(r"\b(?:phủ nhận|loại trừ)\s+(?:có\s+)?\[entity\]"),
    re.compile(r"\bphủ nhận\b[^.;:!?]{0,60}\[entity\]"),
    re.compile(r"\b\[entity\]\s+(?:âm tính|không thấy|không còn|đã khỏi)\b"),
    re.compile(r"\bchưa từng\s+(?:bị|mắc|có)\s+\[entity\]"),
)

UNCERTAIN = (
    re.compile(r"\b(?:nghi|nghi ngờ|nghĩ nhiều đến|gợi ý|theo dõi)\s+(?:có\s+)?\[entity\]"),
    re.compile(r"\b(?:chưa|không)\s+(?:thể\s+)?(?:loại trừ|khẳng định)\s+\[entity\]"),
    re.compile(r"\b(?:có thể|có khả năng|có lẽ|khả năng|nguy cơ)\s+(?:là\s+|bị\s+|mắc\s+)?\[entity\]"),
    re.compile(r"\b\[entity\]\s+(?:được\s+)?(?:gợi ý|nghi ngờ|chưa xác định|chờ xác nhận|cần theo dõi)\b"),
)

HISTORICAL = (
    re.compile(r"\b(?:tiền sử|từng|đã từng)\b[^.;:!?]{0,45}\[entity\]"),
    re.compile(r"\[entity\][^.;:!?]{0,45}\b(?:cách đây|trước đây|năm 19\d{2}|năm 20\d{2}|đã khỏi|đã hồi phục)\b"),
    re.compile(r"\b(?:cách đây|trước đây|\d+\s+(?:ngày|tuần|tháng|năm)\s+trước)\b[^.;:!?]{0,45}\[entity\]"),
)


def _first_match(patterns: tuple[re.Pattern[str], ...], text: str):
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def classify_assertion(sentence: str) -> dict:
    """Return one assertion tag or None, plus auditable evidence."""
    clause = _clause_around_entity(sentence)
    if "[entity]" not in clause:
        raise ValueError(f"Sentence does not contain {ENTITY}: {sentence!r}")

    # Experiencer is checked first because a family mention describes someone
    # other than the patient. With a one-tag schema, that distinction is the
    # most important one to preserve.
    family_match = FAMILY.search(clause)
    if family_match and not re.search(r"\bbệnh (?:nhân|nhi)\b", family_match.group(0)):
        return {"tag": "isFamily", "confidence": 1.0, "evidence": family_match.group(0)}

    for tag, patterns in (
        # "chưa/không thể loại trừ" is uncertainty, not negation, so it is
        # evaluated before the broader negation patterns.
        ("isUncertain", UNCERTAIN),
        ("isNegated", NEGATED),
        ("isHistorical", HISTORICAL),
    ):
        evidence = _first_match(patterns, clause)
        if evidence:
            return {"tag": tag, "confidence": 1.0, "evidence": evidence}

    return {"tag": None, "confidence": 1.0, "evidence": None}


def process_file(input_path: Path, output_path: Path | None = None) -> Path:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    for entity in data:
        entity["assertion_result"] = classify_assertion(entity["sentence"])

    destination = output_path or input_path.with_name(f"{input_path.stem}.json")
    destination.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
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

    if args.input.is_dir():
        if args.output is not None:
            parser.error("--output can only be used when processing one JSON file")

        input_files = sorted(
            path
            for path in args.input.glob("*.json")
            if not path.stem.endswith("_v2")
        )
        if not input_files:
            parser.error(f"No source JSON files found in {args.input}")

        for input_file in input_files:
            print(f"Processing {input_file.name}")
            print(f"Saved {process_file(input_file)}")
    else:
        print(f"Saved {process_file(args.input, args.output)}")
