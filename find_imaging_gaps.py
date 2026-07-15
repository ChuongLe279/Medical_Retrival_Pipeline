"""Tìm các khoảng thiếu của dictionary chẩn đoán hình ảnh trong bệnh án.

Script này chỉ tạo báo cáo, không tự sửa dictionary. Nó quét các file test
thô để tìm những cụm giống tên kỹ thuật hình ảnh, so với matcher hiện tại
từ ``lab_dict_merged.json``, rồi báo các cụm mà dictionary chỉ bắt được
phần chung ngắn hơn hoặc không bắt được gì.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from lab_extractor import RegexLabMatcher, build_lab_matcher, load_seed_dict, normalize_surface


LOGGER = logging.getLogger(__name__)

IMAGING_TRIGGER_WORDS = [
    "Chụp cộng hưởng từ",
    "chụp cộng hưởng từ",
    "Chụp cắt lớp",
    "chụp cắt lớp",
    "Chụp X-quang",
    "chụp x-quang",
    "Chụp CT",
    "chụp ct",
    "Chụp MRI",
    "chụp mri",
    "Điện tâm đồ",
    "điện tâm đồ",
    "Siêu âm",
    "siêu âm",
    "Nội soi",
    "nội soi",
    "Doppler",
    "doppler",
    "Điện tim",
    "điện tim",
    "Điện não",
    "điện não",
]

STOP_PHRASES = (
    "ghi nhận",
    "cho thấy",
    "phát hiện",
    "kết luận",
    "kết quả",
)
STOP_WORDS = {
    "thấy",
    "và",
    "hoặc",
    "nhưng",
    "vào",
    "để",
    "cần",
    "được",
    "cho",
    "trong",
    "là",
    "kể",
    "theo",
    "ở",
    "sau",
}
MAX_WORDS_FROM_TRIGGER = 6
TRIGGER_RE = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(trigger) for trigger in sorted(IMAGING_TRIGGER_WORDS, key=len, reverse=True)) + r")(?!\w)",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"\w+(?:[-/]\w+)*|[^\w\s]", re.UNICODE)
PUNCTUATION_STOPS = {",", ".", ":", ";", "!", "?", "\n", "(", ")", "[", "]"}


def token_count(text: str) -> int:
    """Đếm số token dạng từ trong một cụm."""

    return len(re.findall(r"\w+(?:[-/]\w+)*", text, flags=re.UNICODE))


def should_stop(text: str, position: int, token: str) -> bool:
    """Cho biết có nên dừng mở rộng cụm ngay trước token hiện tại không."""

    if token in PUNCTUATION_STOPS:
        return True
    tail = text[position:].lower()
    if any(tail.startswith(phrase) for phrase in STOP_PHRASES):
        return True
    return token.lower() in STOP_WORDS


def extract_imaging_candidate_phrases(text: str) -> list[tuple[str, int, int]]:
    """Trích các cụm ứng viên hình ảnh từ văn bản thô bằng heuristic.

    Một cụm bắt đầu bằng một trigger trong ``IMAGING_TRIGGER_WORDS`` rồi mở
    rộng sang phải cho đến khi gặp động từ/cụm dừng, liên từ, dấu câu, hoặc
    đã lấy đủ sáu token dạng từ tính từ trigger. Không dùng NLP hay POS tagger.
    """

    candidates: list[tuple[str, int, int]] = []
    seen_spans: set[tuple[int, int]] = set()

    for trigger_match in TRIGGER_RE.finditer(text):
        start = trigger_match.start()
        end = trigger_match.end()
        words_used = token_count(trigger_match.group(0))

        for token_match in TOKEN_RE.finditer(text, end):
            between = text[end : token_match.start()]
            if "\n" in between:
                break
            token = token_match.group(0)
            if should_stop(text, token_match.start(), token):
                break
            if token_match.start() > end and not between.isspace():
                break
            if re.match(r"\w", token, flags=re.UNICODE):
                if words_used >= MAX_WORDS_FROM_TRIGGER:
                    break
                words_used += 1
                end = token_match.end()
                continue
            break

        phrase = re.sub(r"\s+", " ", text[start:end]).strip()
        span = (start, end)
        if phrase and span not in seen_spans:
            seen_spans.add(span)
            candidates.append((phrase, start, end))

    selected: list[tuple[str, int, int]] = []
    for candidate in sorted(candidates, key=lambda item: (-(item[2] - item[1]), item[1])):
        _, start, end = candidate
        if any(start >= kept_start and end <= kept_end for _, kept_start, kept_end in selected):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda item: (item[1], item[2]))


def context_for(text: str, start: int, end: int, radius: int = 90) -> str:
    """Lấy đoạn ngữ cảnh ngắn quanh một span để dễ review."""

    return re.sub(r"\s+", " ", text[max(0, start - radius) : min(len(text), end + radius)]).strip()


def best_match_for_phrase(phrase: str, matcher: RegexLabMatcher) -> tuple[str, int] | None:
    """Lấy match dài nhất bắt đầu từ đầu cụm ứng viên."""

    matches = [match for match in matcher.find(phrase) if match.start == 0]
    if not matches:
        return None
    best = max(matches, key=lambda item: (item.end - item.start, -item.start))
    return best.text, best.end - best.start


def gap_bucket(
    groups: dict[str, dict[str, object]],
    phrase: str,
    file_name: str,
    context: str,
    matched_portion: str | None = None,
) -> None:
    """Thêm một lần xuất hiện gap vào nhóm báo cáo tương ứng."""

    key = normalize_surface(phrase)
    if key not in groups:
        groups[key] = {
            "full_phrase": phrase,
            "frequency": 0,
            "examples": [],
        }
        if matched_portion is not None:
            groups[key]["matched_portion"] = matched_portion

    groups[key]["frequency"] = int(groups[key]["frequency"]) + 1
    examples = groups[key]["examples"]
    if isinstance(examples, list) and len(examples) < 3:
        examples.append({"file": file_name, "context": context})


def find_coverage_gaps(
    candidates: Iterable[tuple[str, int, int, str, str]],
    matcher: RegexLabMatcher,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Tìm các cụm ứng viên chưa được matcher phủ đầy đủ."""

    partial: dict[str, dict[str, object]] = {}
    no_match: dict[str, dict[str, object]] = {}

    for phrase, start, end, file_name, text in candidates:
        match = best_match_for_phrase(phrase, matcher)
        context = context_for(text, start, end)
        if match is None:
            gap_bucket(no_match, phrase, file_name, context)
            continue

        matched_text, matched_len = match
        if matched_len < len(phrase.strip()):
            gap_bucket(partial, phrase, file_name, context, matched_text)

    def sorted_groups(groups: dict[str, dict[str, object]]) -> list[dict[str, object]]:
        return sorted(groups.values(), key=lambda item: (-int(item["frequency"]), normalize_surface(str(item["full_phrase"]))))

    return sorted_groups(partial), sorted_groups(no_match)


def collect_candidates(input_path: Path) -> list[tuple[str, int, int, str, str]]:
    """Gom các cụm ứng viên hình ảnh từ một file hoặc cả thư mục."""

    files = [input_path] if input_path.is_file() else sorted(
        input_path.glob("*.txt"),
        key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
    )
    candidates: list[tuple[str, int, int, str, str]] = []
    for text_file in files:
        text = text_file.read_text(encoding="utf-8-sig")
        for phrase, start, end in extract_imaging_candidate_phrases(text):
            candidates.append((phrase, start, end, text_file.name, text))
    return candidates


def write_json(path: Path, payload: object) -> None:
    """Ghi JSON bằng UTF-8."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def run(dict_path: Path, input_path: Path, output_path: Path) -> dict[str, object]:
    """Chạy rà gap hình ảnh và ghi báo cáo."""

    matcher = build_lab_matcher(load_seed_dict(dict_path))
    candidates = collect_candidates(input_path)
    partial, no_match = find_coverage_gaps(candidates, matcher)
    report = {
        "total_candidate_phrases": len(candidates),
        "gaps_partial_match": partial,
        "gaps_no_match": no_match,
    }
    write_json(output_path, report)

    LOGGER.info("========== IMAGING GAP REPORT ==========")
    LOGGER.info("total_candidate_phrases: %s", len(candidates))
    LOGGER.info("gaps_partial_match: %s", len(partial))
    LOGGER.info("gaps_no_match: %s", len(no_match))
    LOGGER.info("output: %s", output_path)
    LOGGER.info("========== TOP 30 GAPS ==========")
    combined = [{**item, "gap_type": "partial"} for item in partial] + [
        {**item, "gap_type": "no_match"} for item in no_match
    ]
    for item in sorted(combined, key=lambda row: (-int(row["frequency"]), str(row["gap_type"]), normalize_surface(str(row["full_phrase"]))))[:30]:
        matched = f" | matched={item.get('matched_portion')}" if item.get("matched_portion") else ""
        LOGGER.info("%s\t%s\t%s%s", item["frequency"], item["gap_type"], item["full_phrase"], matched)
    return report


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""

    parser = argparse.ArgumentParser(description="Find imaging coverage gaps in lab dictionary.")
    parser.add_argument("--dict", required=True, type=Path, help="Path to lab_dict_merged.json.")
    parser.add_argument("--input", required=True, type=Path, help="Input .txt file or directory.")
    parser.add_argument("--output", required=True, type=Path, help="Output gap report JSON.")
    return parser.parse_args()


def main() -> None:
    """Điểm vào khi chạy từ dòng lệnh."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    run(args.dict, args.input, args.output)


if __name__ == "__main__":
    main()
