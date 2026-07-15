"""Suggest candidate lab-test dictionary additions for manual review."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from lab_extractor import build_lab_matcher, load_seed_dict


NUMBER_PATTERN = r"\d+(?:[.,]\d+)?"
CANDIDATE_RE = re.compile(
    rf"(?P<phrase>[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ0-9 /()._-]{{1,60}}?)"
    rf"\s+(?:(?:(?:tăng|giảm|nâng cao lên|giảm xuống)\s+)?là\s+|(?:nâng cao lên|giảm xuống)\s+)?"
    rf"{NUMBER_PATTERN}",
    re.IGNORECASE,
)


def clean_candidate(value: str) -> str:
    """Normalize a candidate phrase for counting."""

    value = re.sub(r"^[\s,;:.\-–()]+|[\s,;:.\-–()]+$", "", value)
    tokens = value.split()
    return " ".join(tokens[-5:]).lower()


def suggest_candidates(dict_path: Path, input_dir: Path, limit: int) -> list[tuple[str, int, list[str]]]:
    """Find unmatched phrases before numeric result patterns."""

    matcher = build_lab_matcher(load_seed_dict(dict_path))
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)

    for text_file in sorted(input_dir.glob("*.txt")):
        text = text_file.read_text(encoding="utf-8-sig")
        known_spans = [(match.start, match.end) for match in matcher.find(text)]
        for match in CANDIDATE_RE.finditer(text):
            phrase_start, phrase_end = match.start("phrase"), match.end("phrase")
            if any(phrase_start < end and start < phrase_end for start, end in known_spans):
                continue
            candidate = clean_candidate(match.group("phrase"))
            if len(candidate) < 2:
                continue
            counts[candidate] += 1
            if len(examples[candidate]) < 3:
                snippet_start = max(0, match.start() - 60)
                snippet_end = min(len(text), match.end() + 60)
                snippet = re.sub(r"\s+", " ", text[snippet_start:snippet_end]).strip()
                examples[candidate].append(f"{text_file.name}: {snippet}")

    return [(candidate, count, examples[candidate]) for candidate, count in counts.most_common(limit)]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Suggest lab dictionary candidates for manual review.")
    parser.add_argument("--dict", required=True, type=Path, help="Path to lab_test_seed.json.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory with .txt files.")
    parser.add_argument("--limit", default=50, type=int, help="Maximum candidates to print.")
    return parser.parse_args()


def main() -> None:
    """Run candidate suggestion."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    suggestions = suggest_candidates(args.dict, args.input_dir, args.limit)
    print("========== LAB DICT CANDIDATES ==========")
    for candidate, count, examples in suggestions:
        print(f"\n{candidate}\tcount={count}")
        for example in examples:
            print(f"  - {example}")


if __name__ == "__main__":
    main()
