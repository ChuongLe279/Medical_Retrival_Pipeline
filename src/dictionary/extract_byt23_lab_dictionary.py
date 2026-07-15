"""Extract lab/imaging dictionary entries from BYT 23/2024 appendix PDF.

The source PDF is Phu luc 01 of Thong tu 23/2024/TT-BYT. It contains a
four-column table: STT | Ma ky thuat | Ten chuong | Ten ky thuat. This script
parses that table, keeps diagnostic/lab/imaging techniques, removes treatment
procedures, deduplicates by canonical technique name, and writes a JSON
dictionary suitable as a canonical source for later surface-form generation.

Install dependency before running:
    pip install pdfplumber

Example:
    python -m src.dictionary.extract_byt23_lab_dictionary \
      --pdf data/raw/23-byt-kem.pdf \
      --output data/dict/lab_imaging_from_byt23.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


LOGGER = logging.getLogger(__name__)

SOURCE_NAME = "Thong_tu_23_2024_TT_BYT_PhuLuc01"

CHUONG_WHITELIST_FULL = (
    "hoa sinh",
    "huyet hoc",
    "vi sinh",
    "giai phau benh",
    "di truyen sinh hoc phan tu",
    "chan doan hinh anh",
    "dien quang",
    "tham do chuc nang",
)

CHUONG_BLACKLIST_KEYWORDS = (
    "y hoc co truyen",
    "phuc hoi chuc nang",
    "cham",
    "cay chi",
)

WHITELIST_PREFIX = (
    "dinh luong",
    "dinh tinh",
    "xet nghiem",
    "sieu am",
    "chup x quang",
    "chup ct",
    "chup cat lop",
    "chup cong huong tu",
    "chup mri",
    "chup dong mach",
    "dien tim",
    "ghi dien tim",
    "dien nao",
    "ghi dien nao",
    "dien co",
    "ghi dien co",
    "do",
    "test",
    "nghiem phap",
)

BLACKLIST_KEYWORDS = (
    "phau thuat",
    "dieu tri",
    "can thiep",
    "dat",
    "dan luu",
    "loc mau",
    "tiem",
    "noi",
    "cat",
)

ENDOSCOPY_TREATMENT_KEYWORDS = (
    "dieu tri",
    "can thiep",
    "cat",
    "dat",
    "tiem",
    "noi",
    "dan luu",
    "gay dinh",
    "cam mau",
)


@dataclass(frozen=True)
class TechniqueRow:
    """One parsed row from the BYT appendix table."""

    stt: int
    ma_ky_thuat: str
    ten_chuong: str
    ten_ky_thuat: str


@dataclass(frozen=True)
class KeptRow:
    """One row kept by the filtering rules."""

    row: TechniqueRow
    layer: str


@dataclass
class FilterStats:
    """Counters collected during filtering."""

    kept_layer1: int = 0
    kept_layer2: int = 0
    excluded_chapter_blacklist: int = 0
    excluded_treatment: int = 0


def normalize_text(value: str) -> str:
    """Lowercase, strip accents, and normalize whitespace/punctuation."""

    decomposed = unicodedata.normalize("NFD", value)
    without_accents = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    normalized = without_accents.lower().replace("đ", "d")
    normalized = re.sub(r"[-‐‑‒–—]+", " ", normalized)
    normalized = re.sub(r"[()\\[\\],.;:/]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def clean_cell(value: object) -> str:
    """Normalize a PDF table cell into a single-line string."""

    if value is None:
        return ""
    text = str(value).replace("\r", "\n")
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_int(value: str) -> int | None:
    """Parse an STT cell, returning None for header/continuation rows."""

    text = clean_cell(value)
    if not re.fullmatch(r"\d+", text):
        return None
    return int(text)


def import_pdfplumber() -> object:
    """Import pdfplumber with a clear installation error."""

    try:
        import pdfplumber  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "Missing dependency 'pdfplumber'. Install it with: pip install pdfplumber"
        ) from error
    return pdfplumber


def extract_page_tables(page: object) -> list[list[list[object]]]:
    """Extract tables from a pdfplumber page using several robust settings."""

    settings_candidates = [
        {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "intersection_tolerance": 5,
            "text_x_tolerance": 2,
            "text_y_tolerance": 3,
        },
        {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "intersection_tolerance": 5,
            "text_x_tolerance": 2,
            "text_y_tolerance": 3,
            "min_words_vertical": 1,
            "min_words_horizontal": 1,
        },
    ]

    for settings in settings_candidates:
        tables = page.extract_tables(table_settings=settings)
        valid_tables = [table for table in tables if table and any(len(row) >= 4 for row in table)]
        if valid_tables:
            return valid_tables
    return []


def append_continuation(previous: TechniqueRow, cells: Sequence[str]) -> TechniqueRow:
    """Append a physical PDF continuation row to the previous logical row."""

    ten_chuong = previous.ten_chuong
    ten_ky_thuat = previous.ten_ky_thuat
    if len(cells) >= 3 and cells[2]:
        ten_chuong = f"{ten_chuong} {cells[2]}".strip()
    if len(cells) >= 4 and cells[3]:
        ten_ky_thuat = f"{ten_ky_thuat} {cells[3]}".strip()
    return TechniqueRow(previous.stt, previous.ma_ky_thuat, ten_chuong, ten_ky_thuat)


def table_rows_to_records(raw_rows: Iterable[Sequence[object]]) -> list[TechniqueRow]:
    """Convert raw PDF table rows to logical TechniqueRow records."""

    records: list[TechniqueRow] = []
    for raw_row in raw_rows:
        cells = [clean_cell(cell) for cell in raw_row[:4]]
        while len(cells) < 4:
            cells.append("")

        if not any(cells):
            continue
        normalized_row = " ".join(normalize_text(cell) for cell in cells)
        if "stt" in normalized_row and "ma ky thuat" in normalized_row:
            continue

        stt = parse_int(cells[0])
        if stt is None:
            if records:
                records[-1] = append_continuation(records[-1], cells)
            continue

        ma_ky_thuat = cells[1]
        ten_chuong = cells[2]
        ten_ky_thuat = cells[3]
        if not ma_ky_thuat or not ten_ky_thuat:
            if records:
                records[-1] = append_continuation(records[-1], cells)
            continue
        records.append(TechniqueRow(stt, ma_ky_thuat, ten_chuong, ten_ky_thuat))
    return records


def parse_pdf(pdf_path: Path) -> list[TechniqueRow]:
    """Parse the 4-column appendix PDF into structured rows."""

    pdfplumber = import_pdfplumber()
    all_rows: list[TechniqueRow] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            page_rows: list[Sequence[object]] = []
            for table in extract_page_tables(page):
                page_rows.extend(table)
            if not page_rows:
                LOGGER.warning("No table extracted from page %s", page_index)
                continue
            all_rows.extend(table_rows_to_records(page_rows))
    return all_rows


def is_chapter_blacklisted(ten_chuong: str) -> bool:
    """Return whether an entire chapter should be ignored."""

    normalized = normalize_text(ten_chuong)
    return any(keyword in normalized for keyword in CHUONG_BLACKLIST_KEYWORDS)


def is_full_whitelist_chapter(ten_chuong: str) -> bool:
    """Return whether a chapter is mostly diagnostic/lab/imaging."""

    normalized = normalize_text(ten_chuong)
    return any(keyword in normalized for keyword in CHUONG_WHITELIST_FULL)


def is_whitelist_prefix(ten_ky_thuat: str) -> bool:
    """Return whether a technique starts with a diagnostic/lab prefix."""

    normalized = normalize_text(ten_ky_thuat)
    return any(normalized.startswith(prefix) for prefix in WHITELIST_PREFIX)


def is_allowed_endoscopy(ten_ky_thuat: str) -> bool:
    """Return whether an endoscopy technique is diagnostic enough to keep."""

    normalized = normalize_text(ten_ky_thuat)
    if "noi soi" not in normalized:
        return False
    if "chan doan" in normalized:
        return True
    return not any(keyword in normalized for keyword in ENDOSCOPY_TREATMENT_KEYWORDS)


def has_treatment_blacklist(ten_ky_thuat: str) -> bool:
    """Return whether a technique appears to be treatment/procedure-oriented."""

    if is_allowed_endoscopy(ten_ky_thuat):
        return False
    normalized = normalize_text(ten_ky_thuat)
    return any(re.search(rf"(^|\s){re.escape(keyword)}(\s|$)", normalized) for keyword in BLACKLIST_KEYWORDS)


def filter_rows(rows: Iterable[TechniqueRow]) -> tuple[list[KeptRow], FilterStats]:
    """Filter parsed rows to lab/imaging/diagnostic technique candidates."""

    kept: list[KeptRow] = []
    stats = FilterStats()

    for row in rows:
        if is_chapter_blacklisted(row.ten_chuong):
            stats.excluded_chapter_blacklist += 1
            continue

        if is_allowed_endoscopy(row.ten_ky_thuat):
            kept.append(KeptRow(row, "layer2_prefix_keyword"))
            stats.kept_layer2 += 1
            continue

        if has_treatment_blacklist(row.ten_ky_thuat):
            stats.excluded_treatment += 1
            continue

        if is_full_whitelist_chapter(row.ten_chuong):
            kept.append(KeptRow(row, "layer1_chapter_whitelist"))
            stats.kept_layer1 += 1
            continue

        if is_whitelist_prefix(row.ten_ky_thuat):
            kept.append(KeptRow(row, "layer2_prefix_keyword"))
            stats.kept_layer2 += 1

    return kept, stats


def dedupe_rows(kept_rows: Iterable[KeptRow]) -> list[dict[str, object]]:
    """Deduplicate kept rows by normalized canonical technique name."""

    by_name: dict[str, dict[str, object]] = {}
    for kept in kept_rows:
        row = kept.row
        key = normalize_text(row.ten_ky_thuat)
        existing = by_name.get(key)
        if existing is None:
            by_name[key] = {
                "canonical_name": row.ten_ky_thuat,
                "chuong": row.ten_chuong,
                "ma_ky_thuat": [row.ma_ky_thuat],
                "source": SOURCE_NAME,
            }
            continue

        codes = existing["ma_ky_thuat"]
        if isinstance(codes, list) and row.ma_ky_thuat not in codes:
            codes.append(row.ma_ky_thuat)

    return sorted(by_name.values(), key=lambda item: normalize_text(str(item["canonical_name"])))


def write_json(path: Path, payload: object) -> None:
    """Write UTF-8 JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def log_unique_chapters(rows: Sequence[TechniqueRow]) -> None:
    """Log all unique chapter names with row counts for manual confirmation."""

    counts = Counter(row.ten_chuong for row in rows)
    LOGGER.info("========== UNIQUE CHAPTERS ==========")
    for chapter, count in sorted(counts.items(), key=lambda item: normalize_text(item[0])):
        LOGGER.info("%s\t%s", count, chapter)


def log_random_samples(entries: Sequence[dict[str, object]], sample_size: int, seed: int) -> None:
    """Log random output entries for quick manual review."""

    LOGGER.info("========== RANDOM SAMPLES ==========")
    if not entries:
        LOGGER.info("(no entries)")
        return
    rng = random.Random(seed)
    for index, item in enumerate(rng.sample(list(entries), min(sample_size, len(entries))), start=1):
        LOGGER.info(
            "%02d. [%s] %s | %s",
            index,
            ", ".join(str(code) for code in item["ma_ky_thuat"]),
            item["canonical_name"],
            item["chuong"],
        )


def run(pdf_path: Path, output_path: Path, sample_size: int, seed: int) -> list[dict[str, object]]:
    """Run PDF parsing, filtering, dedupe, output, and reporting."""

    rows = parse_pdf(pdf_path)
    kept_rows, stats = filter_rows(rows)
    entries = dedupe_rows(kept_rows)
    write_json(output_path, entries)

    log_unique_chapters(rows)
    LOGGER.info("========== BYT 23 LAB/IMAGING DICTIONARY ==========")
    LOGGER.info("Source PDF: %s", pdf_path)
    LOGGER.info("Output JSON: %s", output_path)
    LOGGER.info("Total raw rows parsed: %s", len(rows))
    LOGGER.info("Kept by layer 1 (chapter whitelist): %s", stats.kept_layer1)
    LOGGER.info("Kept by layer 2 (prefix/keyword): %s", stats.kept_layer2)
    LOGGER.info("Excluded by chapter blacklist: %s", stats.excluded_chapter_blacklist)
    LOGGER.info("Excluded by treatment blacklist: %s", stats.excluded_treatment)
    LOGGER.info("Final entries after dedupe: %s", len(entries))
    log_random_samples(entries, sample_size, seed)
    LOGGER.info(
        "Reminder: this is a canonical legal-source dictionary. Review random samples before using it in production."
    )
    return entries


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Extract lab/imaging dictionary from BYT 23 appendix PDF.")
    parser.add_argument("--pdf", type=Path, default=Path("data/raw/23-byt-kem.pdf"), help="Input BYT appendix PDF.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/dict/lab_imaging_from_byt23.json"),
        help="Output JSON dictionary path.",
    )
    parser.add_argument("--sample-size", type=int, default=30, help="Number of random output samples to print.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for review samples.")
    return parser.parse_args()


def main() -> None:
    """Run the script from CLI."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    run(args.pdf, args.output, args.sample_size, args.seed)


if __name__ == "__main__":
    main()
