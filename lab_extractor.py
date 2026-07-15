"""Rule-based extraction for TÊN_XÉT_NGHIỆM and KẾT_QUẢ_XÉT_NGHIỆM."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


LOGGER = logging.getLogger(__name__)

TEST_NAME = "TÊN_XÉT_NGHIỆM"
TEST_RESULT = "KẾT_QUẢ_XÉT_NGHIỆM"
NUMBER_PATTERN = r"\d+(?:[.,]\d+)?"
RESULT_NUMBER_RE = re.compile(NUMBER_PATTERN)
BP_RE = re.compile(r"\s*[:=]?\s*(?P<value>\d+\s*/\s*\d+)")
SPO2_RE = re.compile(
    rf"\s*[:=]?\s*(?P<value>{NUMBER_PATTERN}(?:-{NUMBER_PATTERN})?)(?:\s*(?:%|RA))?\b",
    re.IGNORECASE,
)
CONNECTOR_RE = re.compile(
    rf"\s*(?:[:=]\s*|(?:(?:tăng|giảm|nâng cao lên|giảm xuống)\s+)?là\s+|(?:nâng cao lên|giảm xuống)\s+)"
    rf"(?P<value>{NUMBER_PATTERN})",
    re.IGNORECASE,
)
ADJACENT_RE = re.compile(rf"\s+(?P<value>{NUMBER_PATTERN})")
VITAL_DESCRIPTOR_RESULT_RE = re.compile(
    rf"\s*(?:nhanh|chậm|không\s+đều|loạn|gấp|khó)\s+(?P<value>{NUMBER_PATTERN})",
    re.IGNORECASE,
)
DIMENSION_AFTER_NUMBER_RE = re.compile(
    rf"\s*x\s*{NUMBER_PATTERN}(?:\s*x\s*{NUMBER_PATTERN})?\s*(?:cm|mm)\b",
    re.IGNORECASE,
)
VITAL_BLOB_RE = re.compile(r"\bVS\s*\d[\d.\s/]*\d\s*RA\b|\bVS\s*\d[\d.\s/]{8,}\b", re.IGNORECASE)
VIETNAMESE_COMPOUND_EXCLUDE = {
    "mạch": {
        "prefix_exclude": ("tim ", "động ", "tĩnh ", "bác sĩ tim "),
        "suffix_exclude": (" vành", " máu", " cửa", " ngoại vi", " chủ"),
    },
    "huyết áp": {
        "prefix_exclude": ("tăng ", "hạ ", "tụt ", "cao "),
        "suffix_exclude": (),
    },
    "nhịp tim": {
        "prefix_exclude": (),
        "suffix_exclude": (" nhanh", " chậm", " không đều", " loạn"),
    },
    "nhịp thở": {
        "prefix_exclude": (),
        "suffix_exclude": (" nhanh", " chậm", " gấp", " khó"),
    },
}


@dataclass(frozen=True)
class LabSeedEntry:
    """One lab/vital/imaging dictionary entry."""

    canonical_id: str
    category: str
    surface_forms: tuple[str, ...]
    surface_confidences: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class LabMatch:
    """One matched lab-test/vital-sign/imaging name span."""

    start: int
    end: int
    text: str
    normalized_form: str
    canonical_id: str
    category: str
    match_confidence: str


@dataclass
class ExtractionStats:
    """Counters collected during lab extraction."""

    dimension_excludes: int = 0
    unparsed_vital_blobs: int = 0
    compound_excludes: int = 0
    rejected_vital_no_number: int = 0


class RegexLabMatcher:
    """Dependency-free regex matcher for curated lab/vital/imaging surface forms.

    The class mirrors the regex-chunk approach used in ``drug_extractor.py``:
    surface forms are escaped, sorted by length, and matched case-insensitively
    with word-boundary guards. The implementation is simple to debug and can be
    swapped for Aho-Corasick later without changing the extraction API.
    """

    def __init__(self, entries: Iterable[LabSeedEntry], chunk_size: int = 1000) -> None:
        """Build regex chunks and metadata lookup from seed entries."""

        self.form_to_entry: dict[str, LabSeedEntry] = {}
        self.form_to_confidence: dict[str, str] = {}
        for entry in entries:
            for surface_form in entry.surface_forms:
                normalized = normalize_surface(surface_form)
                if normalized:
                    self.form_to_entry[normalized] = entry
                    self.form_to_confidence[normalized] = entry.surface_confidences.get(normalized, "high")

        forms = sorted(self.form_to_entry, key=lambda value: (-len(value), value))
        self.patterns = [
            re.compile(
                r"(?<!\w)(?P<lab>" + "|".join(surface_to_regex(form) for form in chunk) + r")(?!\w)",
                re.IGNORECASE,
            )
            for chunk in chunked(forms, chunk_size)
            if chunk
        ]

    def find(self, text: str) -> list[LabMatch]:
        """Find non-overlapping lab/vital matches in text."""

        raw_matches: list[LabMatch] = []
        for pattern in self.patterns:
            for match in pattern.finditer(text):
                normalized = normalize_surface(match.group("lab"))
                entry = self.form_to_entry.get(normalized)
                if entry is None:
                    continue
                raw_matches.append(
                    LabMatch(
                        start=match.start(),
                        end=match.end(),
                        text=text[match.start() : match.end()],
                        normalized_form=normalized,
                        canonical_id=entry.canonical_id,
                        category=entry.category,
                        match_confidence=self.form_to_confidence.get(normalized, "high"),
                    )
                )

        expanded = expand_parenthetical_abbreviations(text, raw_matches, self.form_to_entry, self.form_to_confidence)
        return select_longest_non_overlapping(expanded)


def normalize_surface(value: str) -> str:
    """Normalize dictionary surface forms and matched text."""

    return re.sub(r"\s+", " ", value.strip().lower())


def surface_to_regex(surface: str) -> str:
    """Convert a normalized surface form into a whitespace-tolerant regex."""

    return r"\s+".join(re.escape(part) for part in surface.split())


def chunked(values: list[str], size: int) -> Iterable[list[str]]:
    """Yield chunks of a list."""

    for start in range(0, len(values), size):
        yield values[start : start + size]


def select_longest_non_overlapping(matches: Iterable[LabMatch]) -> list[LabMatch]:
    """Keep the longest span when multiple lab-name matches overlap."""

    selected: list[LabMatch] = []
    for match in sorted(matches, key=lambda item: (-(item.end - item.start), item.start)):
        if any(match.start < kept.end and kept.start < match.end for kept in selected):
            continue
        selected.append(match)
    return sorted(selected, key=lambda item: (item.start, item.end))


def expand_parenthetical_abbreviations(
    text: str,
    matches: list[LabMatch],
    form_to_entry: dict[str, LabSeedEntry],
    form_to_confidence: dict[str, str],
) -> list[LabMatch]:
    """Merge full name plus immediate parenthetical abbreviation when equivalent."""

    expanded: list[LabMatch] = []
    for match in matches:
        paren_match = re.match(r"\s*\((?P<abbr>[^()]{1,20})\)", text[match.end : match.end + 30])
        if paren_match is None:
            expanded.append(match)
            continue

        abbr = normalize_surface(paren_match.group("abbr"))
        abbr_entry = form_to_entry.get(abbr)
        if abbr_entry is None or abbr_entry.canonical_id != match.canonical_id:
            expanded.append(match)
            continue

        new_end = match.end + paren_match.end()
        expanded.append(
            LabMatch(
                start=match.start,
                end=new_end,
                text=text[match.start:new_end],
                normalized_form=match.normalized_form,
                canonical_id=match.canonical_id,
                category=match.category,
                match_confidence=form_to_confidence.get(match.normalized_form, match.match_confidence),
            )
        )

    return expanded


def infer_surface_confidences(surface_forms: tuple[str, ...]) -> dict[str, str]:
    """Infer debug confidence for surface forms when per-form metadata is absent.

    The merged dictionary currently stores only the final surface strings.
    Therefore this function marks every form as ``high`` by default and marks
    a form as ``medium`` only when it appears as the content of parentheses in
    another surface form from the same canonical entry, which approximates the
    lower-confidence Rule 3 variants generated from parenthetical text.
    """

    confidences = {normalize_surface(form): "high" for form in surface_forms if normalize_surface(form)}
    normalized_forms = set(confidences)
    for form in surface_forms:
        for paren_match in re.finditer(r"\(([^()]{2,80})\)", form):
            inner = normalize_surface(paren_match.group(1))
            if inner in normalized_forms:
                confidences[inner] = "medium"
    return confidences


def load_seed_dict(path: Path) -> list[LabSeedEntry]:
    """Load a lab/vital/imaging dictionary from UTF-8 JSON.

    Both the old curated seed schema and the merged dictionary schema are
    supported. The merged dictionary may include category ``imaging``; imaging
    entries are handled like lab tests, meaning the name can be emitted without
    a numeric result.
    """

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    entries: list[LabSeedEntry] = []
    for index, item in enumerate(payload):
        try:
            surface_forms = tuple(str(form) for form in item["surface_forms"])
            raw_confidences = item.get("surface_form_confidence", {})
            if isinstance(raw_confidences, dict):
                confidences = {
                    normalize_surface(str(form)): str(confidence)
                    for form, confidence in raw_confidences.items()
                    if str(confidence) in {"high", "medium", "low"}
                }
            else:
                confidences = {}
            if not confidences:
                confidences = infer_surface_confidences(surface_forms)

            entries.append(
                LabSeedEntry(
                    canonical_id=str(item["canonical_id"]),
                    category=str(item["category"]),
                    surface_forms=surface_forms,
                    surface_confidences=confidences,
                )
            )
        except KeyError as error:
            raise ValueError(f"{path}: seed item {index} missing key {error}") from error

    return entries


def build_lab_matcher(seed_dict: list[LabSeedEntry]) -> RegexLabMatcher:
    """Build a case-insensitive word-boundary matcher over all surface forms."""

    return RegexLabMatcher(seed_dict)


def is_dimension_measurement(text: str, number_end_pos: int) -> bool:
    """Return whether a number is followed by image dimension pattern."""

    return DIMENSION_AFTER_NUMBER_RE.match(text, number_end_pos) is not None


def has_number_soon(text: str, start: int, window_size: int = 15) -> bool:
    """Return whether a numeric value appears soon after a position."""

    return RESULT_NUMBER_RE.search(text[start : start + window_size]) is not None


def is_compound_exclusion(text: str, match_start: int, match_end: int, surface_form: str) -> bool:
    """Return whether a Vietnamese vital match is part of a non-vital compound.

    Short Vietnamese vital-sign names are valid words inside other medical
    compounds. For example, ``mạch`` is a vital sign in ``Mạch: 88`` but not in
    ``động mạch`` or ``tim mạch``. Suffix exclusions for ``nhịp tim`` and
    ``nhịp thở`` are ignored when a concrete number follows soon after the
    descriptor, such as ``nhịp tim nhanh 130``.
    """

    normalized_surface = normalize_surface(surface_form)
    rules = VIETNAMESE_COMPOUND_EXCLUDE.get(normalized_surface)
    if rules is None:
        return False

    before = text[max(0, match_start - 25) : match_start].lower()
    for prefix in rules["prefix_exclude"]:
        normalized_prefix = normalize_surface(prefix)
        if re.search(surface_to_regex(normalized_prefix) + r"\s*$", before, re.IGNORECASE):
            return True

    after = text[match_end : match_end + 35].lower()
    for suffix in rules["suffix_exclude"]:
        normalized_suffix = normalize_surface(suffix)
        suffix_match = re.match(r"\s*" + surface_to_regex(normalized_suffix), after, re.IGNORECASE)
        if suffix_match is None:
            continue
        if normalized_surface != "mạch" and has_number_soon(after, suffix_match.end(), 15):
            continue
        return True

    return False


def make_entity(text: str, entity_type: str, start: int, end: int) -> dict[str, object]:
    """Create one output entity with the required schema."""

    return {"text": text[start:end], "type": entity_type, "position": [start, end]}


def find_result_for_match(
    text: str,
    match: LabMatch,
    stats: ExtractionStats | None = None,
) -> tuple[int, int] | None:
    """Find a numeric result span for a matched test name, if present."""

    if match.canonical_id == "BLOOD_PRESSURE":
        window = text[match.end : match.end + 15]
        result_match = BP_RE.match(window)
        if result_match is not None:
            return match.end + result_match.start("value"), match.end + result_match.end("value")

    if match.canonical_id == "SPO2":
        window = text[match.end : match.end + 20]
        result_match = SPO2_RE.match(window)
        if result_match is not None:
            return match.end + result_match.start("value"), match.end + result_match.end("value")

    if match.canonical_id in {"HEART_RATE", "RESPIRATORY_RATE"}:
        descriptor_window = text[match.end : match.end + 35]
        descriptor_match = VITAL_DESCRIPTOR_RESULT_RE.match(descriptor_window)
        if descriptor_match is not None:
            return (
                match.end + descriptor_match.start("value"),
                match.end + descriptor_match.end("value"),
            )

    connector_window = text[match.end : match.end + 30]
    connector_match = CONNECTOR_RE.match(connector_window)
    if connector_match is not None:
        start = match.end + connector_match.start("value")
        end = match.end + connector_match.end("value")
        if is_dimension_measurement(text, end):
            if stats is not None:
                stats.dimension_excludes += 1
            LOGGER.info("Excluded dimension-like result: %s", text[start:end])
            return None
        return start, end

    adjacent_window = text[match.end : match.end + 20]
    adjacent_match = ADJACENT_RE.match(adjacent_window)
    if adjacent_match is not None:
        start = match.end + adjacent_match.start("value")
        end = match.end + adjacent_match.end("value")
        if is_dimension_measurement(text, end):
            if stats is not None:
                stats.dimension_excludes += 1
            LOGGER.info("Excluded dimension-like result: %s", text[start:end])
            return None
        return start, end

    return None


def detect_unparseable_vital_blob(text: str) -> list[dict[str, object]]:
    """Detect unparseable compact vital-sign blobs without producing entities."""

    blobs: list[dict[str, object]] = []
    for match in VITAL_BLOB_RE.finditer(text):
        blobs.append(
            {
                "raw_text": match.group(0),
                "position": [match.start(), match.end()],
                "reason": "unparseable compact vital signs blob",
            }
        )
    return blobs


def extract_lab_pairs(
    text: str,
    matcher: RegexLabMatcher,
    stats: ExtractionStats | None = None,
    rejected_vitals: list[dict[str, object]] | None = None,
    debug_matches: list[dict[str, object]] | None = None,
    source_name: str | None = None,
) -> list[dict[str, object]]:
    """Extract lab/vital/imaging test-name entities and numeric result entities.

    Lab-test names may be emitted without a numeric result, because mentions
    like ``công thức máu bình thường`` are still useful. Vital-sign names are
    emitted only when a nearby numeric value is found; standalone Vietnamese
    vital words are often false positives inside disease or anatomy phrases.
    Imaging names follow lab-test behavior and can be emitted without a
    numeric result.
    """

    if stats is None:
        stats = ExtractionStats()

    entities: list[dict[str, object]] = []
    matches = matcher.find(text)
    for match in matches:
        if match.category == "vital_sign" and is_compound_exclusion(text, match.start, match.end, match.text):
            stats.compound_excludes += 1
            continue

        result_span = find_result_for_match(text, match, stats)
        if match.category == "vital_sign" and result_span is None:
            stats.rejected_vital_no_number += 1
            if rejected_vitals is not None:
                rejected_vitals.append(
                    {
                        "file": source_name,
                        "text": match.text,
                        "canonical_id": match.canonical_id,
                        "position": [match.start, match.end],
                        "reason": "vital_sign without nearby numeric result",
                    }
                )
            continue

        entities.append(make_entity(text, TEST_NAME, match.start, match.end))
        if debug_matches is not None:
            debug_matches.append(
                {
                    "file": source_name,
                    "text": match.text,
                    "type": TEST_NAME,
                    "position": [match.start, match.end],
                    "canonical_id": match.canonical_id,
                    "category": match.category,
                    "match_confidence": match.match_confidence,
                }
            )
        if result_span is not None:
            entities.append(make_entity(text, TEST_RESULT, result_span[0], result_span[1]))

    return sorted(entities, key=lambda entity: (entity["position"][0], entity["position"][1], entity["type"]))


def write_json(path: Path, payload: object) -> None:
    """Write JSON with UTF-8 encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def extract_file(
    input_path: Path,
    output_path: Path,
    matcher: RegexLabMatcher,
    stats: ExtractionStats,
    unparsed_logs: list[dict[str, object]],
    rejected_vitals: list[dict[str, object]],
    debug_matches: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Extract lab/vital entities from one text file and write JSON output."""

    text = input_path.read_text(encoding="utf-8-sig")
    blobs = detect_unparseable_vital_blob(text)
    for blob in blobs:
        unparsed_logs.append({"file": input_path.name, **blob})
    stats.unparsed_vital_blobs += len(blobs)

    entities = extract_lab_pairs(text, matcher, stats, rejected_vitals, debug_matches, input_path.name)
    write_json(output_path, entities)
    return entities


def run_batch(dict_path: Path, input_path: Path, output_path: Path) -> None:
    """Run extraction for one file or all .txt files in a directory."""

    stats = ExtractionStats()
    matcher = build_lab_matcher(load_seed_dict(dict_path))
    unparsed_logs: list[dict[str, object]] = []
    rejected_vitals: list[dict[str, object]] = []
    debug_matches: list[dict[str, object]] = []
    outputs: dict[str, list[dict[str, object]]] = {}

    if input_path.is_file():
        target = output_path
        outputs[input_path.name] = extract_file(input_path, target, matcher, stats, unparsed_logs, rejected_vitals, debug_matches)
    elif input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        for text_file in sorted(input_path.glob("*.txt"), key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem):
            target = output_path / f"{text_file.stem}.json"
            outputs[text_file.name] = extract_file(
                text_file,
                target,
                matcher,
                stats,
                unparsed_logs,
                rejected_vitals,
                debug_matches,
            )
            LOGGER.info("Wrote %s", target)
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    log_path = output_path / "unparsed_vitals_log.json" if output_path.is_dir() else output_path.parent / "unparsed_vitals_log.json"
    write_json(log_path, unparsed_logs)
    rejected_path = (
        output_path / "rejected_vital_no_number.json"
        if output_path.is_dir()
        else output_path.parent / "rejected_vital_no_number.json"
    )
    write_json(rejected_path, rejected_vitals)
    debug_path = output_path / "match_debug_log.json" if output_path.is_dir() else output_path.parent / "match_debug_log.json"
    write_json(debug_path, debug_matches)
    log_summary(outputs, stats)


def log_summary(outputs: dict[str, list[dict[str, object]]], stats: ExtractionStats) -> None:
    """Print batch extraction summary."""

    entity_counts: Counter[str] = Counter()
    for entities in outputs.values():
        entity_counts.update(str(entity["type"]) for entity in entities)

    files_with_entities = sum(1 for entities in outputs.values() if entities)
    empty_files = len(outputs) - files_with_entities

    LOGGER.info("========== LAB EXTRACTION SUMMARY ==========")
    LOGGER.info("files: %s", len(outputs))
    LOGGER.info("%s: %s", TEST_NAME, entity_counts.get(TEST_NAME, 0))
    LOGGER.info("%s: %s", TEST_RESULT, entity_counts.get(TEST_RESULT, 0))
    LOGGER.info("files_with_entities: %s", files_with_entities)
    LOGGER.info("empty_files: %s", empty_files)
    LOGGER.info("dimension_excludes: %s", stats.dimension_excludes)
    LOGGER.info("unparsed_vital_blobs: %s", stats.unparsed_vital_blobs)
    LOGGER.info("compound_vital_excludes: %s", stats.compound_excludes)
    LOGGER.info("rejected_vital_no_number: %s", stats.rejected_vital_no_number)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Extract lab/vital entities from clinical text.")
    parser.add_argument("--dict", required=True, type=Path, help="Path to lab_test_seed.json.")
    parser.add_argument("--input", required=True, type=Path, help="Input .txt file or directory.")
    parser.add_argument("--output", required=True, type=Path, help="Output .json file or directory.")
    return parser.parse_args()


def main() -> None:
    """Run the lab extractor CLI."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    run_batch(args.dict, args.input, args.output)


if __name__ == "__main__":
    main()
