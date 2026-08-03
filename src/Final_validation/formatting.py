"""Convert linked pipeline output to the competition submission schema.

Competition ``position`` values are character offsets into the untouched raw
test input. The assertion pipeline's separate ``assertion_position`` belongs
to normalized text and must never be submitted as ``position``.
"""

from __future__ import annotations

import argparse
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
JSON_FOLDER = PROJECT_ROOT / "data" / "new_ner_output"
TEST_FOLDER = PROJECT_ROOT / "data" / "new_test_input"
FINAL_JSON_FOLDER = PROJECT_ROOT / "data" / "output_merged_validation"
SUBMISSION_ZIP = PROJECT_ROOT / "data" / "output.zip"

ALLOWED_TYPES = {
    "TRIỆU_CHỨNG",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
    "CHẨN_ĐOÁN",
    "THUỐC",
}
ASSERTION_TYPES = {"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"}
LINKABLE_TYPES = {"CHẨN_ĐOÁN", "THUỐC"}
ALLOWED_ASSERTIONS = {"isNegated", "isFamily", "isHistorical"}
UNCERTAIN_ASSERTION = "isUncertain"


def _normalise_for_position_lookup(text: str) -> str:
    """Reproduce the pipeline's line-based normalization in memory."""

    normalized_lines: list[str] = []
    bullet_pattern = re.compile(
        r"^\s*(?:[-–—*•+]+|\(?\d+[.)]|\(?[a-zA-Z][.)])\s*"
    )

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = bullet_pattern.sub("", line)
        line = re.sub(r"[\s.!?;:,]+$", "", line).strip()
        if line:
            normalized_lines.append(line)

    normalized_text = ". ".join(normalized_lines)
    if normalized_text:
        normalized_text += "."
    return re.sub(r"\s+", " ", normalized_text).strip()


def _whitespace_tolerant_pattern(text: str) -> re.Pattern[str]:
    """Build a literal pattern while allowing source whitespace to vary."""

    pieces = re.split(r"(\s+)", text)
    pattern = "".join(r"\s+" if piece.isspace() else re.escape(piece) for piece in pieces)
    return re.compile(f"(?=({pattern}))")


def _matches(text: str, source_text: str) -> list[tuple[int, int]]:
    """Return all (including overlapping) whitespace-tolerant occurrences."""

    if not isinstance(text, str) or not text:
        raise ValueError("Entity text must be a non-empty string")
    return [match.span(1) for match in _whitespace_tolerant_pattern(text).finditer(source_text)]


def calculate_position(
    text: str,
    source_text: str,
    *,
    occurrence: int = 0,
) -> list[int]:
    """Calculate ``text``'s character span in an original test document.

    ``occurrence`` is zero-based. Whitespace is matched tolerantly because the
    NER pipeline collapses repeated whitespace, whereas the submitted offsets
    must point into the untouched input document. The returned end offset is
    exclusive, as required by the competition format.
    """

    matches = _matches(text, source_text)
    if occurrence < 0:
        raise ValueError(f"Occurrence must be non-negative, got {occurrence}")
    if occurrence >= len(matches):
        raise ValueError(
            f"Could not find occurrence {occurrence} of {text!r}; "
            f"only {len(matches)} occurrence(s) exist"
        )
    return list(matches[occurrence])


def _occurrence_in_normalized_text(
    text: str,
    normalized_text: str,
    normalized_position: Any,
) -> int:
    """Find which occurrence an entity position identifies after normalization."""

    if (
        not isinstance(normalized_position, (list, tuple))
        or len(normalized_position) != 2
        or not all(isinstance(value, int) and not isinstance(value, bool) for value in normalized_position)
    ):
        raise ValueError(f"Invalid normalized position: {normalized_position!r}")

    expected = (normalized_position[0], normalized_position[1])
    matches = _matches(text, normalized_text)
    try:
        return matches.index(expected)
    except ValueError as exc:
        actual = normalized_text[expected[0] : expected[1]]
        raise ValueError(
            f"Normalized position {list(expected)} does not identify {text!r}; "
            f"it identifies {actual!r}"
        ) from exc


def _format_assertions(entity: dict[str, Any]) -> list[str]:
    """Return one valid assertion, treating null/uncertain as empty."""

    assertion_result = entity.get("assertion_result")
    if isinstance(assertion_result, dict):
        primary = assertion_result.get("tag")
    elif assertion_result is None:
        primary = None
    else:
        raise ValueError(f"Invalid assertion_result: {assertion_result!r}")

    # A present rule-based result is authoritative. Per submission policy,
    # both null and isUncertain become an empty assertion list.
    if isinstance(assertion_result, dict) and primary in (None, UNCERTAIN_ASSERTION):
        return []

    selected = primary if primary is not None else entity.get("qwen_assertion")
    if selected in (None, UNCERTAIN_ASSERTION):
        return []
    if selected not in ALLOWED_ASSERTIONS:
        raise ValueError(f"Unsupported assertion tag: {selected!r}")
    return [selected]


def _format_candidates(entity: dict[str, Any]) -> list[str]:
    """Return the selected candidate code as a one-element string list."""

    candidate = entity.get("candidate")
    if candidate is None:
        return []
    if not isinstance(candidate, dict) or candidate.get("code") is None:
        raise ValueError(f"Invalid candidate: {candidate!r}")
    return [str(candidate["code"])]


def _comparable_entity_text(text: str) -> str:
    """Normalize harmless raw/normalized differences for span validation."""
    return " ".join(text.split()).casefold().rstrip(" .?!;:,")


def _numeric_json_sort_key(path: Path) -> tuple[int, int | str]:
    if path.stem.isdigit():
        return (0, int(path.stem))
    return (1, path.stem.casefold())


def _read_entity_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    if not all(isinstance(entity, dict) for entity in data):
        raise ValueError(f"Every item in {path} must be a JSON object")
    return data


def _format_file(entities: Iterable[dict[str, Any]], source_text: str) -> list[dict[str, Any]]:
    normalized_text = _normalise_for_position_lookup(source_text)
    formatted: list[dict[str, Any]] = []

    for index, entity in enumerate(entities):
        try:
            entity_type = entity["type"]
            if entity_type not in ALLOWED_TYPES:
                raise ValueError(f"Unsupported entity type: {entity_type!r}")

            position = entity.get("position")
            if (
                not isinstance(position, (list, tuple))
                or len(position) != 2
                or not all(isinstance(value, int) and not isinstance(value, bool) for value in position)
            ):
                raise ValueError(f"Invalid raw position: {position!r}")
            start, end = position
            if start < 0 or end <= start or end > len(source_text):
                raise ValueError(
                    f"Raw position {[start, end]} is outside document length {len(source_text)}"
                )

            recorded_text = entity.get("text")
            if not isinstance(recorded_text, str) or not recorded_text:
                raise ValueError(f"Invalid entity text: {recorded_text!r}")

            raw_text = source_text[start:end]
            if _comparable_entity_text(recorded_text) == _comparable_entity_text(raw_text):
                output_position = [start, end]
            else:
                # Backward compatibility for linked files generated before
                # ``position`` was switched to raw offsets. Use the normalized
                # assertion span to identify the occurrence, then map it back.
                normalized_position = entity.get("assertion_position", position)
                occurrence = _occurrence_in_normalized_text(
                    recorded_text,
                    normalized_text,
                    normalized_position,
                )
                output_position = calculate_position(
                    recorded_text,
                    source_text,
                    occurrence=occurrence,
                )

            # Submission text must exactly match the submitted raw span.
            start, end = output_position
            text = source_text[start:end]
            output: dict[str, Any] = {
                "text": text,
                "type": entity_type,
            }
            if entity_type in LINKABLE_TYPES:
                output["candidates"] = _format_candidates(entity)
            if entity_type in ASSERTION_TYPES:
                output["assertions"] = _format_assertions(entity)
            output["position"] = output_position
            formatted.append(output)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"entity[{index}]: {exc}") from exc

    return sorted(formatted, key=lambda entity: tuple(entity["position"]))


def final_formatting(
    json_folder: str | Path = JSON_FOLDER,
    output_folder: str | Path = FINAL_JSON_FOLDER,
    test_folder: str | Path = TEST_FOLDER,
) -> list[Path]:
    """Format every sorted JSON file and return the paths that were written."""

    input_dir = Path(json_folder)
    output_dir = Path(output_folder)
    text_dir = Path(test_folder)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"JSON folder does not exist: {input_dir}")
    if not text_dir.is_dir():
        raise FileNotFoundError(f"Test folder does not exist: {text_dir}")

    json_files = sorted(input_dir.glob("*.json"), key=_numeric_json_sort_key)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for json_path in json_files:
        text_path = text_dir / f"{json_path.stem}.txt"
        if not text_path.is_file():
            raise FileNotFoundError(f"Missing counterpart for {json_path.name}: {text_path}")

        entities = _read_entity_list(json_path)
        source_text = text_path.read_text(encoding="utf-8-sig")
        formatted = _format_file(entities, source_text)

        output_path = output_dir / json_path.name
        with output_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(formatted, file, ensure_ascii=False, indent=2)
            file.write("\n")
        written.append(output_path)

    return written


def create_submission_zip(
    json_files: Iterable[str | Path],
    zip_path: str | Path = SUBMISSION_ZIP,
) -> Path:
    """Create the required ZIP layout: ``output/1.json`` through ``100.json``."""

    paths = sorted((Path(path) for path in json_files), key=_numeric_json_sort_key)
    if not paths:
        raise ValueError("Cannot create a submission ZIP without JSON files")

    destination = Path(zip_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Submission JSON does not exist: {path}")
            archive.write(path, arcname=f"output/{path.name}")
    return destination


# Preserve the original misspelled API name for any existing callers.
final_formating = final_formatting


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-folder", type=Path, default=JSON_FOLDER)
    parser.add_argument("--test-folder", type=Path, default=TEST_FOLDER)
    parser.add_argument("--output-folder", type=Path, default=FINAL_JSON_FOLDER)
    parser.add_argument("--zip-path", type=Path, default=SUBMISSION_ZIP)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    written = final_formatting(
        json_folder=args.json_folder,
        output_folder=args.output_folder,
        test_folder=args.test_folder,
    )
    zip_path = create_submission_zip(written, args.zip_path)
    print(f"Created {len(written)} submission files in: {args.output_folder}")
    print(f"Created submission archive: {zip_path}")


if __name__ == "__main__":
    main()
