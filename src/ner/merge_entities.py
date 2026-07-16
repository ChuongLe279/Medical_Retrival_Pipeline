"""Merge entity JSON outputs from NER, drug rules, and lab rules.

The batch processor uses the union of ``*.json`` basenames found in the three
source directories. If a basename is absent from all three sources, it is not
emitted because there is no source filename to infer from; this is intentional
for the "all sources missing" edge case.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def _configure_utf8_streams() -> None:
    """Prefer UTF-8 console output when the runtime allows reconfiguration."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


_configure_utf8_streams()

NER_SOURCE = "ner"
DRUG_SOURCE = "drug"
LAB_SOURCE = "lab"
RULE_BASED_SOURCES = {DRUG_SOURCE, LAB_SOURCE}
ENTITY_TYPES = (
    "CHẨN_ĐOÁN",
    "TRIỆU_CHỨNG",
    "THUỐC",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
)


def load_entities_from_source(filepath: str | Path) -> list[dict[str, Any]]:
    """Load one JSON entity list, returning an empty list for missing/bad files.

    Missing files are normal for a sparse source and do not produce an error.
    Invalid JSON or a valid non-list JSON value is reported to stderr with the
    filename, then treated as an empty source so one bad file does not stop the
    full merge batch.
    """

    path = Path(filepath)
    if not path.exists():
        return []

    try:
        with path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        _safe_print(f"[merge_entities] Invalid JSON in {path}: {exc}", file=sys.stderr)
        return []
    except OSError as exc:
        _safe_print(f"[merge_entities] Cannot read {path}: {exc}", file=sys.stderr)
        return []

    if not isinstance(data, list):
        _safe_print(
            f"[merge_entities] Expected a JSON list in {path}, got {type(data).__name__}",
            file=sys.stderr,
        )
        return []

    return data


def merge_entities_for_file(
    ner_entities: list[dict[str, Any]],
    drug_entities: list[dict[str, Any]],
    lab_entities: list[dict[str, Any]],
    *,
    source_file: str | None = None,
    conflict_log: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Merge three entity lists, resolve exact-span conflicts, and sort by span.

    Exact duplicates with the same ``position`` and ``type`` collapse to one
    entity. When two different sources emit the same position with different
    types, rule-based entities from drug/lab outputs win over NER entities; the
    dropped entity is logged when ``conflict_log`` is provided. If both entities
    have the same priority, both are kept and the conflict is logged as
    unresolved. Partial overlaps from different sources are never resolved
    automatically; both entities remain in the output and the pair is logged
    with ``"_partial_overlap": true``.
    """

    wrapped = _wrap_entities(ner_entities, NER_SOURCE)
    wrapped.extend(_wrap_entities(drug_entities, DRUG_SOURCE))
    wrapped.extend(_wrap_entities(lab_entities, LAB_SOURCE))

    deduplicated = _deduplicate_exact_entities(wrapped)
    kept = _resolve_same_span_different_type(deduplicated, source_file, conflict_log)
    _log_partial_overlaps(kept, source_file, conflict_log)

    return sorted((_entity_schema(item["entity"]) for item in kept), key=_sort_key)


def validate_merged_output(entities: list[dict[str, Any]], original_text: str) -> list[str]:
    """Validate entity offsets against the original text and return error strings.

    The function checks that every ``position`` is within ``[0, len(text)]`` and
    that ``original_text[start:end]`` exactly equals the entity ``text``.
    """

    errors: list[str] = []
    text_length = len(original_text)

    for index, entity in enumerate(entities):
        position = entity.get("position")
        if not _is_valid_position_shape(position):
            errors.append(f"entity[{index}] has invalid position: {position!r}")
            continue

        start, end = int(position[0]), int(position[1])
        if start < 0 or end < start or end > text_length:
            errors.append(
                f"entity[{index}] position [{start}, {end}] outside [0, {text_length}]"
            )
            continue

        expected = str(entity.get("text", ""))
        actual = original_text[start:end]
        if actual != expected:
            errors.append(
                f"entity[{index}] text mismatch at [{start}, {end}]: "
                f"expected {expected!r}, actual {actual!r}"
            )

    return errors


def process_all_files(
    ner_dir: str | Path,
    drug_dir: str | Path,
    lab_dir: str | Path,
    output_dir: str | Path,
    test_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Process all source basenames, write merged JSON/logs, and return summary.

    The source set is the union of JSON basenames found in ``ner_dir``,
    ``drug_dir``, and ``lab_dir``. If ``test_dir`` is supplied, basenames without
    a matching ``{basename}.txt`` are skipped as non-test diagnostic JSON files.
    A test file that has no JSON in all three sources is intentionally skipped;
    there is no source basename to process, and this avoids inventing outputs.
    Per-file exceptions are caught and recorded without stopping the batch.
    """

    ner_path = Path(ner_dir)
    drug_path = Path(drug_dir)
    lab_path = Path(lab_dir)
    output_path = Path(output_dir)
    test_path = Path(test_dir) if test_dir is not None else None
    output_path.mkdir(parents=True, exist_ok=True)

    source_stems = _collect_source_stems(ner_path, drug_path, lab_path)
    if test_path is not None:
        source_stems = [
            stem
            for stem in source_stems
            if (test_path / f"{stem}.txt").exists()
        ]

    conflicts: list[dict[str, Any]] = []
    mismatch_errors: list[dict[str, Any]] = []
    file_errors: list[dict[str, str]] = []
    type_counts: Counter[str] = Counter({entity_type: 0 for entity_type in ENTITY_TYPES})
    files_with_all_sources = 0
    empty_files = 0
    processed_files = 0
    total_files = len(source_stems)

    for index, stem in enumerate(source_stems, start=1):
        filename = f"{stem}.json"
        try:
            ner_entities = load_entities_from_source(ner_path / filename)
            drug_entities = load_entities_from_source(drug_path / filename)
            lab_entities = load_entities_from_source(lab_path / filename)

            if ner_entities and drug_entities and lab_entities:
                files_with_all_sources += 1
            if not ner_entities and not drug_entities and not lab_entities:
                empty_files += 1

            merged = merge_entities_for_file(
                ner_entities,
                drug_entities,
                lab_entities,
                source_file=filename,
                conflict_log=conflicts,
            )

            if test_path is not None:
                original_text = _read_original_text(test_path / f"{stem}.txt")
                validation_errors = validate_merged_output(merged, original_text)
                if validation_errors:
                    mismatch_errors.append({"file": filename, "errors": validation_errors})
                    _safe_print(
                        f"[merge_entities] Position mismatch in {filename}: "
                        f"{len(validation_errors)} error(s)",
                        file=sys.stderr,
                    )

            _write_json(output_path / filename, merged)
            type_counts.update(entity.get("type", "") for entity in merged)
            processed_files += 1
        except Exception as exc:  # noqa: BLE001 - keep the batch alive per file.
            file_errors.append({"file": filename, "error": str(exc)})
            _safe_print(f"[merge_entities] Failed to process {filename}: {exc}", file=sys.stderr)

        _safe_print(f"Đã xử lý {index}/{total_files} file")

    same_position_conflicts = sum(
        1 for conflict in conflicts if conflict.get("conflict_type") == "same_position_different_type"
    )
    partial_overlap_conflicts = sum(
        1 for conflict in conflicts if conflict.get("conflict_type") == "partial_overlap"
    )
    summary = {
        "total_files_processed": processed_files,
        "entity_counts_by_type": {entity_type: type_counts.get(entity_type, 0) for entity_type in ENTITY_TYPES},
        "overlap_conflicts": {
            "same_position_different_type": same_position_conflicts,
            "partial_overlap": partial_overlap_conflicts,
            "total": same_position_conflicts + partial_overlap_conflicts,
        },
        "files_with_entities_from_all_three_sources": files_with_all_sources,
        "empty_files": empty_files,
        "file_errors": file_errors,
    }

    _write_json(output_path / "overlap_conflicts.json", conflicts)
    _write_json(output_path / "position_mismatch_errors.json", mismatch_errors)
    _write_json(output_path / "merge_summary.json", summary)
    _safe_print(json.dumps(summary, ensure_ascii=False, indent=2))

    return summary


def _safe_print(message: str, *, file: Any = sys.stdout) -> None:
    """Print without crashing on Windows consoles with narrow encodings."""

    try:
        print(message, file=file)
    except UnicodeEncodeError:
        encoding = getattr(file, "encoding", None) or "utf-8"
        safe_message = message.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe_message, file=file)


def _wrap_entities(entities: Iterable[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    """Attach source metadata for merge decisions without changing output schema."""

    return [{"entity": _entity_schema(entity), "source": source} for entity in entities]


def _entity_schema(entity: dict[str, Any]) -> dict[str, Any]:
    """Return only the public entity schema fields."""

    return {
        "text": entity.get("text", ""),
        "type": entity.get("type", ""),
        "position": list(entity.get("position", [])),
    }


def _deduplicate_exact_entities(wrapped: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse entities with identical position and type."""

    best_by_key: dict[tuple[int, int, str], dict[str, Any]] = {}
    for item in wrapped:
        start, end = _position_tuple(item["entity"])
        key = (start, end, item["entity"].get("type", ""))
        current = best_by_key.get(key)
        if current is None or _source_priority(item["source"]) > _source_priority(current["source"]):
            best_by_key[key] = item

    return list(best_by_key.values())


def _resolve_same_span_different_type(
    wrapped: list[dict[str, Any]],
    source_file: str | None,
    conflict_log: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Drop lower-priority entities that have the same position but different type."""

    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for item in wrapped:
        groups.setdefault(_position_tuple(item["entity"]), []).append(item)

    drop_ids: set[int] = set()
    for items in groups.values():
        for left_index, left in enumerate(items):
            for right in items[left_index + 1 :]:
                if left["source"] == right["source"]:
                    continue
                if left["entity"].get("type") == right["entity"].get("type"):
                    continue

                winner = _choose_same_span_winner(left, right)
                dropped = None
                if winner is left:
                    dropped = right
                    drop_ids.add(id(right))
                elif winner is right:
                    dropped = left
                    drop_ids.add(id(left))

                _append_conflict(
                    conflict_log,
                    {
                        "file": source_file,
                        "conflict_type": "same_position_different_type",
                        "entities": [_log_entity(left), _log_entity(right)],
                        "kept_entity": _log_entity(winner) if winner is not None else None,
                        "dropped_entity": _log_entity(dropped) if dropped is not None else None,
                    },
                )

    return [item for item in wrapped if id(item) not in drop_ids]


def _log_partial_overlaps(
    wrapped: list[dict[str, Any]],
    source_file: str | None,
    conflict_log: list[dict[str, Any]] | None,
) -> None:
    """Log partial overlaps while leaving all entities untouched."""

    sorted_items = sorted(wrapped, key=lambda item: _sort_key(item["entity"]))
    for left_index, left in enumerate(sorted_items):
        left_start, left_end = _position_tuple(left["entity"])
        for right in sorted_items[left_index + 1 :]:
            right_start, right_end = _position_tuple(right["entity"])
            if right_start >= left_end:
                break
            if left["source"] == right["source"]:
                continue
            if (left_start, left_end) == (right_start, right_end):
                continue
            if left_start < right_end and right_start < left_end:
                _append_conflict(
                    conflict_log,
                    {
                        "file": source_file,
                        "conflict_type": "partial_overlap",
                        "_partial_overlap": True,
                        "entities": [_log_entity(left), _log_entity(right)],
                    },
                )


def _choose_same_span_winner(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any] | None:
    """Choose rule-based over NER for same-position different-type conflicts."""

    left_priority = _source_priority(left["source"])
    right_priority = _source_priority(right["source"])
    if left_priority == right_priority:
        return None
    return left if left_priority > right_priority else right


def _source_priority(source: str) -> int:
    """Return merge priority for a source."""

    if source in RULE_BASED_SOURCES:
        return 2
    return 1


def _log_entity(item: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return an entity copy plus source for review logs."""

    if item is None:
        return None
    entity = _entity_schema(item["entity"])
    entity["source"] = item["source"]
    return entity


def _append_conflict(conflict_log: list[dict[str, Any]] | None, conflict: dict[str, Any]) -> None:
    """Append a conflict if a log list was supplied."""

    if conflict_log is not None:
        conflict_log.append(conflict)


def _position_tuple(entity: dict[str, Any]) -> tuple[int, int]:
    """Return a normalized ``(start, end)`` tuple."""

    position = entity.get("position")
    if not _is_valid_position_shape(position):
        raise ValueError(f"Invalid entity position: {position!r}")
    return int(position[0]), int(position[1])


def _is_valid_position_shape(position: Any) -> bool:
    """Check that a position value has two integer-like values."""

    if not isinstance(position, list | tuple) or len(position) != 2:
        return False
    try:
        int(position[0])
        int(position[1])
    except (TypeError, ValueError):
        return False
    return True


def _sort_key(entity: dict[str, Any]) -> tuple[int, int]:
    """Sort by start, then end."""

    return _position_tuple(entity)


def _collect_source_stems(*directories: Path) -> list[str]:
    """Collect naturally sorted JSON basenames from source directories."""

    stems: set[str] = set()
    for directory in directories:
        if not directory.exists():
            continue
        stems.update(path.stem for path in directory.glob("*.json"))

    return sorted(stems, key=_natural_stem_key)


def _natural_stem_key(stem: str) -> tuple[int, int | str]:
    """Sort numeric stems numerically, then other stems lexically."""

    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def _read_original_text(path: Path) -> str:
    """Read original test text with explicit UTF-8 handling."""

    with path.open("r", encoding="utf-8-sig") as file:
        return file.read()


def _write_json(path: Path, data: Any) -> None:
    """Write JSON with explicit UTF-8."""

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Merge entity outputs from NER, drug, and lab sources.")
    parser.add_argument("--ner-dir", default="data/output_ner")
    parser.add_argument("--drug-dir", default="data/output_drug")
    parser.add_argument("--lab-dir", default="data/output_lab_v2")
    parser.add_argument("--output-dir", default="data/output_merged")
    parser.add_argument("--test-dir", default="data/test")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = _parse_args()
    process_all_files(
        ner_dir=args.ner_dir,
        drug_dir=args.drug_dir,
        lab_dir=args.lab_dir,
        output_dir=args.output_dir,
        test_dir=args.test_dir,
    )


if __name__ == "__main__":
    main()
