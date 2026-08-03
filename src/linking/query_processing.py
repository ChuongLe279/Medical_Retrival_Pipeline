"""Apply deterministic manual English queries for entity linking.

The previous implementation generated lookup expansions with Qwen.  That
introduced unsupported brands, strengths, and diagnoses.  This replacement
loads the audited mapping in ``manual_queries.json`` and never loads a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.Linking.manual_queries import normalize_key


JSON_FOLDER = PROJECT_ROOT / "data" / "new_ner_output"
MANUAL_QUERY_PATH = Path(__file__).with_name("manual_queries.json")
SUPPORTED_LINKING_TYPES = {"THUỐC", "CHẨN_ĐOÁN"}


def json_sort_key(path: Path) -> tuple[int, int | str]:
    try:
        return 0, int(path.stem)
    except ValueError:
        return 1, path.stem.casefold()


def load_manual_queries(path: str | Path = MANUAL_QUERY_PATH) -> dict[str, dict[str, list[str]]]:
    mapping_path = Path(path)
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{mapping_path} must contain a JSON object")

    validated: dict[str, dict[str, list[str]]] = {}
    for entity_type in SUPPORTED_LINKING_TYPES:
        type_mapping = data.get(entity_type)
        if not isinstance(type_mapping, dict):
            raise ValueError(f"Missing mapping object for {entity_type}")
        validated[entity_type] = {}
        for key, queries in type_mapping.items():
            if not isinstance(key, str) or not isinstance(queries, list):
                raise ValueError(f"Invalid manual mapping entry for {entity_type}: {key!r}")
            if len(queries) > 1 or not all(isinstance(query, str) and query.strip() for query in queries):
                raise ValueError(
                    f"Manual queries must be empty or contain one non-empty string: "
                    f"{entity_type} {key!r} -> {queries!r}"
                )
            validated[entity_type][key] = list(queries)
    return validated


def manual_linking_queries(
    text: str,
    entity_type: str,
    mapping: dict[str, dict[str, list[str]]],
) -> list[str]:
    if entity_type not in SUPPORTED_LINKING_TYPES:
        return []
    key = normalize_key(text)
    try:
        return list(mapping[entity_type][key])
    except KeyError as exc:
        raise KeyError(f"No manual linking query for {entity_type} {text!r}") from exc


def query_expansion(
    json_files: list[Path],
    mapping_path: str | Path = MANUAL_QUERY_PATH,
) -> None:
    """Apply manual queries while preserving every unrelated entity field."""
    mapping = load_manual_queries(mapping_path)
    print(f"Found {len(json_files)} JSON files.")

    for file_path in json_files:
        print(f"Processing file: {file_path.name}")
        data = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not all(isinstance(entity, dict) for entity in data):
            raise ValueError(f"{file_path} must contain a JSON list of objects")

        for entity in data:
            entity_type = entity.get("type", "")
            if entity_type not in SUPPORTED_LINKING_TYPES:
                entity.pop("eng_translation", None)
                continue
            source_text = entity.get("text")
            if not isinstance(source_text, str) or not source_text.strip():
                raise ValueError(f"Invalid entity text in {file_path.name}: {source_text!r}")
            entity["eng_translation"] = manual_linking_queries(
                source_text, entity_type, mapping
            )

        temporary = file_path.with_name(f".{file_path.name}.tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(file_path)
        print(f"Saved updated data to: {file_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-folder", type=Path, default=JSON_FOLDER)
    parser.add_argument("--mapping", type=Path, default=MANUAL_QUERY_PATH)
    args = parser.parse_args()

    if not args.json_folder.is_dir():
        parser.error(f"Folder does not exist: {args.json_folder}")
    json_files = sorted(args.json_folder.glob("*.json"), key=json_sort_key)
    if not json_files:
        parser.error(f"No JSON files found in: {args.json_folder}")
    query_expansion(json_files, args.mapping)


if __name__ == "__main__":
    main()
