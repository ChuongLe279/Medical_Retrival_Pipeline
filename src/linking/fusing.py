"""
Kết hợp kết quả của rapidfuzz và sapbert.
Final confidence score = 0.8 * sapbert + 0.2 * rapidfuzz
Lấy kết quả cao nhất và bỏ vào trường candidate.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import json

JSON_FOLDER_PATH = PROJECT_ROOT / "data" / "mock_data"  # File JSON test


SAPBERT_WEIGHT = 0.8
RAPIDFUZZ_WEIGHT = 0.2


def fuse_entity_result(entity: dict) -> None:
    """
    Fuse kết quả retrieval của một entity và cập nhật trường candidate.
    """

    retriever = entity.get("retriever", {})

    sapbert_results = retriever.get("sapbert", {})
    rapidfuzz_results = retriever.get("rapidfuzz", {})

    # Lấy toàn bộ code từ cả hai retriever
    all_codes = set(sapbert_results) | set(rapidfuzz_results)

    if not all_codes:
        entity["candidate"] = None
        return

    final_scores = {}

    for code in all_codes:
        sapbert_score = float(sapbert_results.get(code, 0.0))
        rapidfuzz_score = float(rapidfuzz_results.get(code, 0.0))

        final_score = (
            SAPBERT_WEIGHT * sapbert_score
            + RAPIDFUZZ_WEIGHT * rapidfuzz_score
        )

        final_scores[code] = round(final_score, 6)

    # Lấy code có final score cao nhất
    best_code, best_score = max(
        final_scores.items(),
        key=lambda item: item[1],
    )

    entity["candidate"] = {
        "code": best_code,
        "confidence": best_score,
    }


def fusing_result(json_files: list[Path]) -> None:
    print(f"Found {len(json_files)} JSON files.")

    for file_path in json_files:
        print(f"Processing file: {file_path.name}")

        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError(
                f"Expected a JSON list in file: {file_path.name}"
            )

        for entity in data:
            fuse_entity_result(entity)

        with file_path.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print(f"Saved updated data to: {file_path.name}")



if __name__ == "__main__":
    if not JSON_FOLDER_PATH.exists():
        raise FileNotFoundError(
            f"Folder does not exist: {JSON_FOLDER_PATH}"
        )
    
    json_files = sorted(
        JSON_FOLDER_PATH.glob("*.json"),
        key=lambda path: int(path.stem)
    )

    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in: {JSON_FOLDER_PATH}"
        )
    
    fusing_result(json_files)