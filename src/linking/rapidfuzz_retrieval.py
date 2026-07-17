"""
Không thể tin tưởng hoàn toàn vào kết quả của Sapbert (VD: 2 mã khác nhau có độ chính xác đều bằng 1.0)
Ta kết hợp kết quả của sapbert và rapidfuzz.
Rapid fuzz không cần embed. So sánh thẳng query với text list luôn.
"""
from rapidfuzz import process, fuzz
import sys
from pathlib import Path
import re
import unicodedata
import json
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RXNORM_TEXT_DIR = PROJECT_ROOT / "src" / "Linking" / "text_list" / "rxnorm_text.json"
ICD10_TEXT_DIR = PROJECT_ROOT / "src" / "Linking" / "text_list" / "icd10_text.json"
JSON_FOLDER_PATH = PROJECT_ROOT / "data" / "output_merged"  # File JSON test

def loading_object(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    candidate_texts = [c["text"] for c in data]
    candidate_codes = [c["code"] for c in data]
    return candidate_codes, candidate_texts

def normalize_for_fuzzy(text: str) -> str:
    """Chuẩn hóa code trước khi retrieve"""
    if not isinstance(text, str):
        return ""

    # Chuẩn hóa Unicode
    text = unicodedata.normalize("NFKC", text)

    # Chuyển toàn bộ chữ hoa thành chữ thường
    text = text.lower().strip()

    # Chuẩn hóa đơn vị
    text = re.sub(r"\bmilligrams?\b", "mg", text)
    text = re.sub(r"\bmilliliters?\b", "ml", text)
    text = re.sub(r"\bmicrograms?\b", "mcg", text)

    # 10mg -> 10 mg
    text = re.sub(
        r"(\d)\s*(mg|ml|mcg|g)\b",
        r"\1 \2",
        text,
    )

    # Thay dấu gạch dưới bằng khoảng trắng
    text = text.replace("_", " ")

    # Bỏ dấu câu, nhưng giữ dấu chấm thập phân và %
    text = re.sub(r"[^\w\s.%]", " ", text)

    # Chuẩn hóa nhiều khoảng trắng thành một
    return " ".join(text.split())

def rapidfuzz_retrieval(text_list, candidate_texts, candidate_codes, top_k=5, scorer=fuzz.WRatio):
    score_matrix = process.cdist(text_list, candidate_texts, scorer=scorer, workers=-1)
    code_scores = {}
    if score_matrix.shape[1] == 0 or top_k <= 0:
        return {}

    top_k = min(top_k, score_matrix.shape[1])

    top_idx = np.argpartition(
        -score_matrix,
        kth=top_k - 1,
        axis=1,
    )[:, :top_k]
    for row_i, row in enumerate(top_idx):
        for idx in row:
            code = candidate_codes[idx]
            score = float(score_matrix[row_i, idx]) / 100.0

            if code not in code_scores or score > code_scores[code]:
                code_scores[code] = round(score, 6)
    return code_scores


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
    rxnorm_code, rxnorm_text = loading_object(RXNORM_TEXT_DIR)
    icd10_code, icd10_text = loading_object(ICD10_TEXT_DIR)

    for file_path in json_files:
        print(f"Processing file: {file_path.name}")

        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        
        # Retrieve dựa trên trường type
        for entity in data:
            entity_type = entity.get("type", "")
            queries = entity.get("eng_translation", [])
            if isinstance(queries, str):
                queries = [queries]
            elif not isinstance(queries, list):
                queries = []

            queries = [
                normalize_for_fuzzy(query)
                for query in queries
                if isinstance(query, str) and query.strip()
            ]
            entity.setdefault("retriever", {})
            entity["retriever"]["rapidfuzz"] = {}

            if entity_type == "THUỐC":
                entity["retriever"]["rapidfuzz"] = rapidfuzz_retrieval(
                    text_list = queries,
                    candidate_texts = rxnorm_text,
                    candidate_codes = rxnorm_code,
                    top_k=3,
                )
            elif entity_type == "CHẨN_ĐOÁN":
                entity["retriever"]["rapidfuzz"] = rapidfuzz_retrieval(
                    text_list = queries,
                    candidate_texts = icd10_text,
                    candidate_codes = icd10_code,
                    top_k=3,
                )
            else:
                entity["retriever"]["rapidfuzz"] = {}
            # Sort rapidfuzz results by score, descending
            entity["retriever"]["rapidfuzz"] = dict(
                sorted(
                    entity["retriever"]["rapidfuzz"].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            )
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=4,
            )

        print(f"Saved updated data to: {file_path.name}")