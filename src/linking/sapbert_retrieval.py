"""
Retrieve trên chroma của SAPBERT
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.Linking.embedding import create_sapbert_embeddings
import json
from langchain_chroma import Chroma
import re
import unicodedata

RXNORM_CHROMA_DIR = PROJECT_ROOT / "data" / "db" / "RxNorm_chroma"
ICD10_CHROMA_DIR = PROJECT_ROOT / "data" / "db" / "ICD10_chroma"
JSON_FOLDER_PATH = PROJECT_ROOT / "data" / "new_ner_output"  # File JSON test

K = 5 # LẤY 5 THẰNG ĐẦU TIÊN
embeddings = create_sapbert_embeddings()

def cosine_relevance_score(distance: float) -> float:
    return max(0.0, min(1.0, 1.0 - distance))

rxnorm_vector_store = Chroma(
    collection_name="rxnorm", 
    persist_directory=str(RXNORM_CHROMA_DIR),
    embedding_function=embeddings,
    relevance_score_fn=cosine_relevance_score,
)

icd10_vector_store = Chroma(
    collection_name="icd10",
    persist_directory=str(ICD10_CHROMA_DIR),
    embedding_function=embeddings,
    relevance_score_fn=cosine_relevance_score,
)

import unicodedata

def normalize_for_sapbert(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.strip().split())

def sapbert_retrieval(vector_store, texts, code_key, k):
    """
        Embed query bằng SapBERT, retrieve top-k và trả về:

        {
            "code_1": score_1,
            "code_2": score_2
        }
    """

    """
    Embed từng query trong list, retrieve và hợp nhất kết quả.

    texts có thể là:
    - string
    - list[string]
    """

    # Cho phép truyền một string
    if isinstance(texts, str):
        texts = [texts]

    # Làm sạch và loại query trùng
    cleaned_texts = []

    for text in texts:
        if not isinstance(text, str):
            continue

        text = " ".join(text.strip().split())

        if text and text not in cleaned_texts:
            cleaned_texts.append(text)

    if not cleaned_texts:
        return {}

    retrieve_k = k * 4

    code_to_score = {}
    code_hit_count = {}

    for query in cleaned_texts:
        # Chroma tự embed từng query bằng SapBERT
        results = vector_store.similarity_search_with_relevance_scores(
            query=query,
            k=retrieve_k,
        )

        for document, score in results:
            code = document.metadata.get(code_key)

            if code is None or not str(code).strip():
                continue

            code = str(code).strip()
            score = float(score)

            # Giữ score tốt nhất của code trên toàn bộ query
            code_to_score[code] = max(
                score,
                code_to_score.get(code, float("-inf")),
            )

            # Đếm code xuất hiện trong bao nhiêu lần retrieval
            code_hit_count[code] = (
                code_hit_count.get(code, 0) + 1
            )

    # Ưu tiên score cao; nếu bằng nhau thì ưu tiên code xuất hiện nhiều
    sorted_results = sorted(
        code_to_score.items(),
        key=lambda item: (
            item[1],
            code_hit_count[item[0]],
        ),
        reverse=True,
    )[:k]

    return {
        code: round(score, 6)
        for code, score in sorted_results
    }

def json_sort_key(path: Path):
    try:
        return 0, int(path.stem)
    except ValueError:
        return 1, path.stem


if __name__ == "__main__":

    if not JSON_FOLDER_PATH.exists():
        raise FileNotFoundError(
            f"Folder does not exist: {JSON_FOLDER_PATH}"
        )
    
    json_files = sorted(
        JSON_FOLDER_PATH.glob("*.json"),
        key=json_sort_key
    )
    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in: {JSON_FOLDER_PATH}"
        )
    retrieval_cache = {}
    for file_path in json_files:
        print(f"Processing file: {file_path.name}")

        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        
        # Retrieve dựa trên trường type
        for entity in data:
            entity_type = entity.get("type", "")
            queries = entity.get("eng_translation", [])

            #Chuẩn hóa text
            if isinstance(queries, str):
                queries = [queries]
            elif not isinstance(queries, list):
                queries = []

            queries = [
                normalize_for_sapbert(query)
                for query in queries
                if isinstance(query, str) and query.strip()
            ]
            entity.setdefault("retriever", {})
            entity["retriever"]["sapbert"] = {}

            cache_key = (entity_type, tuple(queries))
            if cache_key not in retrieval_cache:
                if entity_type == "THUỐC":
                    retrieval_cache[cache_key] = sapbert_retrieval(
                        vector_store=rxnorm_vector_store,
                        texts=queries,
                        code_key="rxcui",
                        k=K,
                    )
                elif entity_type == "CHẨN_ĐOÁN":
                    retrieval_cache[cache_key] = sapbert_retrieval(
                        vector_store=icd10_vector_store,
                        texts=queries,
                        code_key="code",
                        k=K,
                    )
                else:
                    retrieval_cache[cache_key] = {}
            entity["retriever"]["sapbert"] = dict(retrieval_cache[cache_key])
            
            
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=4,
            )

        print(f"Saved updated data to: {file_path.name}")

