"""
Retrieve trên chroma của SAPBERT
"""

from embedding import create_sapbert_embeddings
from pathlib import Path
import json
from langchain_chroma import Chroma

RXNORM_CHROMA_DIR = Path("")
ICD10_CHROMA_DIR = Path("")
JSON_FOLDER_PATH = Path("") # File json kết quả

K = 5 # LẤY 5 THẰNG ĐẦU TIÊN
embeddings = create_sapbert_embeddings()

rxnorm_vector_store = Chroma(
    collection_name="rxnorm", 
    persist_directory=str(RXNORM_CHROMA_DIR),
    embedding_function=embeddings,
)

icd10_vector_store = Chroma(
    collection_name="icd10",
    persist_directory=str(ICD10_CHROMA_DIR),
    embedding_function=embeddings,
)

def sapbert_retrieval(vector_store, text, code_key, k):
    """
        Embed query bằng SapBERT, retrieve top-k và trả về:

        {
            "code_1": score_1,
            "code_2": score_2
        }
    """

    text = " ".join(text.strip().split())

    if not text:
        return {}

    # Lấy nhiều hơn vì nhiều synonym có thể chung một code
    retrieve_k = k * 4

    results = vector_store.similarity_search_with_relevance_scores(
        query=text,
        k=retrieve_k,
    )

    code_to_score = {}

    for document, score in results:
        code = document.metadata.get(code_key)

        if code is None or not str(code).strip():
            continue

        code = str(code).strip()
        score = float(score)

        # Giữ score cao nhất nếu nhiều synonym chung code
        code_to_score[code] = max(
            score,
            code_to_score.get(code, float("-inf")),
        )

    sorted_results = sorted(
        code_to_score.items(),
        key=lambda item: item[1],
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
    for file_path in json_files:
        print(f"Processing file: {file_path.name}")

        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        
        # Retrieve dựa trên trường type
        for entity in data:
            entity_type = entity.get("type", "")
            query = entity.get("eng_translation", "")

            entity["retriever"] = {
                "sapbert": {}
            }
            if entity_type == "THUỐC":
                entity["retriever"]["sapbert"] = sapbert_retrieval(
                    vector_store=rxnorm_vector_store,
                    text=query,
                    code_key="rxcui",
                    k=K,
                )
            elif entity_type == "CHẨN ĐOÁN":
                entity["retriever"]["sapbert"] = sapbert_retrieval(
                    vector_store=icd10_vector_store,
                    text=query,
                    code_key="code",
                    k=K,
                )
            else:
                entity["retriever"]["sapbert"] = {}
            
            
        with file_path.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=4,
            )

        print(f"Saved updated data to: {file_path.name}")

