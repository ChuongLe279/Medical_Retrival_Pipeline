"""
Retrieve RxNorm hoặc ICD10 dựa vào CANDIDATE.
"""
import sys
from pathlib import Path
import shutil
import json
import torch
from transformers import AutoTokenizer, AutoModel
from langchain_core.documents import Document
from langchain_chroma import Chroma
device = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.Embeddings.SapBert import SapBERTEmbeddings

MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
"""
Tải tokenizer đi kèm mô hình.
Chuyển văn bản thành token và token ID mà SAPBert hiểu được.
"""
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

"""Tải SAPBERT"""
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
"""Đưa mô hình vào chế độ inference/evaluation."""
model.eval()

JSON_FOLDER = Path()
RXNORM_CHROMA = Path()
ICD10_CHROMA = Path()

"""
Embed text
"""
def embedding_text(s: str):
    pass



def retriever(CHROMA_DIR):
    embeddings = SapBERTEmbeddings(
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=128,
        max_length=64,
    )

    vectorstore = Chroma(
        collection_name="icd10",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )

if __name__ == "__main__":

    if not JSON_FOLDER.exists():
        raise FileNotFoundError(
            f"Folder does not exist: {JSON_FOLDER}"
        )

    json_files = sorted(
        JSON_FOLDER.glob("*.json"),
        key=lambda path: int(path.stem)
    )

    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in: {JSON_FOLDER}"
        )
    
    print(f"Found {len(json_files)} JSON files.")

    for file_path in json_files:
        print(f"Processing file: {file_path.name}")

        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        
        for entity in data:
            if entity["type"] == "THUỐC":
                retriever(RXNORM_CHROMA)
            elif entity["type"] == "CHẨN ĐOÁN":
                retriever(ICD10_CHROMA)
            else:
                entity["candidates"] = []

        with file_path.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print(f"Saved updated data to: {file_path.name}")

        json_mock = [
            {
                "code": "A00.0",
                "text": "text content",
                "retriever": {
                    "sapbert": {
                        "001": 2,
                        "034": 2,
                        "012": 2,
                    },
                    "rapidfuzz": {
                        "001": 2,
                        "001": 2,
                        "001": 2,
                    },
                }
            }
        ]
    




