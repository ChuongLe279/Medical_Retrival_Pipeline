import sys
from pathlib import Path
import shutil
import torch
import json
from transformers import AutoTokenizer, AutoModel
from langchain_core.documents import Document
from langchain_chroma import Chroma

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.Embeddings.SapBert import SapBERTEmbeddings

CHROMA_DIR = PROJECT_ROOT / "data" / "db" / "ICD10_chroma"
JSON_PATH = PROJECT_ROOT / "data" / "db" / "ICD10" / "icd10.json"
MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
device = "cuda" if torch.cuda.is_available() else "cpu"
"""
Tải tokenizer đi kèm mô hình.
Chuyển văn bản thành token và token ID mà SAPBert hiểu được.
"""
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

"""Tải SAPBERT"""
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
"""Đưa mô hình vào chế độ inference/evaluation."""
model.eval()

def object_to_document(obj: dict) -> Document:
    code = obj["code"].strip()
    text = obj["text"].strip()

    return Document(
        page_content=text,   
        metadata = {
            "code": code,
            "text": text,
            "category": code[:3],
            "source": "ICD-10-CM",
        }
    )

def build_vectorstore():
    print(f"Loading ICD-10 records from {JSON_PATH}...")

    if not JSON_PATH.is_file():
        raise FileNotFoundError(f"ICD-10 JSON file not found: {JSON_PATH}")

    with JSON_PATH.open("r", encoding="utf-8") as file:
        objects = json.load(file)

    if not isinstance(objects, list):
        raise ValueError(
            f"Expected a JSON list in {JSON_PATH}, "
            f"but received {type(objects).__name__}"
        )

    print(f"Loaded {len(objects):,} ICD-10 records")

    documents = [
        object_to_document(obj)
        for obj in objects
    ]
    print(f"Created {len(documents):,} documents")

    embeddings = SapBERTEmbeddings(
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=128,
        max_length=64,
    )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma(
        collection_name="icd10",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )

    chroma_batch_size = 1_000

    for start in range(0, len(documents), chroma_batch_size):
        end = min(start + chroma_batch_size, len(documents))

        batch_documents = documents[start:end]
        batch_objects = objects[start:end]

        # ICD-10 codes are unique, so they provide stable Chroma IDs.
        batch_ids = [
            f"icd10:{obj['code'].strip().upper()}"
            for obj in batch_objects
        ]

        vectorstore.add_documents(
            documents=batch_documents,
            ids=batch_ids,
        )

        print(
            f"Inserted {end:,}/{len(documents):,} "
            "documents into Chroma"
        )

    print("Done building ICD-10 vector store.")
    return vectorstore

if __name__ == "__main__":
    # Xóa vector store cũ nếu đã tồn tại
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    # build_vectorstore() sẽ tự tạo lại thư mục
    build_vectorstore()