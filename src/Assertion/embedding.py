"""
Embed các file trong folder json để nhét vào vector store. Dùng bge-m3.
"""

import torch
import chromadb
from FlagEmbedding import BGEM3FlagModel
import shutil
import sys
import json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ASSERTION_CHROMA_DIR = PROJECT_ROOT / "src" / "Assertion" / "data" / "assertion_chroma"
JSON_FOLDER_PATH = PROJECT_ROOT / "src" / "Assertion" / "data" / "json" # Chứa example phrases
MODEL_NAME = "BAAI/bge-m3"

model = BGEM3FlagModel(
    MODEL_NAME,
    use_fp16=torch.cuda.is_available()
)


def check_folder_exist(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Folder does not exist: {path}"
        )

def batch_iterable(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def get_documents(json_file) -> tuple[list[str], list[str], list[dict]]:
    documents = []
    ids = []
    metadatas = []
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    for entity in data:
        documents.append(entity["text"])
        ids.append(entity["id"])
        metadatas.append(entity.get("metadata", {}))
    return documents, ids, metadatas

def assertion_embedding(json_folder, chroma_folder, batch_size: int = 100):
    # Initialize Local Chroma Client ONCE
    client = chromadb.PersistentClient(path=chroma_folder)
    collection = client.get_or_create_collection(
        name="bge_m3_hybrid",
        metadata={"hnsw:space": "cosine"} # Using cosine distance
    )

    for file_path in json_folder.glob("*.json"):
        print(f"Embedding {file_path.name}")
        documents, ids, metadatas  = get_documents(file_path)

        doc_batches = list(batch_iterable(documents, batch_size))
        id_batches = list(batch_iterable(ids, batch_size))
        meta_batches = list(batch_iterable(metadatas, batch_size))

        total_batches = len(doc_batches)
        for batch_idx, (doc_batch, id_batch, meta_batch) in enumerate(
            zip(doc_batches, id_batches, meta_batches), start=1
        ):
            print(f"  Batch {batch_idx}/{total_batches} ({len(doc_batch)} docs)")

            # Generate BOTH dense and sparse embeddings in a SINGLE forward pass
            outputs = model.encode(doc_batch, return_dense=True, return_sparse=True)
            dense_embeddings = outputs['dense_vecs'].tolist()  # type: ignore
            sparse_embeddings = outputs['lexical_weights']  # List of dicts: [{'token': weight}, ...]

            # Merge original metadata fields with the sparse weights (flattened, Chroma-safe)
            final_metadatas = []
            for meta, sparse in zip(meta_batch, sparse_embeddings):
                merged = dict(meta)  # copy so we don't mutate original
                merged["sparse_weights"] = json.dumps(
                    {str(token): float(weight) for token, weight in sparse.items()} # type: ignore
                )
                final_metadatas.append(merged)

            # Add batch to Chroma
            collection.add(
                ids=id_batch,
                embeddings=dense_embeddings,
                documents=doc_batch,
                metadatas=final_metadatas  # type: ignore
            )
        print("Ingestion complete!")



if __name__ == "__main__":
    check_folder_exist(JSON_FOLDER_PATH)
    check_folder_exist(ASSERTION_CHROMA_DIR)
    
    print("Deleting existing chroma.")
    for item in ASSERTION_CHROMA_DIR.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

    assertion_embedding(JSON_FOLDER_PATH, ASSERTION_CHROMA_DIR)


    