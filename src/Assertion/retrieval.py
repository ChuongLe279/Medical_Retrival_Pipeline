"""
Ta sẽ dùng bge-m3 để làm phần assertion. 
Ta sẽ tính luôn hybrid score trong file này.
"""
import torch
import chromadb
from FlagEmbedding import BGEM3FlagModel
import sys
import json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ASSERTION_CHROMA_DIR = PROJECT_ROOT / "src" / "Assertion" / "data" / "assertion_chroma"
NER_FOLDER_PATH = PROJECT_ROOT / "src" / "Assertion" / "data" / "mock_ner"
MODEL_NAME = "BAAI/bge-m3"

def check_folder_exist(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Folder does not exist: {path}"
        )

def hybrid_assertion_retrieval(ner_json, top_k = 5, alpha = 0.5):
    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    
    for entity in data:
        query_text  = entity["text"]
        # Encoding query
        query_outputs = model.encode([query_text], return_dense=True, return_sparse=True)
        query_dense = query_outputs['dense_vecs'][0].tolist() # type: ignore
        query_sparse = query_outputs['lexical_weights'][0]

    # Fetch semantic candidates using the dense query vector
    # Retrieve more candidates (e.g., top_k * 3) to give the lexical search room to rerank
    candidate_pool_size = max(top_k * 3, 10)
    candidates = collection.query(
        query_embeddings=[query_dense], # type: ignore
        n_results=candidate_pool_size,
        include=["documents", "metadatas", "distances"]
    )
    # If database is empty
    if not candidates['ids'] or len(candidates['ids'][0]) == 0:
        print("No documents found in database.")
        return
    
    # Narrow type so no error
    documents = candidates.get("documents")
    metadatas = candidates.get("metadatas")
    distances = candidates.get("distances")

    if documents is None or metadatas is None or distances is None:
        raise RuntimeError("Chroma did not return all requested fields")
    # Rerank candidates using BGE-M3's lexical matching scores

    for i in range(len(candidates['ids'][0])):
        text = documents[0][i]
        meta = metadatas[0][i]
        dense_distance = distances[0][i]

        # Convert Chroma's cosine distance (1 - similarity) back to a similarity score
        dense_score = 1.0 - dense_distance  
        
        # Retrieve and parse the stored sparse weights from metadata
        doc_sparse_weights = json.loads(meta['sparse_weights']) # type: ignore
        
        # Compute the lexical match score
        sparse_score = model.compute_lexical_matching_score(query_sparse, doc_sparse_weights) # type: ignore
        
        # Calculate combined hybrid score
        # alpha = 1.0 -> Pure Semantic (Dense)
        # alpha = 0.0 -> Pure Keyword (Sparse)
        hybrid_score = (alpha * dense_score) + ((1.0 - alpha) * sparse_score)

if __name__ == "__main__":
    check_folder_exist(NER_FOLDER_PATH)
    check_folder_exist(ASSERTION_CHROMA_DIR) 

    print("Loading BGE-M3 model...")
    model = BGEM3FlagModel(MODEL_NAME, use_fp16=True)

    # 2. Connect to the existing local Chroma DB
    print("Connecting to local Chroma DB...")
    client = chromadb.PersistentClient(path=ASSERTION_CHROMA_DIR)
    
    try:
        collection = client.get_collection(name="bge_m3_hybrid")
        for file_path in NER_FOLDER_PATH.glob("*.json"):
            print(f"Processing {file_path.name}")

            hybrid_assertion_retrieval(file_path)

    except Exception as e:
        print("Error: Could not find collection. Did you run ingest.py first?")
        
        
        
            