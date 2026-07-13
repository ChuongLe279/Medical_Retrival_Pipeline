"""
Embedding test data -> embedding vector
"""

import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.Embeddings.SapBert import SapBERTEmbeddings


def create_sapbert_embeddings():
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

    return SapBERTEmbeddings(
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=128,
        max_length=64,
    )


def sapbert_embedding(s: str) -> list[float]:
    if not isinstance(s, str):
        raise TypeError("Input must be a string.")

    s = " ".join(s.strip().split())

    if not s:
        raise ValueError("Input text cannot be empty.")
    
    embeddings = create_sapbert_embeddings()

    return embeddings.embed_query(s)