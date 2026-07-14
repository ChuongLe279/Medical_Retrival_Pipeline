"""
Không thể tin tưởng hoàn toàn vào kết quả của Sapbert (VD: 2 mã khác nhau có độ chính xác đều bằng 1.0)
Ta kết hợp kết quả của sapbert và rapidfuzz.
Rapid fuzz không cần embed. So sánh thẳng query với text list luôn.
"""
from rapidfuzz import process, fuzz
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.Linking.embedding import create_sapbert_embeddings
import json
from langchain_chroma import Chroma

RXNORM_CHROMA_DIR = PROJECT_ROOT / "src" / "Linking" / "text_list" / "rxnorm_text.json"
ICD10_CHROMA_DIR = PROJECT_ROOT / "src" / "Linking" / "text_list" / "icd10_text.json"
JSON_FOLDER_PATH = PROJECT_ROOT / "data" / "mock_data"  # File JSON test

import numpy as np


def rapidfuzz_retrieval(text_list, candidate_texts, candidate_codes, top_k=5, scorer=fuzz.WRatio):
    score_matrix = process.cdist(text_list, candidate_texts, scorer=scorer, workers=-1)
    code_scores = {}
    top_idx = np.argsort(-score_matrix, axis=1)[:, :top_k]
    for row_i, row in enumerate(top_idx):
        for idx in row:
            code = candidate_codes[idx]
            score = score_matrix[row_i, idx]
            if code not in code_scores or score > code_scores[code]:
                code_scores[code] = round(float(score), 2)
    return code_scores

if __name__ == "__main__":
    pass