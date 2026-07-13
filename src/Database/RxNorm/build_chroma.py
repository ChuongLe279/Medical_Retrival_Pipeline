"""
Từ rxnorm.db -> Vector store
"""
import sqlite3
import sys
from pathlib import Path
import shutil
import torch
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


DB_PATH = PROJECT_ROOT / "data" / "db" / "RxNorm" / "rxnorm.db"
CHROMA_DIR = PROJECT_ROOT / "data" / "db" / "RxNorm_chroma"
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

# Tránh dữ liệu trùng lặp
def make_document_id(obj: dict) -> str:
    rxaui = obj.get("RXAUI")

    if not rxaui:
        raise ValueError(
            f"Missing RXAUI for RXCUI={obj.get('RXCUI')}"
        )

    return f"rxnorm:{rxaui}"

def fetch_rxnorm_objects(db_path: Path):
    """
    rxnorm.db → Gom thành object → json object
    VD:
    {
        "RXCUI": "1049630",
        "LAT": "ENG",
        "TTY": "SCD",
        "STR": "acetaminophen 325 MG Oral Tablet",
        "SAB": "RXNORM",
        "CODE": "1049630",
        ...
    }
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    query = """
    SELECT 
        RXCUI,
        LAT,
        TS,
        LUI,
        STT,
        SUI,
        ISPREF,
        RXAUI,
        SAUI,
        SCUI,
        SDUI,
        SAB,
        TTY,
        CODE,
        STR,
        SRL,
        SUPPRESS,
        CVF
    FROM rxnconso
    WHERE LAT = 'ENG'
      AND SUPPRESS = 'N'
      AND STR IS NOT NULL
    """

    """
    Giải thích điều kiện "WHERE" của câu truy vấn trên:
    LAT = 'ENG' : Chỉ lấy tiếng anh, bỏ tiếng tây ban nha
    SUPPRESS = 'N': Chỉ lấy tên thuốc không bị đánh dấu loại bỏ.
    VD như : O là tên thuốc lỗi thời
    STR IS NOT NULL: Nếu null thì khỏi retrieve 
    """
    rows = cur.execute(query).fetchall()
    conn.close()

    objects = []
    for row in rows:
        obj = dict(row)
        objects.append(obj)

    return objects

def object_to_document(obj: dict) -> Document:
    """
    Chuyển một RxNorm entity thành LangChain Document.

    - page_content: tên thuốc (STR), là phần được SapBERT embed.
    - metadata: thông tin không embed, dùng để trả kết quả, lọc
      hoặc kiểm tra sau retrieval.
    """

    drug_name = obj.get("STR", "")
    rxcui = obj.get("RXCUI", "")
    tty = obj.get("TTY", "")
    sab = obj.get("SAB", "")
    code = obj.get("CODE", "")

    metadata = {
        # Mã RxNorm concept: mã đầu ra chính sau retrieval
        "rxcui": rxcui,

        # Loại term, ví dụ IN/PIN/SCD/SBD
        "tty": tty,

        # Nguồn từ vựng, ví dụ RXNORM
        "sab": sab,

        # Mã gốc của term trong source vocabulary
        "code": code,

        # Giữ lại tên thuốc để dễ đọc/kiểm tra kết quả
        "drug_name": drug_name,
    }

    """
    Mỗi entity RxNorm rất ngắn, nên mỗi entity được xem là một
    document/chunk riêng. Chỉ STR được embed để đo độ giống ngữ nghĩa
    giữa câu truy vấn và tên thuốc RxNorm.
    """
    return Document(
        page_content=drug_name,
        metadata=metadata
    )

def build_vectorstore():
    print("Loading RxNorm objects from database...")
    objects = fetch_rxnorm_objects(DB_PATH)
    print(f"Loaded {len(objects):,} objects")

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
        max_length=64
    )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma(
        collection_name="rxnorm",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )

    chroma_batch_size = 1_000

    for start in range(0, len(documents), chroma_batch_size):
        end = start + chroma_batch_size

        batch_documents = documents[start:end]
        batch_objects = objects[start:end]

        batch_ids = [
            make_document_id(obj)
            for obj in batch_objects
        ]

        vectorstore.add_documents(
            documents=batch_documents,
            ids=batch_ids,
        )

        inserted = start + len(batch_documents)
        print(
            f"Inserted {inserted:,}/{len(documents):,} "
            "documents into Chroma"
        )

    print("Done building RxNorm vector store.")
    return vectorstore


if __name__ == "__main__":
    # Xóa vector store cũ nếu đã tồn tại
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    # build_vectorstore() sẽ tự tạo lại thư mục
    build_vectorstore()
