"""
Từ rxnorm.db -> Vector store
"""
import sqlite3
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_ROOT / "data" / "db" / "RxNorm" / "rxnorm.db"
CHROMA_DIR = PROJECT_ROOT / "data" / "db" / "chroma"


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

    rows = cur.execute(query).fetchall()
    conn.close()

    objects = []
    for row in rows:
        obj = dict(row)
        objects.append(obj)

    return objects

def object_to_document(obj: dict) -> Document:
    """
    Chuyển một object RxNorm thành LangChain Document.
    page_content là text dùng để embedding.
    metadata là thông tin phụ để filter/search sau này. 
    """

    drug_name = obj.get("STR", "")
    rxcui = obj.get("RXCUI", "")
    tty = obj.get("TTY", "")
    sab = obj.get("SAB", "")
    code = obj.get("CODE", "")

    text = f"""
        Drug name: {drug_name}
        RxCUI: {rxcui}
        Term type: {tty}
        Source: {sab}
        Code: {code}
    """.strip()

    metadata = {
        "rxcui": rxcui,
        "tty": tty,
        "sab": sab,
        "code": code,
        "drug_name": drug_name,
    }

    return Document(
        page_content=text,
        metadata=metadata
    )

def build_vectorstore():
    print("Loading RxNorm objects from database...")
    objects = fetch_rxnorm_objects(DB_PATH)

    print(f"Loaded {len(objects):,} objects")

    documents = []
    for obj in objects:
        doc = object_to_document(obj)
        documents.append(doc)

    print(f"Created {len(documents):,} documents")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = Chroma(
        collection_name="rxnorm",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    batch_size = 1000

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        vectorstore.add_documents(batch)
        print(f"Inserted {i + len(batch):,}/{len(documents):,} documents into Chroma")

    print("Done building RxNorm vector store.")


if __name__ == "__main__":
    build_vectorstore()