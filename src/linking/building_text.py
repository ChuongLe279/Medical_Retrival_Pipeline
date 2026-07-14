"""Tạo text list để cho rapidfuzz retrieve"""

from pathlib import Path
import json
import unicodedata
import re
import sqlite3


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ICD10_SOURCE_PATH = PROJECT_ROOT / "data" / "db" / "ICD10" / "output.json"
ICD10_COPY_PATH = PROJECT_ROOT / "src" / "Linking" / "text_list" / "icd10_text.json"

RXNORM_DB_PATH = PROJECT_ROOT / "data" / "db" / "RxNorm" / "rxnorm.db"
RXNORM_TEXT_PATH = PROJECT_ROOT / "src" / "Linking" / "text_list" / "rxnorm_text.json"
PROGRESS_INTERVAL = 25_000

def check_exist(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Path does not exist: {path}"
        )
    
def dump_compact_records(data, file, indent=0):
    """Write a list of flat dicts as JSON, one object per line."""
    pad = " " * indent
    file.write("[\n")
    total = len(data)
    for i, entity in enumerate(data):
        line = json.dumps(entity, ensure_ascii=False)
        comma = "," if i < total - 1 else ""
        file.write(f"{pad}{line}{comma}\n")
    file.write("]\n")


def normalize_for_fuzzy(text: str) -> str:
    """Chuẩn hóa code trước khi retrieve"""
    if not isinstance(text, str):
        return ""

    # Chuẩn hóa Unicode
    text = unicodedata.normalize("NFKC", text)

    # Chuyển toàn bộ chữ hoa thành chữ thường
    text = text.lower().strip()

    # Chuẩn hóa đơn vị
    text = re.sub(r"\bmilligrams?\b", "mg", text)
    text = re.sub(r"\bmilliliters?\b", "ml", text)
    text = re.sub(r"\bmicrograms?\b", "mcg", text)

    # 10mg -> 10 mg
    text = re.sub(
        r"(\d)\s*(mg|ml|mcg|g)\b",
        r"\1 \2",
        text,
    )

    # Thay dấu gạch dưới bằng khoảng trắng
    text = text.replace("_", " ")

    # Bỏ dấu câu, nhưng giữ dấu chấm thập phân và %
    text = re.sub(r"[^\w\s.%]", " ", text)

    # Chuẩn hóa nhiều khoảng trắng thành một
    return " ".join(text.split())

def build_icd10_list():
    open(ICD10_COPY_PATH, "w").close()
    print(f"Loading ICD-10 records from {ICD10_SOURCE_PATH}")
    with open(ICD10_SOURCE_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)

    total = len(data)
    print(f"Loaded {total:,} ICD-10 records")
    for index, entity in enumerate(data, start=1):
        entity["text"] = normalize_for_fuzzy(entity["text"])
        if index % PROGRESS_INTERVAL == 0 or index == total:
            print(f"Normalized ICD-10: {index:,}/{total:,}")

    print(f"Writing ICD-10 records to {ICD10_COPY_PATH}")
    with open(ICD10_COPY_PATH, "w", encoding="utf-8") as file:
        dump_compact_records(data, file)
    print("Finished building ICD-10 text list")


def build_rxnorm_list():
    open(RXNORM_TEXT_PATH, "w").close()
    print(f"Connecting to RxNorm database at {RXNORM_DB_PATH}")
    with sqlite3.connect(RXNORM_DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        print("Querying RxNorm records...")
        cursor.execute("""
            SELECT DISTINCT RXCUI, STR
            FROM RXNCONSO
            WHERE LAT = 'ENG'
            AND SAB = 'RXNORM'
            AND SUPPRESS = 'N'
            AND RXCUI IS NOT NULL
            AND STR IS NOT NULL
            AND TRIM(STR) != ''
        """)

        data = [dict(row) for row in cursor.fetchall()]
        total = len(data)
        print(f"Loaded {total:,} RxNorm records")
        # Đổi tên key
        for index, entity in enumerate(data, start=1):
            entity["code"] = entity.pop("RXCUI")
            entity["text"] = normalize_for_fuzzy(entity.pop("STR"))
            if index % PROGRESS_INTERVAL == 0 or index == total:
                print(f"Normalized RxNorm: {index:,}/{total:,}")

        print(f"Writing RxNorm records to {RXNORM_TEXT_PATH}")
        with open(RXNORM_TEXT_PATH, "w", encoding="utf-8") as file:
            dump_compact_records(data, file)
        print("Finished building RxNorm text list")

if __name__ == "__main__":

    check_exist(ICD10_SOURCE_PATH)
    check_exist(RXNORM_DB_PATH)
    ICD10_COPY_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Building icd10 list")
    build_icd10_list()
    print(f"Building rxnorm list")
    build_rxnorm_list()
