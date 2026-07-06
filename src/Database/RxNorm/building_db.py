"""
File này sẽ lấy các file dữ liệu dạng RRF của 
RxNorm để nhét vào rxnorm.db
"""

import sqlite3
from pathlib import Path

RXNORM_DIR = Path("rxnorm")          # folder containing .RRF files
DB_PATH = Path("rxnorm.db")
# Insert theo batch vì nhiều cột quá
BATCH_SIZE = 50_000
conn = sqlite3.connect(DB_PATH)

def create_table():
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS RXNCONSO (
        RXCUI TEXT,
        LAT TEXT,
        TS TEXT,
        LUI TEXT,
        STT TEXT,
        SUI TEXT,
        ISPREF TEXT,
        RXAUI TEXT,
        SAUI TEXT,
        SCUI TEXT,
        SDUI TEXT,
        SAB TEXT,
        TTY TEXT,
        CODE TEXT,
        STR TEXT,
        SRL TEXT,
        SUPPRESS TEXT,
        CVF TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS RXNREL (
        RXCUI1 TEXT,
        RXAUI1 TEXT,
        STYPE1 TEXT,
        REL TEXT,
        RXCUI2 TEXT,
        RXAUI2 TEXT,
        STYPE2 TEXT,
        RELA TEXT,
        RUI TEXT,
        SRUI TEXT,
        SAB TEXT,
        SL TEXT,
        DIR TEXT,
        RG TEXT,
        SUPPRESS TEXT,
        CVF TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS RXNSAT (
        RXCUI TEXT,
        LUI TEXT,
        SUI TEXT,
        RXAUI TEXT,
        STYPE TEXT,
        CODE TEXT,
        ATUI TEXT,
        SATUI TEXT,
        ATN TEXT,
        SAB TEXT,
        ATV TEXT,
        SUPPRESS TEXT,
        CVF TEXT
    )
    """)

    conn.commit()

def read_row(file_path, expected_cols):
    """
    Các file RRF được phân tách bằng dấu gạch đứng `|` và thường kết thúc bằng một dấu `|` ở cuối dòng. 
    Ví dụ: `col1|col2|col3|` Sau khi dùng `split('|')`, phần tử cuối cùng 
    thường sẽ là chuỗi rỗng `''`, nên cần loại bỏ nó.
    """
    with open(file_path,"r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            if parts and parts[-1] == "": 
                parts = parts[:-1] 

            if len(parts) != expected_cols: 
                print(f"Bỏ {file_path.name}: expected {expected_cols}, got {len(parts)}") 
                continue

            # Dùng yield vì không đọc toàn bộ file vào RAM cùng lúc
            yield parts 
            # VD output: ["C0000005", "ENG", "P", "L0000005"]

def insert_rrf():
    pass


def create_indexes():
        """
        Thêm index để truy vấn nhanh hơn
        """
        cur = conn.cursor()

        cur.execute("CREATE INDEX IF NOT EXISTS idx_rxnconso_rxcui ON RXNCONSO(RXCUI)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rxnconso_str ON RXNCONSO(STR)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rxnconso_sab ON RXNCONSO(SAB)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rxnconso_tty ON RXNCONSO(TTY)")

        cur.execute("CREATE INDEX IF NOT EXISTS idx_rxnrel_rxcui1 ON RXNREL(RXCUI1)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rxnrel_rxcui2 ON RXNREL(RXCUI2)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rxnrel_rela ON RXNREL(RELA)")

        cur.execute("CREATE INDEX IF NOT EXISTS idx_rxnsat_rxcui ON RXNSAT(RXCUI)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rxnsat_atn ON RXNSAT(ATN)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_rxnsat_sab ON RXNSAT(SAB)")

        conn.commit()

def build_db_rxnorm():
    if DB_PATH.exists():
        DB_PATH.unlink()

    

if __name__ == "__main__":
    build_db_rxnorm()
