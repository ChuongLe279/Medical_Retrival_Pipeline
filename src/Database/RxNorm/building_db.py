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

def read_row():
    pass

def insert_rrf():
    pass

def create_indexes():
    pass

def build_db_rxnorm():
    if DB_PATH.exists():
        DB_PATH.unlink()

    

if __name__ == "__main__":
    build_db_rxnorm()
