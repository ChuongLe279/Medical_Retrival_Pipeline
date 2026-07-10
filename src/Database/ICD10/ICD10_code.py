"""
File này sẽ trích xuất code và text của dữ liệu icd10 và đưa nó về json
VD: A000    Cholera due to Vibrio cholerae 01, biovar cholerae
{
    "code": "A000",
    "text": "Cholera due to Vibrio cholerae 01, biovar cholerae"
}
Các chuẩn ICD10 khá đơn giản nên ta không cần phải dùng sqlite.
"""
from pathlib import Path
import re
import json

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ICD10_CODE_PATH = PROJECT_ROOT / "data" / "raw" / "ICD10" / "icd10cm-codes-2027.txt"
SAVE_PATH = PROJECT_ROOT / "data" / "db" / "ICD10" / "output.json"



if __name__ == "__main__":
    records = []

    with open(ICD10_CODE_PATH, "r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                code, text = line.split(maxsplit=1)
            except ValueError:
                print(f"Skipped invalid line {line_number}: {line}")
                continue

            records.append({
                "code": code,
                "text": text
            })

    with open(SAVE_PATH, "w", encoding="utf-8") as output_file:
        output_file.write("[\n")

        for index, record in enumerate(records):
            output_file.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            )

            if index < len(records) - 1:
                output_file.write(",")

            output_file.write("\n")

        output_file.write("]")

    print(f"Saved {len(records)} records to {SAVE_PATH}")
    # Saved 74879 records to D:\SIDE PROJECTS\Ontological Reasoning in Medical Knowledge Retrieval\data\db\ICD10\output.json
