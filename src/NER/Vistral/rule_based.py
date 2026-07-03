"""
Hard code rule base để phân loại trường text và type.
Mỗi file input sẽ là một output file.
"""

import json
from langchain_community.document_loaders import TextLoader
from pathlib import Path
import re
from dataclasses import dataclass

# Hard coded rule based_________________________________________________


TXT_DATA_DIR = Path(__file__).resolve().parent / "txt_data"


def load_txt_data(filename: str) -> list[str]:
    """Load one term per line from txt_data, skipping blank lines and comments."""
    path = TXT_DATA_DIR / filename
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


# 5 type hợp lệ theo đề bài. Mọi entity ngoài 5 type này sẽ bị loại.
ALLOWED_TYPES = {
    "TRIỆU_CHỨNG",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
    "CHẨN_ĐOÁN",
    "THUỐC",
}


# Entity trung gian do rule-based extractor sinh ra.
# source dùng để biết entity đến từ rule nào, giúp debug/merge với VinStrat sau này.
@dataclass
class Entity:
    text: str
    type: str
    source: str = "rule"


# Dictionary rule-based. Mở rộng dữ liệu bằng cách thêm dòng vào file trong txt_data.
DRUG_NAMES = load_txt_data("drug_names.txt")
SYMPTOM_TERMS = load_txt_data("symptom_terms.txt")
DIAGNOSIS_TERMS = load_txt_data("diagnosis_terms.txt")
LAB_NAMES = load_txt_data("lab_names.txt")
BLACKLIST_TERMS = set(load_txt_data("blacklist_terms.txt"))
SYMPTOM_HEADERS = load_txt_data("symptom_headers.txt")
DIAGNOSIS_HEADERS = load_txt_data("diagnosis_headers.txt")
DRUG_HEADERS = load_txt_data("drug_headers.txt")


def normalize_text(s: str) -> str:
    """Chuẩn hóa khoảng trắng và bỏ dấu câu thừa ở hai đầu entity."""
    return re.sub(r"\s+", " ", s.strip(" \t\n\r:;,.()[]-"))


def add_entity(entities, text, type_, source="rule"):
    """Validate entity trước khi thêm vào danh sách."""
    text = normalize_text(text)
    if not text:
        return

    # Chỉ giữ type nằm trong schema của đề.
    if type_ not in ALLOWED_TYPES:
        return

    if text.lower() in BLACKLIST_TERMS:
        return

    entities.append(Entity(text=text, type=type_, source=source))


def extract_drugs(text: str):
    """Bắt thuốc bằng dictionary + pattern liều/cách dùng."""
    entities = []

    # Ví dụ match:
    # - aspirin
    # - aspirin 325mg
    # - metoprolol 25mg po bid
    drug_pattern = (
        r"\b("
        + "|".join(map(re.escape, DRUG_NAMES))
        + r")\b"
        r"(?:\s+\d+(?:[.,]\d+)?\s*(?:mg|ml|mcg|g))?"
        r"(?:\s+(?:po|iv|im|bid|tid|qid|daily|q\d+h|prn|x\s*\d+))*"
    )

    for m in re.finditer(drug_pattern, text, flags=re.IGNORECASE):
        add_entity(entities, m.group(0), "THUỐC")

    return entities


def extract_labs(text: str):
    """Bắt tên xét nghiệm và kết quả xét nghiệm dạng số."""
    entities = []

    lab_pattern = r"\b(" + "|".join(map(re.escape, LAB_NAMES)) + r")\b"
    value_pattern = r"[-+]?\d+(?:[,.]\d+)?%?"

    # Ví dụ match:
    # - ast là 319
    # - wbc 11.6
    # - bilirubin toàn phần (tbili) là 2.4
    pattern = rf"({lab_pattern})(?:\s*\([^)]+\))?\s*(?:là|:|=)?\s*({value_pattern})"

    for m in re.finditer(pattern, text, flags=re.IGNORECASE):
        lab_name = m.group(1)
        lab_value = m.group(3)

        add_entity(entities, lab_name, "TÊN_XÉT_NGHIỆM")
        add_entity(entities, lab_value, "KẾT_QUẢ_XÉT_NGHIỆM")

    # Bắt tên xét nghiệm nếu nó xuất hiện riêng lẻ, không có value đi kèm.
    for m in re.finditer(lab_pattern, text, flags=re.IGNORECASE):
        add_entity(entities, m.group(0), "TÊN_XÉT_NGHIỆM")

    return entities


def extract_by_dictionary(text: str, terms, type_):
    """Bắt symptom/diagnosis bằng dictionary exact match."""
    entities = []

    # Sort cụm dài trước để ưu tiên "xơ gan do rượu" hơn "xơ gan".
    for term in sorted(terms, key=len, reverse=True):
        pattern = r"(?<!\w)" + re.escape(term) + r"(?!\w)"
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            add_entity(entities, m.group(0), type_)

    return entities


def extract_by_sections(text: str):
    """Dựa vào header section để gán type cho các bullet bên dưới."""
    entities = []
    lines = text.splitlines()

    current_section = None

    for line in lines:
        raw = line.strip()
        low = raw.lower()

        if not raw:
            continue

        # Khi gặp header, đổi context hiện tại.
        if any(h in low for h in SYMPTOM_HEADERS):
            current_section = "TRIỆU_CHỨNG"
            continue

        if any(h in low for h in DIAGNOSIS_HEADERS):
            current_section = "CHẨN_ĐOÁN"
            continue

        if any(h in low for h in DRUG_HEADERS):
            current_section = "THUỐC"
            continue

        # Chỉ lấy bullet ngắn trong section để tránh nuốt cả câu dài.
        if raw.startswith("-") and current_section:
            item = normalize_text(raw.lstrip("-"))

            # Dòng dài thường chứa nhiều ý, để VinStrat xử lý tốt hơn.
            if len(item.split()) <= 8:
                add_entity(entities, item, current_section, source="section_rule")

    return entities


def deduplicate(entities):
    """Gộp trùng theo text lowercase + type."""
    seen = set()
    output = []

    for e in entities:
        key = (e.text.lower(), e.type)
        if key in seen:
            continue

        seen.add(key)
        output.append({
            "text": e.text,
            "type": e.type,
            "source": e.source,
        })

    return output


def rule_based_ner(text: str):
    """Chạy toàn bộ rule extractor và trả về list entity đã deduplicate."""
    entities = []

    entities.extend(extract_drugs(text))
    entities.extend(extract_labs(text))
    entities.extend(extract_by_dictionary(text, SYMPTOM_TERMS, "TRIỆU_CHỨNG"))
    entities.extend(extract_by_dictionary(text, DIAGNOSIS_TERMS, "CHẨN_ĐOÁN"))
    entities.extend(extract_by_sections(text))

    return deduplicate(entities)


# Lưu output_________________________________________________________________________________
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[3]
    data_path = base_dir / "data" / "test"
    save_path = Path(__file__).resolve().parent / "output" / "rule_based"

    save_path.mkdir(parents=True, exist_ok=True)

    for txt_file in sorted(data_path.glob("*.txt"), key=lambda p: int(p.stem)):
        loader = TextLoader(str(txt_file), encoding="utf-8")
        docs = loader.load()

        text = docs[0].page_content
        results = rule_based_ner(text)

        output_file = save_path / f"{txt_file.stem}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Saved: {output_file}")
