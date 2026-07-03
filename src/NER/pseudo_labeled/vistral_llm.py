"""
Dùng local vistral (model đã được pretrain cho dữ liệu y khoa)
Mỗi output là 1 file và sẽ được merged với rule_based
"""
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import TextLoader
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json


# Prompt để hỏi local llm_________________________________________________________________________________
template = """
Bạn là hệ thống trích xuất thực thể y tế tiếng Việt.

Nhiệm vụ:
Từ văn bản bệnh án đầu vào, hãy trích xuất các cụm từ y tế xuất hiện NGUYÊN VĂN trong input và phân loại chúng vào đúng type.

Chỉ được dùng 5 type sau:
- TRIỆU_CHỨNG
- TÊN_XÉT_NGHIỆM
- KẾT_QUẢ_XÉT_NGHIỆM
- CHẨN_ĐOÁN
- THUỐC

Bạn chỉ được trả về JSON object hợp lệ.
Không giải thích.
Không markdown.
Không dùng ```json.
Không thêm chữ trước hoặc sau JSON.

Output bắt buộc có đúng format:

{{
  "entities": [
    {{
      "text": "chuỗi y tế xuất hiện nguyên văn trong input",
      "type": "một trong 5 type hợp lệ"
    }}
  ]
}}

Nếu không có khái niệm y tế nào, trả về:

{{
  "entities": []
}}

Quy tắc bắt buộc:
- "text" phải được copy chính xác từ input.
- Không tự sửa lỗi chính tả.
- Không tự chuẩn hóa thuật ngữ.
- Không tự dịch sang tiếng Anh.
- Không thêm field khác ngoài "text" và "type".
- Không trả về "position".
- Không trích xuất tên khoa/phòng, ngày tháng, thông tin hành chính.
- Không trích xuất các cụm chung chung như "bệnh nhân", "nhập viện", "điều trị", "theo dõi", "kết quả", "triệu chứng".
- Nếu cùng một cụm xuất hiện nhiều lần trong input, hãy trả về nhiều object tương ứng.

Định nghĩa type:

1. THUỐC
Dùng cho tên thuốc hoặc cụm thuốc có liều lượng/cách dùng đi kèm.
Ví dụ:
- "amlodipine 10 mg po daily"
- "aspirin 81 mg po daily"
- "metoprolol 25mg po bid"
- "nystatin oral suspension 5 ml po qid:prn"

2. TRIỆU_CHỨNG
Dùng cho biểu hiện lâm sàng, dấu hiệu, triệu chứng, cảm giác bất thường của bệnh nhân.
Ví dụ:
- "ho"
- "sốt"
- "khó thở"
- "đau bụng"
- "đau ngực"
- "ý thức suy giảm"
- "buồn nôn"
- "chóng mặt"

Nếu triệu chứng bị phủ định như "không đau ngực", vẫn trích xuất "đau ngực".

3. CHẨN_ĐOÁN
Dùng cho tên bệnh, hội chứng, hoặc tình trạng bệnh lý đã được chẩn đoán hoặc nghi ngờ.
Ví dụ:
- "tăng huyết áp"
- "đái tháo đường"
- "viêm phổi"
- "suy tim"
- "xơ gan do rượu"
- "hội chứng não gan"

Phân biệt:
- "khó thở" là TRIỆU_CHỨNG.
- "viêm phổi" là CHẨN_ĐOÁN.
- "đau bụng" là TRIỆU_CHỨNG.
- "xơ gan do rượu" là CHẨN_ĐOÁN.
- "hội chứng não gan" là CHẨN_ĐOÁN.

4. TÊN_XÉT_NGHIỆM
Dùng cho tên xét nghiệm, chỉ số xét nghiệm, marker, hoặc thông số cận lâm sàng.
Ví dụ:
- "WBC"
- "NEUT%"
- "CRP"
- "creatinin"
- "glucose máu"
- "AST"
- "ALT"
- "bilirubin toàn phần"

5. KẾT_QUẢ_XÉT_NGHIỆM
Dùng cho giá trị hoặc mô tả kết quả xét nghiệm.
Ví dụ:
- "14,43"
- "76,4"
- "dương tính"
- "âm tính"
- "tăng"
- "giảm"
- "319"
- "690"

Quy tắc xét nghiệm:
- Trong "WBC 14,43", "WBC" là TÊN_XÉT_NGHIỆM, "14,43" là KẾT_QUẢ_XÉT_NGHIỆM.
- Trong "CRP tăng", "CRP" là TÊN_XÉT_NGHIỆM, "tăng" là KẾT_QUẢ_XÉT_NGHIỆM.
- Trong "ast là 319", "ast" là TÊN_XÉT_NGHIỆM, "319" là KẾT_QUẢ_XÉT_NGHIỆM.
- Không gộp tên xét nghiệm và kết quả xét nghiệm thành một entity.

Ví dụ 1:

Input:
"Danh sách thuốc trước nhập viện. 1. amlodipine 10 mg po daily 2. aspirin 81 mg po daily 3. metoprolol 25mg po bid điều trị ho."

Output:
{{
  "entities": [
    {{
      "text": "amlodipine 10 mg po daily",
      "type": "THUỐC"
    }},
    {{
      "text": "aspirin 81 mg po daily",
      "type": "THUỐC"
    }},
    {{
      "text": "metoprolol 25mg po bid",
      "type": "THUỐC"
    }},
    {{
      "text": "ho",
      "type": "TRIỆU_CHỨNG"
    }}
  ]
}}

Ví dụ 2:

Input:
"Các bệnh lý mãn tính: nghi ngờ xơ gan do rượu. Triệu chứng hiện tại: hội chứng não gan, ý thức suy giảm."

Output:
{{
  "entities": [
    {{
      "text": "xơ gan do rượu",
      "type": "CHẨN_ĐOÁN"
    }},
    {{
      "text": "hội chứng não gan",
      "type": "CHẨN_ĐOÁN"
    }},
    {{
      "text": "ý thức suy giảm",
      "type": "TRIỆU_CHỨNG"
    }}
  ]
}}

Ví dụ 3:

Input:
"Kết quả xét nghiệm ghi nhận WBC 14,43, NEUT% 76,4 và CRP tăng."

Output:
{{
  "entities": [
    {{
      "text": "WBC",
      "type": "TÊN_XÉT_NGHIỆM"
    }},
    {{
      "text": "14,43",
      "type": "KẾT_QUẢ_XÉT_NGHIỆM"
    }},
    {{
      "text": "NEUT%",
      "type": "TÊN_XÉT_NGHIỆM"
    }},
    {{
      "text": "76,4",
      "type": "KẾT_QUẢ_XÉT_NGHIỆM"
    }},
    {{
      "text": "CRP",
      "type": "TÊN_XÉT_NGHIỆM"
    }},
    {{
      "text": "tăng",
      "type": "KẾT_QUẢ_XÉT_NGHIỆM"
    }}
  ]
}}

Bây giờ hãy xử lý văn bản sau.

Input:
{input_text}

Output:
"""

vistral = ChatOllama(
    model="hf.co/Ethan2004/vistral-7b-medical-vi:Q4_K_M",
    temperature=0,
    format="json"
)
embedding_prompt = ChatPromptTemplate.from_template(template)

generate_emd_json = (
    embedding_prompt
    | vistral
    | StrOutputParser()
)

def parse_json_output(raw: str):
    data = json.loads(raw)

    if isinstance(data, dict):
        entities = data.get("entities", [])
    elif isinstance(data, list):
        entities = data
    else:
        entities = []

    allowed_types = {
        "TRIỆU_CHỨNG",
        "TÊN_XÉT_NGHIỆM",
        "KẾT_QUẢ_XÉT_NGHIỆM",
        "CHẨN_ĐOÁN",
        "THUỐC",
    }

    cleaned = []
    for item in entities:
        if not isinstance(item, dict):
            continue

        text = item.get("text")
        type_ = item.get("type")

        if not text or type_ not in allowed_types:
            continue

        cleaned.append({
            "text": text.strip(),
            "type": type_,
        })

    return cleaned

# Lưu output_________________________________________________________________________________
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[3]
    data_path = base_dir / "data" / "test"
    save_path = Path(__file__).resolve().parent / "output" / "vistral_output"

    for txt_file in sorted(data_path.glob("*.txt"), key=lambda p: int(p.stem)):
        loader = TextLoader(str(txt_file), encoding="utf-8")
        docs = loader.load()

        text = docs[0].page_content
        raw = generate_emd_json.invoke({
            "input_text": text
        })
        results = parse_json_output(raw)
        output_file = save_path / f"{txt_file.stem}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"Saved: {output_file}")
