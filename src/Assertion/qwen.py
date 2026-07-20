"""
Thêm trường "qwen_assertion".

- Nếu rule-based đã phát hiện assertion hợp lệ:
  sử dụng trực tiếp kết quả rule-based, không gọi Qwen.
- Chỉ gọi Qwen3-8B khi assertion_result.tag == null.
- Theo quy định cuộc thi, chỉ có 3 assertion hợp lệ:
  isNegated, isFamily, isHistorical.
- Các trường hợp uncertain/suspected trả về None.
"""

import re
import sys
import json
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


JSON_PATH = PROJECT_ROOT / "data" / "output_merged"

SUPPORTED_ENTITY_TYPES = {
    "CHẨN_ĐOÁN",
    "THUỐC",
    "TRIỆU_CHỨNG",
}

# Theo quy định của cuộc thi
VALID_TAGS = {
    "isFamily",
    "isHistorical",
    "isNegated",
}

MODEL_ID = "Qwen/Qwen3-8B"

_model = None
_tokenizer = None


def get_model():
    """
    Load Qwen3-8B ở chế độ 4-bit.

    Model chỉ được load một lần rồi tái sử dụng cho tất cả file JSON.
    """
    global _model, _tokenizer

    if _model is None:
        print(f"Loading model: {MODEL_ID}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
        )

        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            attn_implementation="eager",
            trust_remote_code=True,
        )

        _model.eval()

        print("Model loaded successfully.")

    return _model, _tokenizer


SYSTEM_PROMPT = """
Bạn là bộ phân loại assertion trong văn bản bệnh án tiếng Việt.

NHIỆM VỤ:
Xác định trạng thái assertion của DUY NHẤT thực thể y khoa được chỉ định.

Thực thể thuộc một trong ba loại:
- CHẨN_ĐOÁN
- THUỐC
- TRIỆU_CHỨNG

Thực thể mục tiêu có thể được đánh dấu bằng [ENTITY] trong câu.
Chuỗi gốc của thực thể cũng được cung cấp riêng trong trường "Thực thể".

Bạn chỉ được trả về MỘT trong ba nhãn sau hoặc null.

1. isNegated

Chọn isNegated khi câu xác nhận rằng thực thể không tồn tại,
không xảy ra hoặc không được sử dụng đối với chủ thể đang được nhắc đến.

Các dấu hiệu thường gặp:
- không
- không có
- không ghi nhận
- không kèm
- không thấy
- phủ nhận
- chưa từng
- không dùng
- không sử dụng
- không phát hiện

Ví dụ:
- "Bệnh nhân không sốt."
- "Không ghi nhận khó thở."
- "Bệnh nhân không sử dụng aspirin."

2. isFamily

Chọn isFamily khi thực thể thuộc về người thân hoặc thành viên gia đình
của bệnh nhân, không phải bản thân bệnh nhân.

Các dấu hiệu thường gặp:
- cha, bố
- mẹ
- anh, chị, em
- con
- ông, bà
- cô, dì, chú, bác
- gia đình
- người thân
- người nhà

Ví dụ:
- "Mẹ bệnh nhân bị ung thư vú."
- "Gia đình có người mắc đái tháo đường."

3. isHistorical

Chọn isHistorical khi thực thể được trình bày như một phần của tiền sử,
hoặc là một sự kiện, triệu chứng hay điều trị xảy ra trước đây,
không phải biểu hiện hoặc điều trị hiện tại.

Bao gồm:
- Bệnh được ghi rõ trong tiền sử.
- Bệnh hoặc triệu chứng từng xảy ra trước đây và hiện không còn.
- Thuốc đã từng sử dụng.
- Thuốc đã ngưng.
- Thuốc thuộc danh sách thuốc trước nhập viện.
- Chẩn đoán hoặc điều trị cũ được nhắc lại như thông tin quá khứ.

Các dấu hiệu thường gặp:
- tiền sử
- trước đây
- từng
- cách đây
- đã ngưng
- đã điều trị
- trước nhập viện
- hồi nhỏ
- nhiều năm trước
- hiện đã hết

4. null

Trả về null nếu không có nhãn hợp lệ nào ở trên áp dụng.

Bao gồm:
- Thực thể đang hiện diện ở bệnh nhân hiện tại.
- Thuốc đang được bệnh nhân sử dụng hiện tại.
- Thuốc đang được kê hoặc chỉ định hiện tại.
- Không có đủ bằng chứng để gán assertion.
- Thực thể chỉ được mô tả là nghi ngờ.
- Thực thể chỉ được mô tả là có thể xảy ra.
- Thực thể đang được theo dõi.
- Thực thể cần được loại trừ.
- Chẩn đoán chưa chắc chắn.

Theo quy định cuộc thi, isUncertain KHÔNG phải nhãn assertion hợp lệ.
Vì vậy, các trường hợp nghi ngờ, theo dõi, có thể hoặc cần loại trừ
phải trả về null.

QUY TẮC BẮT BUỘC:

1. Chỉ phân loại thực thể mục tiêu được cung cấp.
Không phân loại các thực thể khác xuất hiện trong câu.

2. Chỉ dựa vào bằng chứng được viết rõ trong câu.
Không suy diễn bằng kiến thức y khoa.

3. Từ "đã" đứng một mình không đủ để kết luận isHistorical.

Ví dụ:
"Bệnh nhân đã được chẩn đoán viêm phổi khi nhập viện"
không nhất thiết là isHistorical nếu viêm phổi vẫn là tình trạng hiện tại.

Để chọn isHistorical, cần có ngữ cảnh quá khứ rõ ràng như:
"tiền sử", "trước đây", "trước nhập viện", "đã ngưng",
"cách đây", "từng mắc" hoặc "hiện đã hết".

4. Một bệnh mạn tính đang được điều trị hiện tại không tự động là
isHistorical, trừ khi câu đặt bệnh đó trong phần tiền sử hoặc mô tả
rõ ràng như thông tin quá khứ.

5. Nếu nhiều assertion đồng thời áp dụng nhưng chỉ được chọn một,
sử dụng thứ tự ưu tiên:

isNegated > isFamily > isHistorical

6. Không bao giờ trả về isUncertain.

7. Không giải thích.
Không sử dụng Markdown.
Không thêm bất kỳ nội dung nào ngoài JSON.

Chỉ trả về đúng một trong các JSON sau:

{"tag": "isNegated"}
{"tag": "isFamily"}
{"tag": "isHistorical"}
{"tag": null}
""".strip()


FEWSHOT = [
    {
        "role": "user",
        "content": (
            'Câu: "Bệnh nhân không có [ENTITY] kèm theo."\n'
            'Thực thể: "sốt"\n'
            'Loại: "TRIỆU_CHỨNG"'
        ),
    },
    {
        "role": "assistant",
        "content": '{"tag":"isNegated"}',
    },

    {
        "role": "user",
        "content": (
            'Câu: "Không ghi nhận [ENTITY] trong quá trình theo dõi."\n'
            'Thực thể: "khó thở"\n'
            'Loại: "TRIỆU_CHỨNG"'
        ),
    },
    {
        "role": "assistant",
        "content": '{"tag":"isNegated"}',
    },

    {
        "role": "user",
        "content": (
            'Câu: "Bệnh nhân không sử dụng [ENTITY]."\n'
            'Thực thể: "aspirin"\n'
            'Loại: "THUỐC"'
        ),
    },
    {
        "role": "assistant",
        "content": '{"tag":"isNegated"}',
    },

    {
        "role": "user",
        "content": (
            'Câu: "Mẹ bệnh nhân đang điều trị [ENTITY] tại bệnh viện khác."\n'
            'Thực thể: "ung thư vú"\n'
            'Loại: "CHẨN_ĐOÁN"'
        ),
    },
    {
        "role": "assistant",
        "content": '{"tag":"isFamily"}',
    },

    {
        "role": "user",
        "content": (
            'Câu: "Cha bệnh nhân có tiền sử [ENTITY]."\n'
            'Thực thể: "tăng huyết áp"\n'
            'Loại: "CHẨN_ĐOÁN"'
        ),
    },
    {
        "role": "assistant",
        "content": '{"tag":"isFamily"}',
    },

    {
        "role": "user",
        "content": (
            'Câu: "Tiền sử [ENTITY] được chẩn đoán 10 năm trước."\n'
            'Thực thể: "hen phế quản"\n'
            'Loại: "CHẨN_ĐOÁN"'
        ),
    },
    {
        "role": "assistant",
        "content": '{"tag":"isHistorical"}',
    },

    {
        "role": "user",
        "content": (
            'Câu: "Bệnh nhân từng bị [ENTITY] khi còn nhỏ."\n'
            'Thực thể: "co giật"\n'
            'Loại: "TRIỆU_CHỨNG"'
        ),
    },
    {
        "role": "assistant",
        "content": '{"tag":"isHistorical"}',
    },

    # Tương tự ví dụ thuốc trước nhập viện trong PDF
    {
        "role": "user",
        "content": (
            'Câu: "Danh sách thuốc trước nhập viện gồm [ENTITY]."\n'
            'Thực thể: "aspirin 81 mg po daily"\n'
            'Loại: "THUỐC"'
        ),
    },
    {
        "role": "assistant",
        "content": '{"tag":"isHistorical"}',
    },

    {
        "role": "user",
        "content": (
            'Câu: "Bệnh nhân đã ngưng [ENTITY] cách đây hai năm."\n'
            'Thực thể: "metformin"\n'
            'Loại: "THUỐC"'
        ),
    },
    {
        "role": "assistant",
        "content": '{"tag":"isHistorical"}',
    },

    {
        "role": "user",
        "content": (
            'Câu: "Hiện bệnh nhân sốt và có [ENTITY]."\n'
            'Thực thể: "khó thở"\n'
            'Loại: "TRIỆU_CHỨNG"'
        ),
    },
    {
        "role": "assistant",
        "content": '{"tag":null}',
    },

    {
        "role": "user",
        "content": (
            'Câu: "Bệnh nhân đang sử dụng [ENTITY] mỗi ngày."\n'
            'Thực thể: "amlodipine"\n'
            'Loại: "THUỐC"'
        ),
    },
    {
        "role": "assistant",
        "content": '{"tag":null}',
    },

    {
        "role": "user",
        "content": (
            'Câu: "Theo dõi [ENTITY], chưa đủ tiêu chuẩn chẩn đoán."\n'
            'Thực thể: "viêm phổi"\n'
            'Loại: "CHẨN_ĐOÁN"'
        ),
    },
    {
        "role": "assistant",
        "content": '{"tag":null}',
    },

    {
        "role": "user",
        "content": (
            'Câu: "Nghi ngờ [ENTITY], cần làm thêm xét nghiệm."\n'
            'Thực thể: "xơ gan"\n'
            'Loại: "CHẨN_ĐOÁN"'
        ),
    },
    {
        "role": "assistant",
        "content": '{"tag":null}',
    },

    {
        "role": "user",
        "content": (
            'Câu: "Không ai trong gia đình mắc [ENTITY]."\n'
            'Thực thể: "đái tháo đường"\n'
            'Loại: "CHẨN_ĐOÁN"'
        ),
    },
    {
        "role": "assistant",
        "content": '{"tag":"isNegated"}',
    },
]


def parse_model_output(raw: str) -> str | None:
    """
    Parse JSON do model sinh ra.

    Chỉ chấp nhận:
    - isNegated
    - isFamily
    - isHistorical
    - None
    """
    match = re.search(
        r'\{\s*"tag"\s*:\s*(?:"[^"]+"|null)\s*\}',
        raw,
    )

    if not match:
        print(
            f"    [WARN] Không parse được JSON từ output: "
            f"{raw!r} -> mặc định null"
        )
        return None

    json_text = match.group(0)

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        print(
            f"    [WARN] JSON không hợp lệ: "
            f"{json_text!r} -> mặc định null"
        )
        return None

    tag = parsed.get("tag")

    if tag is None:
        return None

    if tag not in VALID_TAGS:
        print(
            f"    [WARN] Tag không hợp lệ: "
            f"{tag!r} -> mặc định null"
        )
        return None

    return tag


def classify_single_tag(
    sentence: str,
    entity_text: str,
    entity_type: str,
) -> str | None:
    """
    Gọi Qwen3-8B để phân loại một assertion.

    Returns:
        "isNegated", "isFamily", "isHistorical" hoặc None.
    """
    model, tokenizer = get_model()

    user_message = (
        f'Câu: "{sentence}"\n'
        f'Thực thể: "{entity_text}"\n'
        f'Loại: "{entity_type}"'
    )

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + FEWSHOT
        + [{"role": "user", "content": user_message}]
    )

    prompt_text = tokenizer.apply_chat_template( # type: ignore
        messages, 
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer( # type: ignore
        [prompt_text], # type: ignore
        return_tensors="pt", # type: ignore
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate( # type: ignore
            **inputs,
            max_new_tokens=30,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id, # type: ignore
        )

    prompt_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][prompt_length:]

    raw_output = tokenizer.decode( # type: ignore
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return parse_model_output(raw_output)


def qwen_assertion(json_file: Path):
    """
    Thêm qwen_assertion cho từng entity.

    Logic:
    - Type không hỗ trợ:
        xóa qwen_assertion nếu có.
    - Rule-based có tag hợp lệ:
        dùng tag rule-based, không gọi Qwen.
    - Rule-based có isUncertain hoặc tag không hợp lệ:
        chuyển thành None, không gọi Qwen.
    - Rule-based tag là None:
        gọi Qwen.
    """
    with json_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            f"Nội dung {json_file.name} phải là một list entity."
        )

    qwen_call_count = 0
    rule_based_count = 0
    unsupported_count = 0

    for index, entity in enumerate(data):
        if not isinstance(entity, dict):
            print(
                f"    [WARN] Entity thứ {index} không phải dictionary."
            )
            continue

        entity_type = entity.get("type")

        if entity_type not in SUPPORTED_ENTITY_TYPES:
            entity.pop("qwen_assertion", None)
            unsupported_count += 1
            continue

        assertion_result = entity.get("assertion_result") or {}

        if not isinstance(assertion_result, dict):
            print(
                f"    [WARN] assertion_result của entity thứ {index} "
                f"không phải dictionary -> xem như null"
            )
            assertion_result = {}

        rule_based_tag = assertion_result.get("tag")

        # Rule-based đã phát hiện được một tag hợp lệ
        if rule_based_tag in VALID_TAGS:
            entity["qwen_assertion"] = rule_based_tag
            rule_based_count += 1
            continue

        # Rule-based trả về tag không được BTC hỗ trợ,
        # ví dụ isUncertain.
        # Không gọi Qwen vì điều kiện gọi Qwen chỉ là tag == null.
        if rule_based_tag is not None:
            print(
                f"    [WARN] Entity {index}: rule-based tag "
                f"{rule_based_tag!r} không hợp lệ theo quy định -> null"
            )
            entity["qwen_assertion"] = None
            continue

        sentence = entity.get("sentence")
        entity_text = entity.get("text")

        if not isinstance(sentence, str) or not sentence.strip():
            print(
                f"    [WARN] Entity {index}: thiếu sentence -> null"
            )
            entity["qwen_assertion"] = None
            continue

        if not isinstance(entity_text, str) or not entity_text.strip():
            print(
                f"    [WARN] Entity {index}: thiếu text -> null"
            )
            entity["qwen_assertion"] = None
            continue

        predicted_tag = classify_single_tag(
            sentence=sentence.strip(),
            entity_text=entity_text.strip(),
            entity_type=entity_type,
        )

        entity["qwen_assertion"] = predicted_tag
        qwen_call_count += 1

        print(
            f"    Entity {index}: "
            f"{entity_text!r} -> {predicted_tag!r}"
        )

    with json_file.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            default=lambda value: (
                value.item()
                if hasattr(value, "item")
                else str(value)
            ),
        )

    print(
        f"    Rule-based: {rule_based_count} | "
        f"Qwen calls: {qwen_call_count} | "
        f"Unsupported types: {unsupported_count}"
    )


def check_folder_exists(path: Path):
    """Kiểm tra folder đầu vào."""
    if not path.exists():
        raise FileNotFoundError(
            f"Folder does not exist: {path}"
        )

    if not path.is_dir():
        raise NotADirectoryError(
            f"Path is not a folder: {path}"
        )


def numeric_file_sort(path: Path):
    """
    Sắp xếp file JSON theo số nếu tên file là 1.json, 2.json,...

    File không có tên dạng số sẽ được đặt ở cuối.
    """
    if path.stem.isdigit():
        return 0, int(path.stem)

    return 1, path.stem


def main():
    check_folder_exists(JSON_PATH)

    json_files = sorted(
        JSON_PATH.glob("*.json"),
        key=numeric_file_sort,
    )

    if not json_files:
        print(f"Không tìm thấy file JSON trong: {JSON_PATH}")
        return

    print(f"Tìm thấy {len(json_files)} file JSON.")
    print("-" * 60)

    for file_index, file_path in enumerate(json_files, start=1):
        print(
            f"[{file_index}/{len(json_files)}] "
            f"Qwen assertion: {file_path.name}"
        )

        try:
            qwen_assertion(file_path)
        except Exception as error:
            print(
                f"    [ERROR] Không xử lý được "
                f"{file_path.name}: {error}"
            )

        print("-" * 60)


if __name__ == "__main__":
    main()