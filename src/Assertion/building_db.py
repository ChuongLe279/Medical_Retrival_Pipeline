"""
Xây dựng các file JSON chứa câu mẫu assertion để đưa vào Chroma.

Mỗi câu đều chứa placeholder [ENTITY]. Khi xử lý test data, entity đích
cũng phải được thay bằng đúng placeholder này trước khi embedding.
"""
from pathlib import Path
import json
import re
import unicodedata


# =========================================================
# CONFIG
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "src" / "Assertion" / "data" / "json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ADD_UNACCENTED = True
ENTITY_TOKEN = "[ENTITY]"
PERIOD_TOKEN = "[PERIOD]"


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def normalize_text(text: str) -> str:
    """Chuẩn hóa phrase trước khi lưu."""
    text = unicodedata.normalize("NFKC", text)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    # Placeholder là token có phân biệt hoa/thường trong dữ liệu đầu ra.
    text = text.replace(PERIOD_TOKEN.lower(), PERIOD_TOKEN)
    return text


def remove_accents(text: str) -> str:
    """Chuyển: 'không ghi nhận' -> 'khong ghi nhan'."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        char for char in text
        if unicodedata.category(char) != "Mn"
    )
    return text.replace("đ", "d").replace("Đ", "D")


def prepare_phrases(phrases):
    """Chuẩn hóa, thêm bản không dấu và loại trùng."""
    results = set()

    for phrase in phrases:
        phrase = normalize_text(phrase)

        if not phrase:
            continue

        results.add(phrase)

        if ADD_UNACCENTED:
            results.add(remove_accents(phrase))

    return sorted(results, key=lambda x: (-len(x), x))


def attach_entity(phrase: str, cue_position: str) -> str:
    """Gắn [ENTITY] đúng phía so với assertion cue."""
    phrase = phrase.strip().rstrip(".!?;,:")

    if cue_position == "left":
        text = f"{phrase} {ENTITY_TOKEN}."
    elif cue_position == "right":
        text = f"{ENTITY_TOKEN} {phrase}."
    else:
        raise ValueError(
            "cue_position must be either 'left' or 'right'"
        )

    return text


def build_output(label, left_high=None, right_high=None,
                 left_expanded=None, right_expanded=None):
    """
    Tạo record sẵn sàng để đưa vào Chroma.

    Output item:
    {
        "id": "isFamily_000001",
        "text": "mẹ bệnh nhân mắc [ENTITY].",
        "metadata": {
            "assertion": "isFamily",
            "precision": "high",
            "cue_position": "left"
        }
    }
    """
    records_by_text = {}

    groups = [
        (left_high, "high", "left"),
        (right_high, "high", "right"),
        (left_expanded, "expanded", "left"),
        (right_expanded, "expanded", "right"),
    ]

    # Xử lý high trước. Nếu một câu xuất hiện ở cả high và expanded,
    # metadata high sẽ được giữ lại.
    for phrases, precision, cue_position in groups:
        for phrase in prepare_phrases(phrases or []):
            text = attach_entity(phrase, cue_position)

            records_by_text.setdefault(
                text,
                {
                    "text": text,
                    "metadata": {
                        "assertion": label,
                        "precision": precision,
                        "cue_position": cue_position,
                        "entity_token": ENTITY_TOKEN,
                    },
                },
            )

    records = sorted(
        records_by_text.values(),
        key=lambda item: item["text"],
    )

    for index, record in enumerate(records, start=1):
        record["id"] = f"{label}_{index:06d}"

    return records


def to_chroma_batch(records):
    """Chuyển record JSON thành arguments cho collection.add()."""
    return {
        "ids": [record["id"] for record in records],
        "documents": [record["text"] for record in records],
        "metadatas": [record["metadata"] for record in records],
    }


def save_json(filename, data):
    path = OUTPUT_DIR / filename

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    count = len(data)

    print(f"Created: {path} — {count:,} phrases")


# =========================================================
# 1. isFamily
# =========================================================

family_relations_high = [
    "cha",
    "bố",
    "ba",
    "mẹ",
    "má",
    "cha ruột",
    "bố ruột",
    "ba ruột",
    "mẹ ruột",
    "ông nội",
    "ông ngoại",
    "bà nội",
    "bà ngoại",
    "anh trai",
    "chị gái",
    "em trai",
    "em gái",
    "con trai",
    "con gái",
    "con ruột",
    "chồng",
    "vợ",
    "cô ruột",
    "dì ruột",
    "chú ruột",
    "bác ruột",
    "cậu ruột",
    "anh chị em ruột",
    "chị em ruột",
    "anh em ruột",
]

family_relations_expanded = [
    "cha mẹ",
    "bố mẹ",
    "ba mẹ",
    "ông bà",
    "anh chị em",
    "anh em",
    "chị em",
    "con",
    "cô",
    "dì",
    "chú",
    "bác",
    "cậu",
    "người thân",
    "thân nhân",
    "họ hàng",
    "gia đình",
    "gia tộc",
    "bên nội",
    "bên ngoại",
    "họ nội",
    "họ ngoại",
    "người cùng huyết thống",
    "thành viên trong gia đình",
]

family_verbs = [
    "bị",
    "mắc",
    "có",
    "được chẩn đoán",
    "được xác định mắc",
    "đang mắc",
    "đang điều trị",
    "từng bị",
    "từng mắc",
    "đã bị",
    "đã mắc",
    "có tiền sử",
    "ghi nhận",
    "được ghi nhận",
]

patient_references = [
    "bệnh nhân",
    "người bệnh",
    "bn",
    "bệnh nhi",
    "sản phụ",
]

family_left_high = set()
family_right_high = set()
family_left_expanded = set()
family_right_expanded = set()

# Ví dụ:
# mẹ bị...
# mẹ bệnh nhân mắc...
# mẹ của bệnh nhân có tiền sử...
for relation in family_relations_high:
    for verb in family_verbs:
        family_left_high.add(f"{relation} {verb}")

        for patient in patient_references:
            family_left_high.add(f"{relation} {patient} {verb}")
            family_left_high.add(
                f"{relation} của {patient} {verb}"
            )

    # Entity đứng trước:
    # tăng huyết áp ở mẹ bệnh nhân
    family_right_high.update([
        f"ở {relation}",
        f"ở người {relation}",
        f"trên người {relation}",
        f"ghi nhận ở {relation}",
    ])

    for patient in patient_references:
        family_right_high.update([
            f"ở {relation} {patient}",
            f"ở {relation} của {patient}",
            f"trên {relation} của {patient}",
        ])

for relation in family_relations_expanded:
    for verb in family_verbs:
        family_left_expanded.add(f"{relation} {verb}")

        for patient in patient_references:
            family_left_expanded.add(
                f"{relation} của {patient} {verb}"
            )

family_history_starters = [
    "tiền sử gia đình",
    "tiền căn gia đình",
    "bệnh sử gia đình",
    "gia đình có tiền sử",
    "gia đình ghi nhận",
    "trong gia đình có",
    "trong gia đình ghi nhận",
    "người thân có tiền sử",
    "họ hàng có tiền sử",
    "gia tộc có tiền sử",
    "phía gia đình",
    "phía bên nội",
    "phía bên ngoại",
    "bên nội có người",
    "bên ngoại có người",
    "họ nội có người",
    "họ ngoại có người",
]

for starter in family_history_starters:
    family_left_high.add(starter)

    for verb in family_verbs:
        family_left_high.add(f"{starter} {verb}")

    for relation in family_relations_high:
        for verb in family_verbs:
            family_left_expanded.add(
                f"{starter} {relation} {verb}"
            )

family_right_high.update([
    "trong gia đình",
    "ở người thân",
    "thuộc tiền sử gia đình",
    "về phía gia đình",
    "về phía bên nội",
    "về phía bên ngoại",
])

is_family = build_output(
    label="isFamily",
    left_high=family_left_high,
    right_high=family_right_high,
    left_expanded=family_left_expanded,
    right_expanded=family_right_expanded,
)


# =========================================================
# 2. isHistorical
# =========================================================

historical_left_high = {
    "tiền sử",
    "tiền căn",
    "có tiền sử",
    "có tiền căn",
    "trong tiền sử",
    "ghi nhận tiền sử",
    "bệnh sử ghi nhận",
    "trước đây bị",
    "trước đây mắc",
    "trước đó bị",
    "trước đó mắc",
    "đã từng bị",
    "đã từng mắc",
    "đã từng được chẩn đoán",
    "từng bị",
    "từng mắc",
    "từng được chẩn đoán",
    "đã có lần bị",
    "đã có lần mắc",
    "hồi trước bị",
    "hồi trước mắc",
    "trong quá khứ bị",
    "trong quá khứ mắc",
    "lúc trước bị",
    "lúc trước mắc",
    "khi còn nhỏ bị",
    "khi còn nhỏ mắc",
    "thời thơ ấu bị",
    "thời thơ ấu mắc",
    "khi còn bé bị",
    "khi còn bé mắc",
    "trước khi nhập viện đã bị",
    "trước khi nhập viện đã mắc",
    "trước lần nhập viện này bị",
    "trước lần nhập viện này mắc",
}

historical_right_high = {
    "trước đây",
    "trước đó",
    "trong quá khứ",
    "từ trước",
    "từ trước đến nay",
    "đã khỏi",
    "đã hết",
    "đã ổn định",
    "đã hồi phục",
    "đã phục hồi",
    "đã được điều trị",
    "đã điều trị khỏi",
    "đã điều trị ổn",
    "đã được chữa khỏi",
    "không còn triệu chứng",
    "hiện đã khỏi",
    "hiện đã ổn định",
    "thuộc tiền sử",
    "thuộc tiền căn",
}

historical_left_expanded = {
    "hồi nhỏ",
    "hồi bé",
    "hồi trẻ",
    "lúc nhỏ",
    "lúc bé",
    "lúc trẻ",
    "nhiều năm trước",
    "nhiều tháng trước",
    "một thời gian trước",
    "từ lâu",
    "từ rất lâu",
    "đợt trước",
    "lần trước",
    "ở lần nhập viện trước",
    "trong đợt nhập viện trước",
    "trong lần điều trị trước",
    "sau một đợt",
    "đã có",
    "đã bị",
    "đã mắc",
    "đã điều trị",
    "đã phẫu thuật",
    "đã nhập viện vì",
}

historical_right_expanded = {
    "cũ",
    "trước kia",
    "hồi trước",
    "lúc trước",
    "nhiều năm về trước",
    "ở lần nhập viện trước",
    "trong đợt trước",
    "đã qua",
    "đã giải quyết",
    "đã điều trị trước đây",
}

time_units = [
    "ngày",
    "tuần",
    "tháng",
    "năm",
]

# Mọi giá trị thời gian dạng số dùng chung [PERIOD]. Sinh trực tiếp token
# thay vì tạo 1..100 rồi mới thay thế để không tạo hàng loạt record trùng.
# Ví dụ: "cách đây 3 năm" -> "cách đây [PERIOD] năm".
for unit in time_units:
    expressions = [
        f"cách đây {PERIOD_TOKEN} {unit}",
        f"{PERIOD_TOKEN} {unit} trước",
        f"từ {PERIOD_TOKEN} {unit} trước",
        f"khoảng {PERIOD_TOKEN} {unit} trước",
        f"vào {PERIOD_TOKEN} {unit} trước",
        f"đã {PERIOD_TOKEN} {unit}",
    ]

    historical_left_high.update(expressions)
    historical_right_high.update(expressions)

    for expression in expressions:
        historical_left_expanded.update([
            f"{expression} bị",
            f"{expression} mắc",
            f"{expression} được chẩn đoán",
            f"{expression} được điều trị",
            f"{expression} nhập viện vì",
        ])

year_words = [
    "một",
    "hai",
    "ba",
    "bốn",
    "năm",
    "sáu",
    "bảy",
    "tám",
    "chín",
    "mười",
    "vài",
    "nhiều",
]

for word in year_words:
    for unit in time_units:
        historical_left_high.update([
            f"cách đây {word} {unit}",
            f"{word} {unit} trước",
            f"khoảng {word} {unit} trước",
        ])

        historical_right_high.update([
            f"cách đây {word} {unit}",
            f"{word} {unit} trước",
        ])

is_historical = build_output(
    label="isHistorical",
    left_high=historical_left_high,
    right_high=historical_right_high,
    left_expanded=historical_left_expanded,
    right_expanded=historical_right_expanded,
)


# =========================================================
# 3. isNegated
# =========================================================

negation_words = [
    "không",
    "chưa",
    "chẳng",
]

negation_verbs = [
    "bị",
    "mắc",
    "có",
    "ghi nhận",
    "phát hiện",
    "thấy",
    "xuất hiện",
    "biểu hiện",
    "than phiền",
    "được chẩn đoán",
    "có dấu hiệu",
    "có triệu chứng",
    "có bằng chứng",
    "có tiền sử",
]

negated_left_high = {
    "không",
    "chưa",
    "không hề",
    "chưa hề",
    "hoàn toàn không",
    "tuyệt đối không",
    "phủ nhận",
    "bệnh nhân phủ nhận",
    "người bệnh phủ nhận",
    "bn phủ nhận",
    "không ghi nhận",
    "chưa ghi nhận",
    "không phát hiện",
    "chưa phát hiện",
    "không tìm thấy",
    "không quan sát thấy",
    "không thấy",
    "chưa thấy",
    "không có dấu hiệu",
    "không có triệu chứng",
    "không có biểu hiện",
    "không có bằng chứng",
    "không có tiền sử",
    "không có tiền căn",
    "không nghĩ đến",
    "không phù hợp với",
    "không gợi ý",
    "không ủng hộ chẩn đoán",
    "không đủ tiêu chuẩn chẩn đoán",
    "không đáp ứng tiêu chuẩn chẩn đoán",
    "được loại trừ",
    "đã loại trừ",
    "đã được loại trừ",
    "xét nghiệm không phát hiện",
    "kết quả không phát hiện",
}

for negation in negation_words:
    for verb in negation_verbs:
        negated_left_high.add(f"{negation} {verb}")

negated_right_high = {
    "âm tính",
    "cho kết quả âm tính",
    "kết quả âm tính",
    "xét nghiệm âm tính",
    "không phát hiện",
    "không được phát hiện",
    "chưa phát hiện",
    "không ghi nhận",
    "chưa ghi nhận",
    "không thấy",
    "không tìm thấy",
    "không hiện diện",
    "không tồn tại",
    "được loại trừ",
    "đã được loại trừ",
    "đã loại trừ",
    "không được xác nhận",
    "không đủ tiêu chuẩn chẩn đoán",
    "không còn",
    "đã hết",
    "đã khỏi",
}

negated_left_expanded = {
    "gần như không",
    "hầu như không",
    "để loại trừ",
    "nhằm loại trừ",
    "chẩn đoán loại trừ",
    "ngoại trừ",
    "trừ",
}

negated_right_expanded = {
    "không điển hình",
    "không phù hợp",
}

negative_test_prefixes = [
    "xét nghiệm",
    "kết quả xét nghiệm",
    "kết quả kiểm tra",
    "kết quả sàng lọc",
    "test",
    "pcr",
    "nuôi cấy",
    "cấy máu",
    "cấy đàm",
    "cấy nước tiểu",
    "kháng nguyên",
    "kháng thể",
]

for prefix in negative_test_prefixes:
    negated_left_high.update([
        f"{prefix} âm tính với",
        f"{prefix} không phát hiện",
        f"{prefix} không ghi nhận",
        f"{prefix} cho kết quả âm tính với",
    ])

is_negated = build_output(
    label="isNegated",
    left_high=negated_left_high,
    right_high=negated_right_high,
    left_expanded=negated_left_expanded,
    right_expanded=negated_right_expanded,
)


# =========================================================
# 4. isUncertain
# =========================================================

uncertain_left_high = {
    "nghi",
    "nghi ngờ",
    "nghi mắc",
    "nghi bị",
    "nghĩ đến",
    "nghĩ nhiều đến",
    "hướng đến",
    "gợi ý",
    "có thể",
    "có khả năng",
    "nhiều khả năng",
    "khả năng cao",
    "không loại trừ",
    "chưa loại trừ",
    "cần loại trừ",
    "cần được loại trừ",
    "theo dõi",
    "theo dõi loại trừ",
    "cân nhắc",
    "cần cân nhắc",
    "chẩn đoán sơ bộ",
    "chẩn đoán ban đầu",
    "chẩn đoán tạm thời",
    "chẩn đoán phân biệt",
}

uncertain_right_high = {
    "được nghi ngờ",
    "được nghĩ đến",
    "được gợi ý",
    "có thể xảy ra",
    "có khả năng",
    "nhiều khả năng",
    "khả năng cao",
    "chưa thể loại trừ",
    "không thể loại trừ",
    "cần loại trừ",
    "cần được loại trừ",
    "cần theo dõi",
    "đang được theo dõi",
    "chờ xác nhận",
    "chưa được xác nhận",
}

uncertain_left_expanded = {
    "ít nghĩ",
    "ít nghĩ đến",
    "khó nghĩ đến",
    "khả năng thấp",
    "nguy cơ thấp",
    "không rõ",
    "không chắc có",
    "chưa xác định",
    "chưa xác nhận",
    "chưa được xác nhận",
    "có vẻ",
    "dường như",
    "có lẽ",
    "có khả năng là",
    "đặt vấn đề",
    "đặt ra khả năng",
    "chưa thể khẳng định",
    "chưa đủ cơ sở khẳng định",
    "cần làm thêm xét nghiệm xác định",
    "cần xét nghiệm xác nhận",
}

uncertain_right_expanded = {
    "ít khả năng",
    "khả năng thấp",
    "chưa xác nhận",
    "chưa được xác nhận",
    "chưa xác định",
    "chưa rõ",
    "không rõ ràng",
    "không chắc chắn",
    "còn nghi ngờ",
    "còn đang nghi ngờ",
    "mang tính nghi ngờ",
    "chỉ là chẩn đoán sơ bộ",
    "cần xác nhận thêm",
    "cần làm thêm xét nghiệm",
}

uncertain_subjects = [
    "bác sĩ",
    "nhóm điều trị",
    "kíp điều trị",
    "lâm sàng",
    "kết quả ban đầu",
    "hình ảnh",
    "xét nghiệm ban đầu",
]

uncertain_verbs = [
    "nghi ngờ",
    "nghĩ đến",
    "hướng đến",
    "gợi ý",
    "chưa loại trừ",
    "không loại trừ",
]

for subject in uncertain_subjects:
    for verb in uncertain_verbs:
        uncertain_left_expanded.add(f"{subject} {verb}")

is_uncertain = build_output(
    label="isUncertain",
    left_high=uncertain_left_high,
    right_high=uncertain_right_high,
    left_expanded=uncertain_left_expanded,
    right_expanded=uncertain_right_expanded,
)


# =========================================================
# SAVE JSON FILES
# =========================================================

if __name__ == "__main__":
    save_json("isFamily.json", is_family)
    save_json("isHistorical.json", is_historical)
    save_json("isNegated.json", is_negated)
    save_json("isUncertain.json", is_uncertain)
