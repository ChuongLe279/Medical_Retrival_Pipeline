"""
Chiến lược merge output của vistral và rule based
1. Nếu không overlap:
   giữ nguyên entity.

2. Nếu overlap cùng type + cùng span:
   merge thành 1 entity, source = "both".

3. Nếu overlap cùng type nhưng khác span:
   chọn theo TYPE_PRIORITY[type].
   Riêng TÊN_XÉT_NGHIỆM:
     - nếu một span là viết tắt chuẩn như ECG/WBC/CRP/HbA1c → rule tốt
     - nếu Vistral bắt tên đầy đủ hơn → cân nhắc chọn span dài hơn
   log lại nếu span lệch nhiều.

4. Nếu overlap khác type:
   chọn entity có source ưu tiên cao hơn theo type của chính entity đó.
   Luôn log conflict này để review thủ công.

5. Nếu source = section_rule:
   chỉ dùng khi không có rule hoặc vistral overlap tốt hơn.
   
"""

from pathlib import Path
import json
from bisect import bisect_right
BASE_DIR = Path(__file__).parent

input_dir = BASE_DIR.parents[2] / "data" / "test" 
vistral_output_path = BASE_DIR / "output" / "vistral_output"
rule_based_output_path = BASE_DIR / "output" / "rule_based"
merged_output_path = BASE_DIR / "output" / "merged_output"/ "merged_json"

normalized_rule_based_path = BASE_DIR / "output" / "merged_output"/ "normalized_rule_based"
normalized_vistral_path = BASE_DIR / "output" / "merged_output"/ "normalized_vistral"

vistral_folder = Path(vistral_output_path)
rule_based_folder = Path(rule_based_output_path)

merged_output_folder = Path(merged_output_path)
merged_output_folder.mkdir(parents=True, exist_ok=True)

normalized_rule_based_folder = Path(normalized_rule_based_path)
normalized_vistral_folder = Path(normalized_vistral_path)
normalized_rule_based_folder.mkdir(parents=True, exist_ok=True)
normalized_vistral_folder.mkdir(parents=True, exist_ok=True)


# Điểm ưu tiên khi nhiều source cùng bắt một span hoặc các span bị overlap.
# "both" nghĩa là cả rule-based và Vistral đều đồng ý, nên đáng tin nhất.
# "section_rule" yếu hơn vì nó chỉ suy luận từ header/bullet, dễ bắt cả câu rộng.
SOURCE_PRIORITY_BY_TYPE = {
    "THUỐC": {
        "both": 90,
        "rule": 70,
        "vistral": 65,
        "section_rule": 5,
        "unknown": 0,
    },
    "TÊN_XÉT_NGHIỆM": {
        "both": 90,
        "rule": 70,
        "vistral": 55,
        "section_rule": 5,
        "unknown": 0,
    },
    "KẾT_QUẢ_XÉT_NGHIỆM": {
        "both": 90,
        "rule": 70,
        "vistral": 55,
        "section_rule": 5,
        "unknown": 0,
    },
    "CHẨN_ĐOÁN": {
        "both": 90,
        "rule": 65,
        "vistral": 60,
        "section_rule": 15,
        "unknown": 0,
    },
    "TRIỆU_CHỨNG": {
        "both": 90,
        "vistral": 60,
        "rule": 55,
        "section_rule": 15,
        "unknown": 0,
    },
}


def normalize_entities(data):
    """Đổi position [start, end] thành start/end và bỏ entity không tìm được span."""
    normalized_data = []

    for obj in data:
        start, end = obj["position"]
        if start < 0 or end < 0:
            # Span -1 nghĩa là text không xuất hiện nguyên văn/linh hoạt trong input.
            continue

        new_obj = {
            "text": obj["text"],
            "type": obj["type"],
            "source": obj.get("source", "unknown"),
            "start": start,
            "end": end,
        }
        normalized_data.append(new_obj)

    return normalized_data


def source_parts(source):
    """Tách source dạng 'rule+vistral' thành set {'rule', 'vistral'}."""
    return set(str(source).split("+"))


def merge_sources(sources):
    """Gộp nhiều source thành một nhãn source dễ đọc."""
    parts = set()
    for source in sources:
        parts.update(source_parts(source))

    if "rule" in parts and "vistral" in parts:
        # Khi rule và Vistral cùng bắt một entity, ghi là both.
        return "both"

    return "+".join(sorted(parts))


def source_priority(entity):
    """Lấy điểm ưu tiên của source theo từng type entity."""
    type_priorities = SOURCE_PRIORITY_BY_TYPE.get(entity["type"], {})
    parts = source_parts(entity.get("source", "unknown"))

    if "rule" in parts and "vistral" in parts:
        return type_priorities.get("both", 0)

    return max((type_priorities.get(part, 0) for part in parts), default=0)


def entity_score(entity):
    """Tính điểm để chọn entity khi có duplicate/overlap."""
    span_len = entity["end"] - entity["start"]
    # Bonus nhẹ cho span dài hơn, nhưng giới hạn để source vẫn là yếu tố chính.
    length_bonus = min(span_len, 30)
    return source_priority(entity) * 100 + length_bonus


def deduplicate_exact_entities(entities):
    """Gộp các entity có cùng start/end/type thành một entity duy nhất."""
    grouped = {}

    for entity in entities:
        key = (entity["start"], entity["end"], entity["type"])
        grouped.setdefault(key, []).append(entity)

    deduplicated = []
    for group in grouped.values():
        best = max(
            group,
            key=lambda item: (
                source_priority(item),
                len(item["text"]),
                item["source"] != "section_rule",
            ),
        )
        deduplicated.append({
            "text": best["text"],
            "type": best["type"],
            # Giữ lại thông tin entity này đến từ những nguồn nào.
            "source": merge_sources(item["source"] for item in group),
            "start": best["start"],
            "end": best["end"],
        })

    return deduplicated


def select_non_overlapping_entities(entities):
    """
    Chọn tập entity có tổng điểm cao nhất nhưng không overlap.

    Đây là bài toán interval scheduling có trọng số:
    mỗi entity là một đoạn [start, end], điểm là entity_score(entity).
    """
    # Sort theo điểm kết thúc để tính dynamic programming.
    entities = sorted(
        entities,
        key=lambda item: (item["end"], item["start"], -entity_score(item)),
    )
    ends = [entity["end"] for entity in entities]

    # previous[i] là entity gần nhất đứng trước i mà không overlap với i.
    previous = [
        bisect_right(ends, entity["start"], hi=index) - 1
        for index, entity in enumerate(entities)
    ]

    # dp[i] là tổng điểm tốt nhất khi xét i entity đầu tiên.
    dp = [0] * (len(entities) + 1)
    take = [False] * len(entities)

    for index, entity in enumerate(entities, start=1):
        # Hai lựa chọn: lấy entity hiện tại hoặc bỏ nó.
        include_score = entity_score(entity) + dp[previous[index - 1] + 1]
        exclude_score = dp[index - 1]

        if include_score > exclude_score:
            dp[index] = include_score
            take[index - 1] = True
        else:
            dp[index] = exclude_score

    selected = []
    index = len(entities)
    while index > 0:
        # Truy vết lại các entity đã chọn từ bảng dp.
        if take[index - 1] and entity_score(entities[index - 1]) + dp[previous[index - 1] + 1] >= dp[index - 1]:
            selected.append(entities[index - 1])
            index = previous[index - 1] + 1
        else:
            index -= 1

    return sorted(selected, key=lambda item: (item["start"], item["end"], item["type"]))


def merge_entities(*entity_lists):
    """Merge nhiều list entity thành một list sạch duplicate và không overlap."""
    entities = []
    for entity_list in entity_lists:
        entities.extend(entity_list)

    deduplicated = deduplicate_exact_entities(entities)
    return select_non_overlapping_entities(deduplicated)


"""Sort file json lại theo tên"""
vistral_files = sorted(
    vistral_folder.glob("*.json"),
    key=lambda p: int(p.stem)
)

"""Chuẩn hóa file json. Cho trường position về dạng start end"""
for json_vistral_file in vistral_files:
    name = json_vistral_file.name
    json_vistral_file = vistral_folder / name
    json_rule_based_file = rule_based_folder / name

    # Chuẩn hóa vistral data
    with open(json_vistral_file, "r", encoding="utf-8-sig") as f:
        vistral_data = json.load(f)

    normalized_data = normalize_entities(vistral_data)

    output_file = normalized_vistral_folder / name
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(normalized_data, f, ensure_ascii=False, indent=4)
    print(f"Normalized vistral {name}")

    # Chuẩn hóa rule based data
    with open(json_rule_based_file, "r", encoding="utf-8-sig") as f:
        rule_based_data = json.load(f)

    normalized_data = normalize_entities(rule_based_data)

    output_file = normalized_rule_based_folder / name
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(normalized_data, f, ensure_ascii=False, indent=4)
    print(f"Normalized rule based {name}")

"""Sort file normalized json lại theo tên"""
vistral_files = sorted(
    normalized_vistral_path.glob("*.json"),
    key=lambda p: int(p.stem)
)

"""Merge 2 file json lại với nhau và giải quyết duplicate/overlap."""
for json_vistral_file in vistral_files:
    json_rule_based_file = normalized_rule_based_folder / json_vistral_file.name

    if not json_rule_based_file.exists():
        print(f"Thiếu json file: {json_vistral_file.name } (vistral)")
        continue

    with open(json_vistral_file, "r", encoding="utf-8-sig") as f:
        vistral_data = json.load(f)

    with open(json_rule_based_file, "r", encoding="utf-8-sig") as f:
        rule_based_data = json.load(f)

    merged_data = merge_entities(vistral_data, rule_based_data)

    save_file = merged_output_folder / json_vistral_file.name

    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print("Merged:", json_vistral_file.name)

