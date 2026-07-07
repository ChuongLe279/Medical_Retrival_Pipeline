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
from string_matching_pos import string_matching_pos
import json
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

"""Xóa hết content của file output merged trước khi merge"""
for old_file in merged_output_folder.glob("*.json"):
    old_file.unlink()

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

    normalized_data = []
    for obj in vistral_data:
        start, end = obj["position"]

        new_obj = {
            "text": obj["text"],
            "type": obj["type"],
            "source": obj["source"],
            "start": start,
            "end": end,
        }
        normalized_data.append(new_obj)

    output_file = normalized_vistral_folder / name
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(normalized_data, f, ensure_ascii=False, indent=4)
    print(f"Normalized vistral {name}")

    # Chuẩn hóa rule based data
    with open(json_rule_based_file, "r", encoding="utf-8-sig") as f:
        rule_based_data = json.load(f)

    normalized_data = []
    for obj in rule_based_data:
        start, end = obj["position"]

        new_obj = {
            "text": obj["text"],
            "type": obj["type"],
            "source": obj["source"],
            "start": start,
            "end": end,
        }
        normalized_data.append(new_obj)

    output_file = normalized_rule_based_folder / name
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(normalized_data, f, ensure_ascii=False, indent=4)
    print(f"Normalized rule based {name}")

"""Sort file normalized json lại theo tên"""
vistral_files = sorted(
    normalized_vistral_path.glob("*.json"),
    key=lambda p: int(p.stem)
)

"""Merge 2 file json lại với nhau. Xếp chồng."""
for json_vistral_file in vistral_files:
    json_rule_based_file = normalized_rule_based_folder / json_vistral_file.name

    if not json_rule_based_file.exists():
        print(f"Thiếu json file: {json_vistral_file.name } (vistral)")
        continue

    with open(json_vistral_file, "r", encoding="utf-8-sig") as f:
        vistral_data = json.load(f)

    with open(json_rule_based_file, "r", encoding="utf-8-sig") as f:
        rule_based_data = json.load(f)

    merged_data = []
    merged_data.extend(vistral_data)
    merged_data.extend(rule_based_data)

    save_file = merged_output_folder / json_vistral_file.name

    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print("Merged:", json_vistral_file.name)

