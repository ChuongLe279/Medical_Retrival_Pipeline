"""
Tính trường position cho 2 output file là vistral và rule based
"""
from pathlib import Path
from string_matching_pos import string_matching_pos

BASE_DIR = Path(__file__).parent
input_dir = BASE_DIR.parents[2] / "data" / "test" 

# Tính string matching cho trường position của cả 2 output của vistral và rule-based
# Đầu tiên tính cho file output của vistral trước
vistral_output_path = BASE_DIR / "output" / "vistral_output"

# Quality of life: sort các output json file theo thứ tự của cả hai output file
json_files = sorted(
    vistral_output_path.glob("*.json"),
    key=lambda p: int(p.stem)
)

for json_file in json_files:
   input_path = input_dir / f"{json_file.stem}.txt"
   if not input_path.exists():
        print(f"Cảnh báo: {input_path.name} không tìm thấy, bỏ qua {json_file.name}")
        continue

   print("Calculating vistral:", json_file.name, "->", input_path.name)
   string_matching_pos(json_file, input_path)

# Tiếp đó là xử lý output file cho rule-based
rule_based_output_path = BASE_DIR / "output" / "rule_based"

# Quality of life: sort các output json file theo thứ tự của cả hai output file
json_files = sorted(
    rule_based_output_path.glob("*.json"),
    key=lambda p: int(p.stem)
)

for json_file in json_files:
   input_path = input_dir / f"{json_file.stem}.txt"
   if not input_path.exists():
        print(f"Cảnh báo: {input_path.name} không tìm thấy, bỏ qua {json_file.name}")
        continue

   print("Calculating rule_based:", json_file.name, "->", input_path.name)
   string_matching_pos(json_file, input_path)
    