"""
merge vistral và hardcode lại
100 vistral + 100 rule-based -> 100 merge output file
"""
import json
from pathlib import Path
from string_matching_pos import string_matching_pos

BASE_DIR = Path(__file__).parent
input_path = input_path = BASE_DIR.parents[2] / "data" / "test" / "2.txt"

# Tính string matching cho trường position của cả 2 output của vistral và rule-based
# Đầu tiên tính cho file output của vistral trước
vistral_output_path = Path("./output/vistral_output")

for json_file in vistral_output_path.glob("*.json"):
   print("Calculating:", json_file.name)
   string_matching_pos(json_file, input_path)



    