"""
Xử lý trường position dựa vào json.
Vì bước string matching này được diễn ra 2 lần trong pipeline NER
nên t chỉ viết hàm ở đây thôi. Tránh việc code chạy lặp không
cần thiết.
"""
import json
from pathlib import Path
import re

BASE_DIR = Path(__file__).parent

def string_matching_pos(json_file, input_file):
    """
        Nhận directory của jsonfile và input file (test data) tương ứng. Thêm trường position dựa
        vào name của json file.
    """
    json_file = Path(json_file)
    input_file = Path(input_file)
    with open(json_file, "r", encoding="utf-8-sig") as f:
        json_objects = json.load(f)
    with open(input_file, "r", encoding="utf-8-sig") as f:
        input_text = f.read()
    input_text_lower = input_text.lower()

    # Xử lý trường hợp mà json output chỉ có 1 object
    if isinstance(json_objects, dict):
        json_objects = [json_objects]
    elif not isinstance(json_objects, list):
        raise TypeError(f"{json_file} must contain a JSON object or a list of JSON objects")

    for obj in json_objects:
        if not isinstance(obj, dict):
            raise TypeError(f"{json_file} contains a non-object item: {obj!r}")
        
        # Chuyển chữ hoa thành chữ thường để tránh trường hợp pos [-1,-1]
        search_text = obj["text"].lower()
        start = input_text_lower.find(search_text)
        if start == -1:
            obj["position"] = [-1, -1]
        else:
            end = start + len(obj["text"])
            obj["position"] = [start, end]

    # Cập nhật nội dung json mới
    json_str = json.dumps(json_objects, ensure_ascii=False, indent=4)

    # Chỉnh json lại cho đúng định dạng của đề thi
    json_str = re.sub(
            r'\[\s*(-?\d+),\s*(-?\d+)\s*\]',
            r'[\1, \2]',
            json_str 
        )

    with open(json_file, "w", encoding="utf-8") as f:
        f.write(json_str)

"""
json_path = BASE_DIR / "output" / "vistral_output" / "2.json"
input_path = BASE_DIR.parents[2] / "data" / "test" / "2.txt"
string_matching_pos(json_path, input_path)
"""
