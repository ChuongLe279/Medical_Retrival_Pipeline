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


def build_flexible_pattern(text: str) -> str:
    """
    Tạo regex từ entity text để tìm lại span trong input gốc.

    Entity có thể đã bị rule/LLM chuẩn hóa khoảng trắng, ví dụ input có
    "bệnh viện  với" nhưng entity chỉ còn "bệnh viện với". Regex này giúp
    vẫn tìm được span thật trong input gốc.
    """
    parts = []

    for char in text.strip():
        if char.isspace():
            # Một dấu cách trong entity có thể tương ứng nhiều whitespace trong input.
            parts.append(r"\s+")
        elif char in "-‑–—":
            # Cho phép các kiểu dấu gạch nối khác nhau, hoặc khoảng trắng quanh dấu nối.
            parts.append(r"[\s\-‑–—]*")
        else:
            # Escape để ký tự đặc biệt như "." hay "(" không bị hiểu là cú pháp regex.
            parts.append(re.escape(char))

    return "".join(parts)


def find_position(entity_text: str, input_text: str, start_at: int = 0) -> tuple[int, int]:
    """Tìm vị trí start/end của một entity trong input gốc."""
    if not entity_text or not entity_text.strip():
        return -1, -1

    # Tìm không phân biệt hoa/thường, nhưng vẫn trả offset theo input gốc.
    input_text_lower = input_text.lower()
    search_text = entity_text.lower()

    # Bước 1: tìm exact match trước vì nhanh và chính xác nhất.
    start = input_text_lower.find(search_text, start_at)
    if start == -1 and start_at:
        # Nếu đang tìm occurrence tiếp theo mà không thấy, fallback tìm từ đầu.
        start = input_text_lower.find(search_text)

    if start != -1:
        return start, start + len(entity_text)

    # Bước 2: nếu exact match fail, dùng regex linh hoạt cho khoảng trắng/dấu nối.
    pattern = build_flexible_pattern(entity_text)
    match = re.search(pattern, input_text, flags=re.IGNORECASE)
    if not match:
        return -1, -1

    return match.start(), match.end()


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

    # Xử lý trường hợp mà json output chỉ có 1 object
    if isinstance(json_objects, dict):
        json_objects = [json_objects]
    elif not isinstance(json_objects, list):
        raise TypeError(f"{json_file} must contain a JSON object or a list of JSON objects")

    # Ghi nhớ vị trí đã match gần nhất của từng entity.
    # Nhờ vậy nếu cùng một entity xuất hiện nhiều lần, lần sau sẽ tìm tiếp phía sau.
    next_search_start_by_key = {}

    for obj in json_objects:
        if not isinstance(obj, dict):
            raise TypeError(f"{json_file} contains a non-object item: {obj!r}")

        key = (
            obj.get("text", "").lower(),
            obj.get("type", ""),
            obj.get("source", ""),
        )
        start_at = next_search_start_by_key.get(key, 0)
        start, end = find_position(obj["text"], input_text, start_at)
        obj["position"] = [start, end]

        if start != -1:
            next_search_start_by_key[key] = end

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
