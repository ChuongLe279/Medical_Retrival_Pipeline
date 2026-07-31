import re
import sys
import json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TEST_FOLDER_DIR = PROJECT_ROOT / "data" / "new_test_input"
NORMALIZED_TEST_FOLDER = PROJECT_ROOT / "data" / "test_normalized"


def check_folder_exist(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Folder does not exist: {path}"
        )

def normalize_text_file(text_file, save_file):
    """
    Chuẩn hóa nội dung file TXT:
    - Xóa các gạch đầu dòng và số thứ tự đầu dòng.
    - Xóa dòng trống.
    - Ghép các dòng thành một đoạn.
    - Các câu được ngăn cách bằng ". ".
    - Ghi đè kết quả vào file ban đầu.
    """
    text_file = Path(text_file)

    with text_file.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    normalized_lines = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Bỏ bullet: -, –, —, *, •, +, 1., 1), a),...
        line = re.sub(
            r"^\s*(?:[-–—*•+]+|\(?\d+[.)]|\(?[a-zA-Z][.)])\s*",
            "",
            line,
        )

        # Bỏ dấu câu ở cuối để tránh ".." khi ghép
        line = re.sub(r"[\s.!?;:,]+$", "", line).strip()

        if line:
            normalized_lines.append(line)

    normalized_text = ". ".join(normalized_lines)

    if normalized_text:
        normalized_text += "."

    # Chuẩn hóa khoảng trắng
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

    # Ghi đè trực tiếp lên file ban đầu
    with save_file.open("w", encoding="utf-8") as file:
        file.write(normalized_text)

    return normalized_text

if __name__ == "__main__":
    check_folder_exist(TEST_FOLDER_DIR)
    check_folder_exist(NORMALIZED_TEST_FOLDER)

    text_files = sorted(
        TEST_FOLDER_DIR.glob("*.txt"),
        key=lambda p: int(p.stem)
    )

    for file_path in text_files:
        save_path = NORMALIZED_TEST_FOLDER / file_path.name
        normalize_text_file(file_path, save_path)
        print(f"Normalized {file_path.name}")
    
