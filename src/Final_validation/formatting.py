import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

JSON_FOLDER = PROJECT_ROOT / "data" / "output_merged_linking"
FINAL_JSON_FOLDER = PROJECT_ROOT / "data" / "output_merged_validation"


def final_formating(json_folder):
    pass


if __name__ == "__main__":
    final_formating(FINAL_JSON_FOLDER)
    print(f"Submission files created at: {FINAL_JSON_FOLDER}")