"""
Dịch từ tiếng việt sang tiếng anh cho trường "text" của các NER entities được trích ra từ test data (thêm trường eng_translate).
LLM: vinai-vi2en

NER entites có dạng như sau:
[
  {
    "text": "amlodipine 10 mg po daily",
    "type": "THUỐC",
    "assertions": ["isHistorical"],
    "position": [58, 83]
  },
  {
    "text": "đau nhức",
    "type": "TRIỆU_CHỨNG",
    "assertions": [],
    "position": [254, 262]
  }
]
"""

import json
from pathlib import Path


