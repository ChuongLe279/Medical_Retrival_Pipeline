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
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

JSON_FOLDER_PATH = Path("data/mock_data")

MODEL_NAME = "Qwen/Qwen3-8B"

# Load once — no quantization
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)

model.eval()

@torch.inference_mode()
def qwen_translate_vi_to_en(s: str) -> str:
    """
    Translate Vietnamese medical text into English using Qwen3-8B.
    Returns only the translated text.
    """
    if not isinstance(s, str):
        raise TypeError("Input must be a string.")

    s = s.strip()

    if not s:
        return ""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional Vietnamese-to-English medical translator. "
                "Translate the supplied Vietnamese clinical text accurately into English. "
                "Preserve medication names, disease names, laboratory abbreviations, "
                "dosages, units, routes, frequencies, numbers, and medical terminology. "
                "Do not summarize, explain, or add information. "
                "Return only the English translation."
            ),
        },
        {
            "role": "user",
            "content": s,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate( # type: ignore
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Remove the original prompt tokens
    new_tokens = generated_ids[
        0,
        inputs["input_ids"].shape[1]:
    ]

    translation = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    return translation.strip()

def translate_test_data(JSON_FOLDER):
    for file_path in JSON_FOLDER.glob("*.json"):
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        for entities in data:
            entities["eng_translation"] = qwen_translate_vi_to_en(entities["text"])
            print(entities)

            with JSON_FOLDER.open("w", encoding="utf-8") as file:
                json.dump(entities, file, ensure_ascii=False)

translate_test_data(JSON_FOLDER_PATH)
        
    


