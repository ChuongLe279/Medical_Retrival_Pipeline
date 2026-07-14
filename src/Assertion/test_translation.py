"""
Dịch toàn bộ test data sang tiếng anh.
Chưa làm xong phần duyệt toàn bộ folder đẻ dịch. Lười qúa à
"""
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

JSON_FOLDER = Path(
    "/content/drive/MyDrive/Mock data viettel ai"
)
ENGLISH_TEST_PATH = Path()


MODEL_NAME = "Qwen/Qwen3-8B"
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
                "You are an expert medical translator specializing in Vietnamese "
                "clinical records. Translate the provided Vietnamese medical text "
                "into accurate, concise English clinical language.\n\n"
                
                "Strict requirements:\n"
                "1. Preserve the exact medical meaning. Do not summarize, explain, "
                "correct, infer, or add information.\n"
                "2. Preserve assertion cues exactly, including:\n"
                "   - Negation: không, chưa, phủ nhận, không ghi nhận.\n"
                "   - Uncertainty: nghi ngờ, có thể, theo dõi, không loại trừ.\n"
                "   - Temporality: tiền sử, đã từng, trước đây, trước nhập viện.\n"
                "   - Experiencer: patient, mother, father, family, or another person.\n"
                "3. Preserve medication names, dosages, routes, frequencies, laboratory "
                "values, units, dates, medical codes, and numerical values.\n"
                "4. Preserve headings, lists, line breaks, punctuation, and document structure.\n"
                "5. Preserve common medical abbreviations such as WBC, Hb, IV, PO, BID, "
                "and PRN. Translate an abbreviation only when its meaning is unambiguous.\n"
                "6. If the source is ambiguous or incomplete, keep it ambiguous or incomplete.\n"
                "7. Return only the English translation without commentary, quotation "
                "marks, headings, or Markdown."
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

