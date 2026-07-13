import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

JSON_FOLDER = Path(
    "/content/drive/MyDrive/Mock data viettel ai"
)
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
                "You are a professional Vietnamese-to-English translator. "
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

@torch.inference_mode()
def qwen_query_expansion(s: str) -> list[str]:
    """
    Generate three separate expansions for an English medical query:

    1. Canonical diagnosis
    2. English synonym
    3. Abbreviation expansion
    """
    if not isinstance(s, str):
        raise TypeError("Input must be a string.")

    s = s.strip()

    if not s:
        return ["", "", ""]

    expansion_prompts = [
        (
            "Convert the English medical query into its canonical clinical "
            "diagnosis or standardized medical term. Preserve details such as "
            "anatomical site, severity, acuity, cause, and complications. "
            "Do not add unsupported information or an ICD-10 code. "
            "If it is already canonical, return it unchanged. "
            "Return only the resulting English query."
        ),
        (
            "Rewrite the English medical query using one common English medical "
            "synonym for its main diagnosis or medical concept. Preserve all "
            "clinically important details. Do not add unsupported information. "
            "If no appropriate synonym exists, return the original query unchanged. "
            "Return only the resulting English query."
        ),
        (
            "Expand every medical abbreviation in the English medical query into "
            "its full English form. Do not change words that are not abbreviations. "
            "Do not add explanations or unsupported information. "
            "If there is no abbreviation, return the original query unchanged. "
            "Return only the resulting English query."
        ),
    ]

    expansions = []

    for instruction in expansion_prompts:
        messages = [
            {
                "role": "system",
                "content": instruction,
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
            truncation=True,
            max_length=2048,
        ).to(model.device)

        generated_ids = model.generate(  # type: ignore
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        new_tokens = generated_ids[
            0,
            inputs["input_ids"].shape[1]:
        ]

        result = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        )

        result = " ".join(result.strip().split())

        # Always preserve exactly three additional queries
        expansions.append(result if result else s)

    return expansions

def translate_test_data(json_files):
    print(f"Found {len(json_files)} JSON files.")
    for file_path in json_files:
        print(f"Processing file: {file_path.name}")
        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        
        # Process all entities in the data for the current file
        for entities in data:
            entities["eng_translation"] = qwen_translate_vi_to_en(entities["text"])

        with file_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        print(f"Saved updated data to: {file_path.name}")

def query_expansion(json_files):
    print(f"Found {len(json_files)} JSON files.")

    for file_path in json_files:
        print(f"Processing file: {file_path.name}")

        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        for entity in data:
            english_text = qwen_translate_vi_to_en(entity["text"])

            expansions = qwen_query_expansion(english_text)

            entity["eng_translation"] = [
                english_text,       # Original English translation
                expansions[0],      # Canonical diagnosis
                expansions[1],      # English synonym
                expansions[2],      # Abbreviation expansion
            ]
            candidates = [
                english_text,
                expansions[0],  # Canonical diagnosis
                expansions[1],  # English synonym
                expansions[2],  # Abbreviation expansion
            ]

            """Dedupe vì qwen8B đôi khi không mở rộng được ngữ cảnh."""
            seen = set()
            entity["eng_translation"] = []

            for text in candidates:
                if not isinstance(text, str):
                    continue

                text = " ".join(text.strip().split())

                if not text:
                    continue

                normalized_text = text.casefold()

                if normalized_text not in seen:
                    seen.add(normalized_text)
                    entity["eng_translation"].append(text)


        with file_path.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print(f"Saved updated data to: {file_path.name}")

if __name__ == "__main__":
    if not JSON_FOLDER.exists():
        raise FileNotFoundError(
            f"Folder does not exist: {JSON_FOLDER}"
        )

    json_files = sorted(
        JSON_FOLDER.glob("*.json"),
        key=lambda path: int(path.stem)
    )

    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in: {JSON_FOLDER}"
        )
    query_expansion(json_files)


