import json
import re
from pathlib import Path
import torch

JSON_FOLDER = Path("/content/drive/MyDrive/json")
MODEL_NAME = "Qwen/Qwen3-8B"
SUPPORTED_LINKING_TYPES = {"THUỐC", "CHẨN_ĐOÁN"}

tokenizer = None
model = None


def json_sort_key(path: Path):
    try:
        return 0, int(path.stem)
    except ValueError:
        return 1, path.stem


def get_model_and_tokenizer():
    """Load Qwen only when generation is actually requested."""
    global tokenizer, model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if model is None:
        dtype = torch.float16 if torch.cuda.is_available() else "auto"
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            device_map="auto",
        )
        model.eval()
    return tokenizer, model


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

    tokenizer, model = get_model_and_tokenizer()

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

    tokenizer, model = get_model_and_tokenizer()

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


@torch.inference_mode()
def qwen_linking_queries(text: str, entity_type: str) -> list[str]:
    """Create concise English linking queries with one model generation."""
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    text = " ".join(text.strip().split())
    if not text or entity_type not in SUPPORTED_LINKING_TYPES:
        return []

    if entity_type == "THUỐC":
        instruction = (
            "Convert the medication mention to concise English RxNorm lookup "
            "queries. Include only applicable forms among: original English "
            "translation, generic ingredient name, common brand name, and expanded "
            "medication abbreviation. Preserve strength and dose form when present. "
            "Never output a diagnosis, drug class, indication, explanation, or full "
            "sentence. Return a JSON array of 1 to 4 unique strings, each no longer "
            "than 12 words."
        )
    else:
        instruction = (
            "Convert the diagnosis mention to concise English ICD-10 lookup queries. "
            "Include only applicable forms among: direct English translation, "
            "canonical diagnosis, common clinical synonym, and expanded abbreviation. "
            "Preserve anatomical site, acuity, cause, severity, and complications only "
            "when stated. Never add a code, explanation, treatment, or unsupported "
            "diagnosis. Return a JSON array of 1 to 4 unique strings, each no longer "
            "than 20 words."
        )

    tokenizer, model = get_model_and_tokenizer()
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(
        model.device
    )
    generated_ids = model.generate(  # type: ignore
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    result = tokenizer.decode(
        generated_ids[0, inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()

    array_match = re.search(r"\[[\s\S]*?\]", result)
    if array_match is None:
        return []
    try:
        candidates = json.loads(array_match.group())
    except json.JSONDecodeError:
        return []
    if not isinstance(candidates, list):
        return []

    max_words = 12 if entity_type == "THUỐC" else 20
    queries = []
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, str):
            continue
        candidate = " ".join(candidate.strip().split())
        key = candidate.casefold()
        if candidate and len(candidate.split()) <= max_words and key not in seen:
            seen.add(key)
            queries.append(candidate)
    return queries[:4]

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
    query_cache = {}

    for file_path in json_files:
        print(f"Processing file: {file_path.name}")

        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        for entity in data:
            entity_type = entity.get("type", "")
            source_text = entity.get("text", "")
            if entity_type not in SUPPORTED_LINKING_TYPES or not isinstance(source_text, str):
                entity.pop("eng_translation", None)
                continue

            cache_key = (entity_type, " ".join(source_text.strip().split()).casefold())
            if cache_key not in query_cache:
                query_cache[cache_key] = qwen_linking_queries(source_text, entity_type)
            entity["eng_translation"] = list(query_cache[cache_key])


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
        key=json_sort_key,
    )

    if not json_files:
        raise FileNotFoundError(
            f"No JSON files found in: {JSON_FOLDER}"
        )
    query_expansion(json_files)


