"""
Gán thêm trường sentence cho NER bằng cách dựa vào position.
Sau đó, thay entity = ENTITY (vd: sốt -> ENTITY)
"""

import re
import sys
import json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TEXT_PATH = PROJECT_ROOT / "src" / "Assertion" / "data" / "mock_txt"
JSON_PATH = PROJECT_ROOT / "src" / "Assertion" / "data" / "mock_ner"

ENTITY_PLACEHOLDER = "[ENTITY]"


def _split_sentences_with_spans(text):
    """
    Split `text` into sentences using Vietnamese sentence-ending punctuation
    (., ?, !), while keeping track of each sentence's character offsets
    (start, end) in the ORIGINAL text.

    Returns a list of (start, end) tuples referring to the RAW (unstripped)
    sentence span in `text`. We deliberately do NOT strip/store the sentence
    string itself here, because masking an entity later requires slicing the
    raw text at exact offsets -- stripping first would shift those offsets.
    """
    pattern = re.compile(r'[^.?!]*[.?!]|[^.?!]+$', re.UNICODE)

    spans = []
    for match in pattern.finditer(text):
        raw_start, raw_end = match.start(), match.end()
        if not match.group().strip():
            # Skip empty/whitespace-only chunks (e.g. blank lines).
            continue
        spans.append((raw_start, raw_end))

    return spans


def _find_sentence_span(spans, start, end):
    """
    Given the list of (sent_start, sent_end) tuples and an entity's
    (start, end) character offsets, return the (sent_start, sent_end) of the
    sentence that contains the entity's starting position.

    Falls back to overlap matching if no sentence strictly contains it
    (e.g. punctuation edge cases).
    """
    for sent_start, sent_end in spans:
        if sent_start <= start < sent_end:
            return sent_start, sent_end

    for sent_start, sent_end in spans:
        if start < sent_end and end > sent_start:
            return sent_start, sent_end

    return None


def _build_sentence_with_masked_entity(full_text, sent_start, sent_end, ent_start, ent_end):
    """
    Return the sentence's text (stripped) with THIS entity's own span
    replaced by ENTITY_PLACEHOLDER. Other entities that may share the same
    sentence are left untouched -- only the entity's own offsets are masked.

    We compute the entity's position relative to the sentence's raw start,
    slice the raw (unstripped) sentence text around it, and strip only
    after masking so offsets stay correct.
    """
    raw_sentence = full_text[sent_start:sent_end]
    rel_start = ent_start - sent_start
    rel_end = ent_end - sent_start

    masked = raw_sentence[:rel_start] + ENTITY_PLACEHOLDER + raw_sentence[rel_end:]
    return masked.strip()


def add_sentence_to_entities(txt_path, json_path):
    """
    Read a Vietnamese clinical .txt file and its corresponding NER .json
    entity file, attach the full sentence containing each entity (with that
    entity's own text replaced by "ENTITY"), and write the enriched entities
    back to `json_path`.

    Parameters
    ----------
    txt_path : str or Path
        Path to the UTF-8 .txt file with the full clinical document.
    json_path : str or Path
        Path to the .json file containing a list of entities, each with
        "text", "type", and "pos": [start, end]. This file is overwritten
        in place with the enriched entities.

    Notes
    -----
    - txt_path is only ever opened for reading; it is never modified.
    - Raises ValueError if an entity's recorded text does not match the
      actual substring of the document at its given [start, end] offsets.
    """
    # --- 1. Read the source text (read-only) ---
    with open(txt_path, 'r', encoding='utf-8') as f:
        full_text = f.read()

    # --- 2. Read the entities (read-only) ---
    with open(json_path, 'r', encoding='utf-8') as f:
        entities = json.load(f)

    # --- 3. Pre-compute sentence spans once for the whole document ---
    spans = _split_sentences_with_spans(full_text)

    # --- 4. Process each entity ---
    enriched_entities = []
    for entity in entities:
        start, end = entity['pos']
        expected_text = entity['text']
        actual_text = full_text[start:end]

        # --- Validate offsets match the recorded entity text ---
        if actual_text != expected_text:
            raise ValueError(
                f"Entity/text mismatch at pos [{start}, {end}]: "
                f"expected {expected_text!r} but found {actual_text!r} "
                f"in the source document ({txt_path})."
            )

        sent_span = _find_sentence_span(spans, start, end)
        if sent_span is None:
            raise ValueError(
                f"Could not find a containing sentence for entity "
                f"{expected_text!r} at pos [{start}, {end}] ({txt_path})."
            )
        sent_start, sent_end = sent_span

        # --- Build the sentence text with THIS entity masked as ENTITY ---
        sentence = _build_sentence_with_masked_entity(
            full_text, sent_start, sent_end, start, end
        )

        # --- Preserve original fields, add the new "sentence" field ---
        enriched_entities.append({
            'text': entity['text'],
            'type': entity['type'],
            'pos': entity['pos'],
            'sentence': sentence,
        })

    # --- Write results back (ensure_ascii=False keeps Vietnamese diacritics
    #     as real UTF-8 characters instead of \uXXXX escapes) ---
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_entities, f, ensure_ascii=False, indent=2)


def check_folder_exist(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Folder does not exist: {path}"
        )


if __name__ == '__main__':
    check_folder_exist(JSON_PATH)
    check_folder_exist(TEXT_PATH)

    for json_path in JSON_PATH.glob("*.json"):
        print(f"Processing: {json_path.name}")
        text_path = TEXT_PATH / json_path.with_suffix(".txt").name
        add_sentence_to_entities(text_path, json_path)
