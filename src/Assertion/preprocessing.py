"""
Gán thêm trường sentence cho NER bằng cách dựa vào position.
Sau đó, thay entity = ENTITY (vd: sốt -> ENTITY)
"""

import json
import re


def _split_sentences_with_spans(text):
    """
    Split `text` into sentences using Vietnamese sentence-ending punctuation
    (., ?, !), while keeping track of each sentence's character offsets
    (start, end) in the ORIGINAL text. This lets us map an entity's
    character span back to the sentence that contains it.

    Returns a list of (start, end, sentence_string) tuples, where
    sentence_string is stripped of leading/trailing whitespace but start/end
    still refer to the original (unstripped) sentence boundaries.
    """
    # Match runs of non-terminator characters followed by a terminator
    # (. ? !), or a trailing run with no terminator (last sentence in file).
    pattern = re.compile(r'[^.?!]*[.?!]|[^.?!]+$', re.UNICODE)

    sentences = []
    for match in pattern.finditer(text):
        raw_start, raw_end = match.start(), match.end()
        raw_sentence = match.group()

        if not raw_sentence.strip():
            # Skip empty/whitespace-only chunks (e.g. blank lines).
            continue

        sentences.append((raw_start, raw_end, raw_sentence.strip()))

    return sentences


def _find_sentence_for_span(sentences, start, end):
    """
    Given the list of (sent_start, sent_end, sentence_text) tuples and an
    entity's (start, end) character offsets, return the sentence text that
    contains the entity's starting position.

    Using the entity's start offset is enough because entities should not
    span across sentence boundaries in well-formed clinical text; if no
    exact containing sentence is found (e.g. due to punctuation edge cases),
    we fall back to the closest sentence whose range overlaps the entity.
    """
    for sent_start, sent_end, sentence_text in sentences:
        if sent_start <= start < sent_end:
            return sentence_text

    # Fallback: look for any overlap between entity span and sentence span.
    for sent_start, sent_end, sentence_text in sentences:
        if start < sent_end and end > sent_start:
            return sentence_text

    return None  # No containing sentence found (shouldn't normally happen).


def add_sentence_to_entities(txt_path, json_path, output_path):
    """
    Read a Vietnamese clinical .txt file and its corresponding NER .json
    entity file, attach the full sentence containing each entity, and write
    the enriched entities to `output_path`.

    Parameters
    ----------
    txt_path : str
        Path to the UTF-8 .txt file with the full clinical document.
    json_path : str
        Path to the .json file containing a list of entities, each with
        "text", "type", and "pos": [start, end].
    output_path : str
        Path where the enriched entity list (with a new "sentence" field)
        will be written as JSON.

    Notes
    -----
    - Original files are only opened for reading; they are never modified.
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
    sentences = _split_sentences_with_spans(full_text)

    # --- 4. Process each entity ---
    enriched_entities = []
    for entity in entities:
        start, end = entity['pos']
        expected_text = entity['text']
        actual_text = full_text[start:end]

        # --- 8. Validate offsets match the recorded entity text ---
        if actual_text != expected_text:
            raise ValueError(
                f"Entity/text mismatch at pos [{start}, {end}]: "
                f"expected {expected_text!r} but found {actual_text!r} "
                f"in the source document."
            )

        sentence = _find_sentence_for_span(sentences, start, end)
        if sentence is None:
            raise ValueError(
                f"Could not find a containing sentence for entity "
                f"{expected_text!r} at pos [{start}, {end}]."
            )

        # --- 5. Preserve original fields, add the new "sentence" field ---
        enriched_entities.append({
            'text': entity['text'],
            'type': entity['type'],
            'pos': entity['pos'],
            'sentence': sentence,
        })

    # --- 9. Write results to a new file (ensure_ascii=False keeps Vietnamese
    #        diacritics as real UTF-8 characters instead of \uXXXX escapes) ---
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enriched_entities, f, ensure_ascii=False, indent=2)

    return enriched_entities

"""
if __name__ == '__main__':
    # Example usage
    result = add_sentence_to_entities(
        txt_path='clinical_note.txt',
        json_path='entities.json',
        output_path='entities_with_sentences.json',
    )
    print(f"Wrote {len(result)} enriched entities.")
"""