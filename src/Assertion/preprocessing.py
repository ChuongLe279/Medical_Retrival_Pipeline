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

TEST_INPUT_PATH = PROJECT_ROOT / "data" / "new_test_input"
NORMALIZED_TEXT_PATH = PROJECT_ROOT / "data" / "test_normalized"
JSON_PATH = PROJECT_ROOT / "data" / "new_ner_output"
ENTITY_PLACEHOLDER = "[ENTITY]"
SUPPORTED_ENTITY_TYPES = {"CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"}


def normalized_ner_entity(entity_text: str) -> str:
    """Collapse repeated whitespace and trim an NER entity's text."""
    return re.sub(r"\s+", " ", entity_text).strip()

def find_position(entity_text: str, input_text: str, start_at: int = 0) -> tuple[int, int]:
    """
    Find the position of `entity_text` within `input_text`, searching from
    character offset `start_at` onward. Returns (start, end) offsets.

    Raises ValueError if the entity text cannot be found from that point on.
    """
    pattern = re.compile(re.escape(entity_text))
    match = pattern.search(input_text, start_at)

    # The NER representation collapses whitespace, while raw test input can
    # contain repeated spaces, tabs, or line breaks. Treat each whitespace run
    # in the entity as one-or-more whitespace characters in the source.
    if match is None:
        flexible_parts = [
            r"\s+" if part.isspace() else re.escape(part)
            for part in re.split(r"(\s+)", entity_text)
            if part
        ]
        match = re.compile("".join(flexible_parts), re.IGNORECASE).search(
            input_text, start_at
        )

    # NER output may add sentence-ending punctuation that is not present in
    # the raw input (for example ``"Viêm mô tế bào."`` at the end of a
    # bullet line). In that case, match the textual part and leave the absent
    # punctuation outside the returned span.
    if match is None and entity_text.endswith((".", "?", "!")):
        text_without_terminal_mark = entity_text[:-1].rstrip()
        if text_without_terminal_mark:
            flexible_parts = [
                r"\s+" if part.isspace() else re.escape(part)
                for part in re.split(r"(\s+)", text_without_terminal_mark)
                if part
            ]
            fallback_pattern = re.compile(
                "".join(flexible_parts) + r"(?=\s|$)", re.IGNORECASE
            )
            match = fallback_pattern.search(input_text, start_at)

    if match is None:
        raise ValueError(
            f"Could not find entity text {entity_text!r} in the input "
            f"text (searching from position {start_at})."
        )

    return match.start(), match.end()


def calculating_assertion_pos(json_file, text_file):
    """
    Locate every entity in the raw test input file `text_file`, but only
    store "assertion_position" for supported assertion entity types.

    Overlap-aware: uses each entity's ORIGINAL "position" span to detect nesting
    (entity B is "nested" in entity A if A's original span contains B's
    original start). This matters because normalize_text_file() reshapes
    the raw text, so we can't assume entities occur strictly left-to-right
    with no overlap in the input text -- a naive "search from previous
    match's end" approach would fail to find a child entity whose
    normalized position falls *before* its parent's normalized end.

    We process entities in order of original (start, -length) -- i.e.
    parents before the children nested inside them -- and maintain a stack
    of "open" ancestor spans. A nested entity is searched starting from its
    nearest open ancestor's own normalized start (or, once siblings have
    been found under that ancestor, from the end of the most recent
    sibling), rather than from the ancestor's end. Non-overlapping
    top-level entities behave exactly like the old forward-chaining logic.
    """
    with json_file.open("r", encoding="utf-8") as file:
        entities = json.load(file)

    with text_file.open("r", encoding="utf-8") as file:
        input_text = file.read()

    # Process parents before children nested inside them: earlier original
    # start first, and for ties, the longer (outer) span first.
    order = sorted(
        range(len(entities)),
        key=lambda i: (
            entities[i]["position"][0],
            -(entities[i]["position"][1] - entities[i]["position"][0]),
        ),
    )

    # Each stack frame represents an "open" ancestor span (by original position):
    #   orig_end     - original end offset, used to know when it's fully closed
    #   norm_end     - this entity's own normalized end offset
    #   last_norm_end- normalized cursor for the *next* thing found inside
    #                  it (starts at this entity's own normalized START,
    #                  since children can appear anywhere within its span)
    stack = []
    root_last_end = 0

    for idx in order:
        entity = entities[idx]
        orig_start, orig_end = entity["position"]

        entity_text = entity["text"]

        # Close out ancestor frames that have fully ended before this
        # entity starts (no overlap -> not actually an ancestor), pushing
        # the furthest normalized position they/their children reached up
        # to the next level so siblings search forward correctly.
        while stack and stack[-1]["orig_end"] <= orig_start:
            closed = stack.pop()
            bound = max(closed["norm_end"], closed["last_norm_end"])
            if stack:
                stack[-1]["last_norm_end"] = max(stack[-1]["last_norm_end"], bound)
            else:
                root_last_end = max(root_last_end, bound)

        search_from = stack[-1]["last_norm_end"] if stack else root_last_end

        norm_start, norm_end = find_position(
            entity_text, input_text, search_from
        )
        if entity.get("type") in SUPPORTED_ENTITY_TYPES:
            # Use the exact raw substring. This is important when the NER text
            # contains punctuation that was absent from the source document.
            entity["text"] = input_text[norm_start:norm_end]
            entity["assertion_position"] = [norm_start, norm_end]
        else:
            entity.pop("assertion_position", None)
            entity.pop("sentence", None)

        # Push this entity as a new (potential) ancestor frame for any
        # entities nested inside it.
        stack.append({
            "orig_end": orig_end,
            "norm_end": norm_end,
            "last_norm_end": norm_start,
        })

    with json_file.open("w", encoding="utf-8") as file:
        json.dump(entities, file, ensure_ascii=False, indent=2)

    return entities


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


def _find_normalized_assertion_positions(entities, normalized_text):
    """Locate supported entities in normalized text without changing raw offsets."""
    supported_indices = [
        i for i, entity in enumerate(entities)
        if entity.get("type") in SUPPORTED_ENTITY_TYPES
    ]
    order = sorted(
        supported_indices,
        key=lambda i: (
            entities[i]["position"][0],
            -(entities[i]["position"][1] - entities[i]["position"][0]),
        ),
    )
    positions = {}
    stack = []
    root_last_end = 0

    for idx in order:
        entity = entities[idx]
        orig_start, orig_end = entity["position"]

        while stack and stack[-1]["orig_end"] <= orig_start:
            closed = stack.pop()
            bound = max(closed["text_end"], closed["last_text_end"])
            if stack:
                stack[-1]["last_text_end"] = max(
                    stack[-1]["last_text_end"], bound
                )
            else:
                root_last_end = max(root_last_end, bound)

        search_from = stack[-1]["last_text_end"] if stack else root_last_end
        normalized_entity_text = normalized_ner_entity(entity["text"])
        start, end = find_position(
            normalized_entity_text, normalized_text, search_from
        )
        positions[idx] = (start, end)
        stack.append({
            "orig_end": orig_end,
            "text_end": end,
            "last_text_end": start,
        })

    return positions


def add_sentence_to_entities(txt_path, json_path):
    """
    Read a normalized Vietnamese clinical .txt file and its
    corresponding NER .json entity file, attach the full sentence containing
    each supported assertion entity (with that entity's own text replaced by
    "[ENTITY]"), and write the entities back to `json_path`.

    Parameters
    ----------
    txt_path : str or Path
        Path to the UTF-8 normalized .txt file with the full clinical document.
    json_path : str or Path
        Path to the .json file containing a list of entities, each with
        "text", "type", "position" (offsets in the original raw text) and
        "assertion_position" for supported types (offsets in the raw
        text, computed by calculating_assertion_pos). This file is overwritten
        in place.

    Notes
    -----
    - txt_path is only ever opened for reading; it is never modified.
    - Raises ValueError if an entity's recorded text does not match the
      actual substring of the raw document at its
      "assertion_position" offsets.
    """
    # --- 1. Read the normalized source text (read-only) ---
    with open(txt_path, 'r', encoding='utf-8') as f:
        full_text = f.read()

    # --- 2. Read the entities (read-only) ---
    with open(json_path, 'r', encoding='utf-8') as f:
        entities = json.load(f)

    # --- 3. Pre-compute sentence spans once for the whole document ---
    spans = _split_sentences_with_spans(full_text)
    normalized_positions = _find_normalized_assertion_positions(entities, full_text)

    # --- 4. Process each entity ---
    enriched_entities = []
    for entity_index, entity in enumerate(entities):
        if entity.get("type") not in SUPPORTED_ENTITY_TYPES:
            enriched_entities.append({
                'text': entity['text'],
                'type': entity['type'],
                'position': entity['position'],
            })
            continue

        # assertion_position belongs to the raw input. Locate a separate span
        # in normalized text solely for sentence extraction and masking.
        start, end = normalized_positions[entity_index]
        expected_text = normalized_ner_entity(entity['text'])

        sent_span = _find_sentence_span(spans, start, end)
        if sent_span is None:
            raise ValueError(
                f"Could not find a containing sentence for entity "
                f"{expected_text!r} at normalized position [{start}, {end}] ({txt_path})."
            )
        sent_start, sent_end = sent_span

        # --- Build the sentence text with THIS entity masked as [ENTITY] ---
        sentence = _build_sentence_with_masked_entity(
            full_text, sent_start, sent_end, start, end
        )

        # --- Preserve original fields, add the new fields ---
        enriched_entities.append({
            'text': entity['text'],
            'type': entity['type'],
            'position': entity['position'],
            'assertion_position': entity['assertion_position'],
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
    check_folder_exist(TEST_INPUT_PATH)
    check_folder_exist(NORMALIZED_TEXT_PATH)

    json_files = sorted(
        JSON_PATH.glob("*.json"),
        key=lambda p: int(p.stem)
    )

    for json_path in json_files:
        print(f"Processing: {json_path.name}")
        raw_text_path = TEST_INPUT_PATH / json_path.with_suffix(".txt").name
        normalized_text_path = NORMALIZED_TEXT_PATH / json_path.with_suffix(".txt").name

        print(f"Calculating assertion_position for {json_path.name}")
        calculating_assertion_pos(json_path, raw_text_path)

        print(f"Adding entities {json_path.name}")
        add_sentence_to_entities(normalized_text_path, json_path)

    print("Preprocessing is done.")
