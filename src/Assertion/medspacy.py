"""
Dùng medspacy for assertion.
Phân ra 4 trường: isNegated, isFamily, isHistorial.
"""

import medspacy
from medspacy.ner import TargetRule
from pathlib import Path

nlp = medspacy.load()  # loads default pipeline: tokenizer, target matcher, context, sectionizer


ENGLISH_TEST_PATH = Path()
JSON_FOLDER_PATH = Path()

spans = [
    (34, 42, "DIAGNOSIS"),  # "diabetes"
    (47, 52, "SYMPTOM"),    # "fever"
]
def run_medspacy_on_ner_spans(english_text, ner_spans):
    """
    ner_spans: list of (start_char, end_char, label) tuples,
               e.g. from your ViHealthBERT NER output, already remapped
               to the ENGLISH translated text's character offsets.
    """
    nlp = medspacy.load()
    nlp.remove_pipe("medspacy_target_matcher")  # we supply entities ourselves
 
    doc = nlp.tokenizer(english_text)
 
    ents = []
    for start_char, end_char, label in ner_spans:
        span = doc.char_span(start_char, end_char, label=label, alignment_mode="contract")
        if span is None:
            # Your char offsets didn't line up with any token boundary at all.
            # Log this — it usually means an alignment bug between your NER
            # output and the text you're running medspacy on.
            print(f"WARNING: could not align span ({start_char}, {end_char}) "
                  f"= {english_text[start_char:end_char]!r}")
            continue
        ents.append(span)
 
    doc.ents = ents
 
    # Run remaining pipes (medspacy_pyrush for sentence splitting,
    # medspacy_context for assertion) in order.
    for pipe_name in nlp.pipe_names:
        doc = nlp.get_pipe(pipe_name)(doc)
 
    results = []
    for ent in doc.ents:
        results.append({
            "text": ent.text,
            "label": ent.label_,
            "isNegated_medspacy": ent._.is_negated,
            "isHistorical": ent._.is_historical,
            "isFamily": ent._.is_family,
            "isHypothetical": ent._.is_hypothetical,
            "isUncertain": ent._.is_uncertain,
        })
    return results
 

if __name__ == "__main__":
    if not ENGLISH_TEST_PATH.exists():
        raise FileNotFoundError(
            f"English text folder not found: {ENGLISH_TEST_PATH}"
        )

    if not JSON_FOLDER_PATH.exists():
        raise FileNotFoundError(
            f"JSON folder not found: {JSON_FOLDER_PATH}"
        )