"""Deterministically fuse SapBERT and RapidFuzz linking candidates.

SapBERT is the stronger semantic retriever. RapidFuzz is most useful as
supporting evidence, or as an override when it has one unique exact lexical
match. Scores are also adjusted by top-result margins so a decisive result is
preferred over a near tie.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

JSON_FOLDER_PATH = PROJECT_ROOT / "data" / "new_ner_output"

SUPPORTED_TYPES = {"THUỐC", "CHẨN_ĐOÁN"}
TYPE_WEIGHTS = {
    # Semantic similarity is more reliable for diagnoses and paraphrases.
    "CHẨN_ĐOÁN": (0.90, 0.10),
    # Lexical evidence matters more for drug names, brands, and strengths.
    "THUỐC": (0.60, 0.40),
}
AGREEMENT_BONUS = 0.06
# Do not reward a fuzzy top hit when SapBERT ranks it far below its own top
# result (for example, neonatal UTI for the generic query "urinary tract
# infection").
MAX_SAPBERT_GAP_FOR_AGREEMENT = 0.08
SAPBERT_MARGIN_SCALE = 0.15
RAPIDFUZZ_MARGIN_SCALE = 0.10
MAX_MARGIN_BONUS_INPUT = 0.20
EXACT_FUZZ_SCORE = 0.995
EXACT_FUZZ_MIN_MARGIN = 0.02


def _clean_results(value: object) -> dict[str, float]:
    """Return valid code-score pairs with scores clamped to ``[0, 1]``."""
    if not isinstance(value, dict):
        return {}

    cleaned: dict[str, float] = {}
    for raw_code, raw_score in value.items():
        code = str(raw_code).strip()
        if not code:
            continue
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score):
            continue
        cleaned[code] = max(0.0, min(1.0, score))
    return cleaned


def _ranked(results: dict[str, float]) -> list[tuple[str, float]]:
    """Sort by score and then code so equal scores are reproducible."""
    return sorted(results.items(), key=lambda item: (-item[1], item[0]))


def _top_and_margin(
    results: dict[str, float],
) -> tuple[str | None, float, float]:
    ranked = _ranked(results)
    if not ranked:
        return None, 0.0, 0.0
    code, score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    return code, score, max(0.0, score - runner_up)


def fuse_entity_result(entity: dict) -> None:
    """Update only the entity's ``candidate`` field."""
    if entity.get("type") not in SUPPORTED_TYPES:
        entity["candidate"] = None
        return

    retriever = entity.get("retriever", {})
    if not isinstance(retriever, dict):
        entity["candidate"] = None
        return

    sapbert = _clean_results(retriever.get("sapbert"))
    rapidfuzz = _clean_results(retriever.get("rapidfuzz"))
    all_codes = set(sapbert) | set(rapidfuzz)
    if not all_codes:
        entity["candidate"] = None
        return

    sap_top, sap_top_score, sap_margin = _top_and_margin(sapbert)
    fuzz_top, fuzz_top_score, fuzz_margin = _top_and_margin(rapidfuzz)

    # A unique exact string match is stronger evidence than a conflicting
    # semantic neighbour. Tied WRatio=1.0 results are deliberately excluded.
    if (
        fuzz_top is not None
        and fuzz_top_score >= EXACT_FUZZ_SCORE
        and fuzz_margin >= EXACT_FUZZ_MIN_MARGIN
        and fuzz_top != sap_top
    ):
        entity["candidate"] = {
            "code": fuzz_top,
            "confidence": round(fuzz_top_score, 6),
        }
        return

    sap_weight, fuzz_weight = TYPE_WEIGHTS[entity["type"]]
    scored: list[tuple[str, float, bool, float, float]] = []
    for code in all_codes:
        sap_score = sapbert.get(code, 0.0)
        fuzz_score = rapidfuzz.get(code, 0.0)
        agreed = code in sapbert and code in rapidfuzz
        supported_agreement = agreed and (
            code == sap_top
            or sapbert[code] >= sap_top_score - MAX_SAPBERT_GAP_FOR_AGREEMENT
        )
        score = sap_weight * sap_score + fuzz_weight * fuzz_score

        if supported_agreement:
            score += AGREEMENT_BONUS
        if code == sap_top:
            score += (
                min(sap_margin, MAX_MARGIN_BONUS_INPUT)
                * SAPBERT_MARGIN_SCALE
            )
        if code == fuzz_top:
            score += (
                min(fuzz_margin, MAX_MARGIN_BONUS_INPUT)
                * RAPIDFUZZ_MARGIN_SCALE
            )

        scored.append(
            (
                code,
                max(0.0, min(1.0, score)),
                supported_agreement,
                sap_score,
                fuzz_score,
            )
        )

    # Deterministic ordering is important: the old set/max combination could
    # select different codes across processes when final scores tied.
    scored.sort(
        key=lambda item: (
            -item[1],
            -int(item[2]),
            -item[3],
            -item[4],
            item[0],
        )
    )
    best_code, best_score, _, _, _ = scored[0]
    entity["candidate"] = {
        "code": best_code,
        "confidence": round(best_score, 6),
    }


def fusing_result(json_files: list[Path]) -> None:
    print(f"Found {len(json_files)} JSON files.")
    for file_path in json_files:
        print(f"Processing file: {file_path.name}")
        data = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or not all(
            isinstance(entity, dict) for entity in data
        ):
            raise ValueError(f"Expected a JSON list of objects: {file_path.name}")

        for entity in data:
            fuse_entity_result(entity)

        temporary = file_path.with_name(f".{file_path.name}.tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(file_path)
        print(f"Saved updated data to: {file_path.name}")


def json_sort_key(path: Path) -> tuple[int, int | str]:
    try:
        return 0, int(path.stem)
    except ValueError:
        return 1, path.stem.casefold()


def main() -> None:
    if not JSON_FOLDER_PATH.is_dir():
        raise FileNotFoundError(f"Folder does not exist: {JSON_FOLDER_PATH}")
    json_files = sorted(JSON_FOLDER_PATH.glob("*.json"), key=json_sort_key)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {JSON_FOLDER_PATH}")
    fusing_result(json_files)


if __name__ == "__main__":
    main()
