"""Rule-based THUỐC entity extraction from English-heavy medical text."""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


LOGGER = logging.getLogger(__name__)

RXNORM_TTYS_OF_INTEREST = {"IN", "PIN", "BN", "SCD", "SBD"}
INGREDIENT_TTYS = {"IN", "PIN"}
NAME_MATCH_TTYS = {"IN", "PIN", "BN", "SCD", "SBD"}
TRUSTED_STANDALONE_TTYS = {"BN", "SCD", "SBD"}
ENTITY_TYPE = "THUỐC"
CACHE_VERSION = 3

# RxNorm contains broad ingredient/common-chemistry strings. This blacklist is
# now only a fallback for special cases that TTY filtering does not fully handle;
# it is not the primary filtering mechanism.
DRUG_NAME_BLACKLIST = {
    "albumin",
    "bilirubin",
    "calcium",
    "chloride",
    "cholesterol",
    "creatinine",
    "glucose",
    "hemoglobin",
    "kali",
    "magnesium",
    "phosphate",
    "phosphorus",
    "potassium",
    "protein",
    "sodium",
    "triglycerides",
    "urea",
}

DOSE_RE = re.compile(r"\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu|units?)\b", re.IGNORECASE)
ROUTE_RE = re.compile(r"\s+(?:po|iv|im|sc|sq|pr|sl|top|inh)\b", re.IGNORECASE)
FREQUENCY_RE = re.compile(
    r"\s+(?:daily|bid|tid|qid|qhs|qam|prn|q\d+h|once|weekly)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DrugNameMatch:
    """One matched drug-name span."""

    start: int
    end: int
    text: str
    normalized_name: str
    entries: tuple[tuple[str, str], ...]


class RegexDrugNameMatcher:
    """Regex-based drug-name matcher.

    The matcher uses escaped regex alternations sorted by decreasing length so
    longer names are preferred over shorter substrings. It also splits the
    dictionary into chunks to avoid building one enormous regex. This keeps the
    module dependency-free and easy to debug. For very large dictionaries,
    Aho-Corasick is faster, but this regex matcher is portable and sufficient
    for a pure rule-based baseline.
    """

    def __init__(
        self,
        names: Iterable[str] | dict[str, list[tuple[str, str]]],
        chunk_size: int = 1000,
    ) -> None:
        """Build compiled regex chunks from normalized drug names."""

        if isinstance(names, dict):
            self.entries_by_name = {
                normalize_rxnorm_string(name): tuple(entries)
                for name, entries in names.items()
                if normalize_rxnorm_string(name)
            }
        else:
            self.entries_by_name = {
                normalize_rxnorm_string(name): (("", "BN"),)
                for name in names
                if normalize_rxnorm_string(name)
            }

        normalized_names = sorted(self.entries_by_name, key=lambda value: (-len(value), value))
        self.patterns = [
            re.compile(
                r"(?<!\w)(?P<drug>" + "|".join(term_to_regex(name) for name in chunk) + r")(?!\w)",
                re.IGNORECASE,
            )
            for chunk in chunked(normalized_names, chunk_size)
            if chunk
        ]

    def find(self, text: str) -> list[DrugNameMatch]:
        """Return non-overlapping drug-name matches sorted by start position."""

        raw_matches: list[DrugNameMatch] = []
        for pattern in self.patterns:
            for match in pattern.finditer(text):
                normalized_name = normalize_rxnorm_string(match.group("drug"))
                raw_matches.append(
                    DrugNameMatch(
                        start=match.start(),
                        end=match.end(),
                        text=text[match.start() : match.end()],
                        normalized_name=normalized_name,
                        entries=self.entries_by_name.get(normalized_name, ()),
                    )
                )

        return select_longest_non_overlapping(raw_matches)


def normalize_rxnorm_string(value: str) -> str:
    """Normalize an RxNorm string for dictionary lookup and matching."""

    return re.sub(r"\s+", " ", value.strip().lower())


def term_to_regex(term: str) -> str:
    """Convert a normalized dictionary term into a whitespace-tolerant regex."""

    return r"\s+".join(re.escape(part) for part in term.split())


def chunked(values: list[str], size: int) -> Iterable[list[str]]:
    """Yield list chunks of a fixed maximum size."""

    for start in range(0, len(values), size):
        yield values[start : start + size]


def select_longest_non_overlapping(matches: Iterable[DrugNameMatch]) -> list[DrugNameMatch]:
    """Keep the longest match when drug-name matches overlap."""

    selected: list[DrugNameMatch] = []
    for match in sorted(matches, key=lambda item: (-(item.end - item.start), item.start)):
        if any(match.start < kept.end and kept.start < match.end for kept in selected):
            continue
        selected.append(match)
    return sorted(selected, key=lambda item: (item.start, item.end))


def load_rxnorm_dict(rrf_path: Path) -> tuple[dict[str, list[tuple[str, str]]], dict[str, list[tuple[str, str]]]]:
    """Load a THUỐC dictionary from RxNorm RXNCONSO.RRF.

    Only English rows are kept (``LAT == "ENG"``), suppressed rows are skipped
    (``SUPPRESS == "Y"``), and the selected term types are:

    - ``IN``: Ingredient, useful for generic drug names such as amlodipine.
    - ``PIN``: Precise Ingredient, useful for more specific ingredient forms.
    - ``BN``: Brand Name, useful when notes mention a trade name.
    - ``SCD``: Semantic Clinical Drug, useful because it often already includes
      name + dose + form, matching clinical text directly.
    - ``SBD``: Semantic Branded Drug, the branded equivalent of SCD.

    ``SCD``/``SBD`` improve direct phrase matches for strings like
    "amlodipine 10 MG Oral Tablet"; ``IN``/``PIN``/``BN`` provide fallback
    names when notes use shorter medication mentions.

    Returns:
        A tuple ``(rxnorm_dict, matcher_entries)``. Both dictionaries map a
        normalized STR to ``[(RXCUI, TTY), ...]`` because one name can appear
        under multiple TTY sources. ``matcher_entries`` keeps ``IN/PIN/BN/SCD/SBD``
        names for text matching, while downstream filtering uses the TTY list.
    """

    rxnorm_dict: dict[str, list[tuple[str, str]]] = {}
    matcher_entries: dict[str, list[tuple[str, str]]] = {}
    counts_by_tty: Counter[str] = Counter()

    with rrf_path.open("r", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            columns = line.rstrip("\n").split("|")
            if len(columns) < 17:
                LOGGER.warning("Skipping malformed RXNCONSO row %s", line_number)
                continue

            rxcui = columns[0]
            language = columns[1]
            tty = columns[12]
            drug_string = columns[14]
            suppress = columns[16]

            if language != "ENG" or tty not in RXNORM_TTYS_OF_INTEREST or suppress == "Y":
                continue

            normalized = normalize_rxnorm_string(drug_string)
            if not normalized:
                continue

            rxnorm_dict.setdefault(normalized, [])
            entry = (rxcui, tty)
            if entry not in rxnorm_dict[normalized]:
                rxnorm_dict[normalized].append(entry)
            if tty in NAME_MATCH_TTYS:
                matcher_entries.setdefault(normalized, [])
                if entry not in matcher_entries[normalized]:
                    matcher_entries[normalized].append(entry)
            counts_by_tty[tty] += 1

    LOGGER.info("========== RxNorm THUỐC dictionary ==========")
    for tty in sorted(RXNORM_TTYS_OF_INTEREST):
        LOGGER.info("%s: %s", tty, counts_by_tty.get(tty, 0))
    LOGGER.info("Unique STR entries: %s", len(rxnorm_dict))
    LOGGER.info("Drug names for matching: %s", len(matcher_entries))

    return rxnorm_dict, matcher_entries


def save_dict_cache(
    data: tuple[dict[str, list[tuple[str, str]]], dict[str, list[tuple[str, str]]]],
    path: Path,
) -> None:
    """Save the RxNorm dictionary and ingredient set to a pickle cache."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(
            {
                "version": CACHE_VERSION,
                "rxnorm_dict": data[0],
                "matcher_entries": data[1],
            },
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def load_dict_cache(path: Path) -> tuple[dict[str, list[tuple[str, str]]], dict[str, list[tuple[str, str]]]]:
    """Load the RxNorm dictionary and ingredient set from a pickle cache."""

    with path.open("rb") as file:
        payload = pickle.load(file)
    if payload.get("version") != CACHE_VERSION:
        raise ValueError(f"Outdated RxNorm cache version in {path}; rebuild required.")
    return payload["rxnorm_dict"], payload["matcher_entries"]


def build_drug_name_matcher(
    ingredient_set: set[str] | dict[str, list[tuple[str, str]]],
) -> RegexDrugNameMatcher:
    """Build a case-insensitive word-boundary matcher for drug names.

    This implementation uses a dependency-free regex union split into chunks.
    The API intentionally stays small, so it can later be swapped for
    pyahocorasick when matching a very large dictionary becomes the bottleneck.
    """

    return RegexDrugNameMatcher(ingredient_set)


def extend_span_right(text: str, end: int) -> tuple[int, bool]:
    """Extend a drug-name span through dose, route, and frequency in order."""

    cursor = end
    expanded = False
    dose_match = DOSE_RE.match(text, cursor)
    if not dose_match:
        return cursor, expanded

    expanded = True
    cursor = dose_match.end()
    route_match = ROUTE_RE.match(text, cursor)
    if route_match:
        expanded = True
        cursor = route_match.end()

    frequency_match = FREQUENCY_RE.match(text, cursor)
    if frequency_match:
        expanded = True
        cursor = frequency_match.end()

    return cursor, expanded


def is_blocked_drug_name(name: str) -> bool:
    """Return whether a matched standalone name should be filtered out.

    This blacklist removes common lab/chemistry terms that appear in RxNorm but
    are usually not medication entities in clinical notes, for example
    creatinine or potassium/kali. Caffeine is intentionally not blocked because
    it can be either a substance exposure or a medication-like mention depending
    on the project taxonomy.
    """

    return normalize_rxnorm_string(name) in DRUG_NAME_BLACKLIST


def ttys_for_match(match: DrugNameMatch) -> set[str]:
    """Return the set of RxNorm TTY values supporting a name match."""

    return {tty for _, tty in match.entries}


def should_keep_match(match: DrugNameMatch, has_dose_route_or_frequency: bool) -> tuple[bool, str | None]:
    """Apply final TTY-aware filtering for a matched drug name."""

    if is_blocked_drug_name(match.text):
        return False, "blacklist fallback"

    ttys = ttys_for_match(match)
    if ttys & TRUSTED_STANDALONE_TTYS:
        return True, None

    if ttys and ttys <= INGREDIENT_TTYS and has_dose_route_or_frequency:
        return True, None

    if ttys and ttys <= INGREDIENT_TTYS:
        return False, "IN/PIN standalone, no dose/route/frequency"

    return False, "unsupported TTY source"


def extract_drug_mentions(text: str, matcher: RegexDrugNameMatcher) -> list[dict[str, object]]:
    """Extract THUỐC mentions from text.

    The extractor first matches drug names, then extends each span to the right
    through an optional clinical medication pattern in the strict order:
    ``dose -> route -> frequency``. It never extends through conjunctions such
    as "and" into the next drug mention, because only the three allowed pattern
    groups can be consumed after the drug name.
    """

    entities: list[dict[str, object]] = []
    for match in matcher.find(text):
        end, expanded = extend_span_right(text, match.end)
        keep, reason = should_keep_match(match, expanded)
        if not keep:
            LOGGER.debug("Filtered %s: %s", match.text, reason)
            continue

        entities.append(
            {
                "text": text[match.start : end],
                "type": ENTITY_TYPE,
                "position": [match.start, end],
            }
        )

    return sorted(entities, key=lambda entity: (entity["position"][0], entity["position"][1]))


def analyze_ttv_filtering(
    entities_before: dict[str, list[dict[str, object]]],
    entities_after: dict[str, list[dict[str, object]]],
) -> None:
    """Print a before/after report for TTY-aware filtering.

    The function compares entity dictionaries keyed by file name and reports
    entities removed by the new rule. Removed blacklist terms are counted to
    estimate how much manual blacklist filtering is now covered by the TTY rule.
    """

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    before_total = sum(len(entities) for entities in entities_before.values())
    after_total = sum(len(entities) for entities in entities_after.values())
    removed: list[tuple[str, dict[str, object], str]] = []

    for file_name, before_entities in entities_before.items():
        after_keys = {
            (entity["text"], tuple(entity["position"]), entity["type"])
            for entity in entities_after.get(file_name, [])
        }
        for entity in before_entities:
            key = (entity["text"], tuple(entity["position"]), entity["type"])
            if key not in after_keys:
                reason = "IN/PIN standalone, no dose/route/frequency"
                removed.append((file_name, entity, reason))

    blacklist_removed = sum(
        1 for _, entity, _ in removed if normalize_rxnorm_string(str(entity["text"])) in DRUG_NAME_BLACKLIST
    )

    print("========== TTY FILTERING ANALYSIS ==========")
    print(f"Total entities before: {before_total}")
    print(f"Total entities after: {after_total}")
    print("Removed entities:")
    for file_name, entity, reason in removed:
        print(f"- {file_name}: {entity['text']} {entity['position']} | {reason}")
    print(
        f"{blacklist_removed}/{len(removed)} entity bị loại trước đây phải chặn tay "
        "bằng blacklist, nay rule TTY tự động/chung cơ chế chặn được"
    )


def extract_file(input_path: Path, output_path: Path, matcher: RegexDrugNameMatcher) -> None:
    """Extract THUỐC entities from one UTF-8 text file and write JSON output."""

    text = input_path.read_text(encoding="utf-8-sig")
    entities = extract_drug_mentions(text, matcher)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(entities, file, ensure_ascii=False, indent=2)
        file.write("\n")


def run_cli(rxnorm_path: Path, input_path: Path, output_path: Path, cache_path: Path | None) -> None:
    """Run dictionary loading and extraction for one file or a directory."""

    if cache_path is not None and cache_path.is_file():
        try:
            _, ingredient_names = load_dict_cache(cache_path)
            LOGGER.info("Loaded RxNorm cache from %s", cache_path)
        except ValueError as error:
            LOGGER.info("%s", error)
            data = load_rxnorm_dict(rxnorm_path)
            _, ingredient_names = data
            save_dict_cache(data, cache_path)
            LOGGER.info("Rebuilt RxNorm cache at %s", cache_path)
    else:
        data = load_rxnorm_dict(rxnorm_path)
        _, ingredient_names = data
        if cache_path is not None:
            save_dict_cache(data, cache_path)
            LOGGER.info("Saved RxNorm cache to %s", cache_path)

    matcher = build_drug_name_matcher(ingredient_names)

    if input_path.is_file():
        extract_file(input_path, output_path, matcher)
        LOGGER.info("Wrote %s", output_path)
        return

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    for text_file in sorted(input_path.glob("*.txt")):
        target = output_path / f"{text_file.stem}.json"
        extract_file(text_file, target, matcher)
        LOGGER.info("Wrote %s", target)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Extract THUỐC entities with RxNorm rules.")
    parser.add_argument("--rxnorm", required=True, type=Path, help="Path to RXNCONSO.RRF.")
    parser.add_argument("--input", required=True, type=Path, help="Input .txt file or directory.")
    parser.add_argument("--output", required=True, type=Path, help="Output .json file or directory.")
    parser.add_argument("--cache", type=Path, help="Optional pickle cache path for RxNorm dictionary.")
    return parser.parse_args()


def main() -> None:
    """Run the THUỐC extractor CLI."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    run_cli(args.rxnorm, args.input, args.output, args.cache)


if __name__ == "__main__":
    main()
