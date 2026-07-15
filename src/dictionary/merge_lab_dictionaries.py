"""Gộp dictionary tự curate và dictionary BYT23, có sinh biến thể giới hạn.

Script đọc:
  1. data/dict/lab_test_seed.json
  2. data/dict/lab_imaging_from_byt23.json

Script ghi:
  - data/dict/lab_dict_merged.json
  - data/dict/merge_review_log.csv
  - data/dict/surface_form_conflicts.csv

Chỉ áp dụng đúng các rule sinh biến thể đã được yêu cầu. Không tự đoán
từ đồng nghĩa và không tự bịa viết tắt ngoài danh sách đã review.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from rapidfuzz import fuzz, process


LOGGER = logging.getLogger(__name__)

SOURCE_BYT23 = "Thong_tu_23_2024_TT_BYT_PhuLuc01"
FUZZY_MIN_SEED_KEY_LENGTH = 4
SAFE_SHORT_ABBREVIATIONS = {"ct", "mri", "hr", "bp", "rr", "sa"}

RULE1_PREFIXES = (
    "Định lượng nhanh ",
    "Xét nghiệm định lượng ",
    "Xét nghiệm định tính ",
    "Định lượng ",
    "Định tính ",
    "Đo ",
    "Ghi ",
)

RULE2_SUFFIXES = (
    " trong máu toàn phần tại chỗ bằng máy cầm tay",
    " tại giường",
    " cấp cứu",
    " thường quy",
    " bằng phương pháp ELISA",
    " bằng máy",
)

PAREN_VERB_BLOCKLIST = ("khi", "nếu", "trong trường hợp")

FIXED_ABBREVIATIONS = {
    "công thức máu": ("cbc",),
    "tổng phân tích nước tiểu": ("tptnt",),
    "siêu âm": ("sa",),
    "chụp cắt lớp vi tính": ("ct", "ct scan"),
    "chụp cộng hưởng từ": ("mri",),
    "điện tâm đồ": ("ecg", "đtđ"),
    "điện não đồ": ("eeg",),
}

MODALITY_GENERIC_TERMS = (
    {
        "canonical_id": "generic_noi_soi",
        "category": "imaging",
        "surface_forms": ("Nội soi", "nội soi"),
    },
    {
        "canonical_id": "generic_ct",
        "category": "imaging",
        "surface_forms": ("Chụp cắt lớp vi tính", "chụp cắt lớp vi tính", "Chụp CT", "chụp CT", "chụp ct", "ct"),
    },
    {
        "canonical_id": "generic_xquang",
        "category": "imaging",
        "surface_forms": ("Chụp X-quang", "chụp X-quang", "chụp x-quang"),
    },
    {
        "canonical_id": "generic_sieu_am",
        "category": "imaging",
        "surface_forms": ("Siêu âm", "siêu âm", "sa"),
    },
    {
        "canonical_id": "generic_mri",
        "category": "imaging",
        "surface_forms": ("Chụp cộng hưởng từ", "chụp cộng hưởng từ", "Chụp MRI", "chụp MRI", "chụp mri", "mri"),
    },
    {
        "canonical_id": "generic_dien_tam_do",
        "category": "lab_test",
        "surface_forms": ("Điện tâm đồ", "điện tâm đồ", "Điện tim", "điện tim", "ECG", "ecg"),
    },
    {
        "canonical_id": "generic_dien_nao_do",
        "category": "lab_test",
        "surface_forms": ("Điện não đồ", "điện não đồ", "Điện não", "điện não", "EEG", "eeg"),
    },
    {
        "canonical_id": "generic_ercp",
        "category": "imaging",
        "surface_forms": ("Nội soi mật tụy ngược dòng", "nội soi mật tụy ngược dòng", "ERCP", "ercp"),
    },
)


@dataclass
class UnifiedEntry:
    """Một entry sau khi gộp dictionary."""

    canonical_id: str
    category: str
    surface_forms: list[str]
    source_seed: bool = False
    byt23_ma_ky_thuat: list[str] = field(default_factory=list)
    canonical_name: str | None = None
    protected_surface_keys: set[str] = field(default_factory=set)
    surface_confidences: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SeedEntry:
    """Một entry từ dictionary seed đã curate tay."""

    canonical_id: str
    category: str
    surface_forms: tuple[str, ...]


@dataclass(frozen=True)
class BytEntry:
    """Một entry canonical trích từ Thông tư 23."""

    canonical_name: str
    chuong: str
    ma_ky_thuat: tuple[str, ...]


def normalize_for_match(value: str) -> str:
    """Chuẩn hóa để so khớp: lowercase, bỏ dấu/ký hiệu và gộp khoảng trắng."""

    decomposed = unicodedata.normalize("NFD", value)
    without_accents = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    normalized = without_accents.lower().replace("đ", "d")
    normalized = re.sub(r"[-‐‑‒–—_/]+", " ", normalized)
    normalized = re.sub(r"[()\\[\\],.;:]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def normalize_surface_key(value: str) -> str:
    """Chuẩn hóa surface form để kiểm tra trùng/conflict."""

    return re.sub(r"\s+", " ", value.strip().lower())


def stable_slug(value: str) -> str:
    """Tạo slug ASCII ổn định để dùng làm canonical_id."""

    normalized = normalize_for_match(value)
    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    return slug or "entry"


def load_seed(path: Path) -> list[SeedEntry]:
    """Đọc dictionary seed đã curate tay."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    entries: list[SeedEntry] = []
    for index, item in enumerate(payload):
        try:
            entries.append(
                SeedEntry(
                    canonical_id=str(item["canonical_id"]),
                    category=str(item["category"]),
                    surface_forms=tuple(str(form) for form in item["surface_forms"]),
                )
            )
        except KeyError as error:
            raise ValueError(f"{path}: seed item {index} missing key {error}") from error
    return entries


def load_byt23(path: Path) -> list[BytEntry]:
    """Đọc dictionary canonical trích từ BYT23."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    entries: list[BytEntry] = []
    for index, item in enumerate(payload):
        try:
            entries.append(
                BytEntry(
                    canonical_name=str(item["canonical_name"]),
                    chuong=str(item["chuong"]),
                    ma_ky_thuat=tuple(str(code) for code in item["ma_ky_thuat"]),
                )
            )
        except KeyError as error:
            raise ValueError(f"{path}: BYT23 item {index} missing key {error}") from error
    return entries


def infer_byt23_category(entry: BytEntry) -> str:
    """Suy ra category của entry BYT23 dựa trên tên chương."""

    chapter = normalize_for_match(entry.chuong)
    if any(keyword in chapter for keyword in ("dien quang", "chan doan hinh anh", "y hoc hat nhan")):
        return "imaging"
    return "lab_test"


def confidence_for_surface(value: str, default: str = "high") -> str:
    """Trả về độ tin cậy của surface form; viết tắt ngắn an toàn được hạ xuống low."""

    key = normalize_surface_key(value)
    if key in SAFE_SHORT_ABBREVIATIONS:
        return "low"
    return default


def add_unique_surface(
    forms: list[str],
    value: str,
    confidences: dict[str, str] | None = None,
    confidence: str = "high",
) -> bool:
    """Thêm surface form nếu không rỗng và chưa tồn tại trong entry."""

    cleaned = re.sub(r"\s+", " ", value.strip())
    if not cleaned:
        return False
    key = normalize_surface_key(cleaned)
    if any(normalize_surface_key(existing) == key for existing in forms):
        if confidences is not None:
            confidences.setdefault(key, confidence_for_surface(cleaned, confidence))
        return False
    forms.append(cleaned)
    if confidences is not None:
        confidences[key] = confidence_for_surface(cleaned, confidence)
    return True


def build_seed_entries(seed_entries: Iterable[SeedEntry]) -> list[UnifiedEntry]:
    """Chuyển entry seed sang schema hợp nhất."""

    unified: list[UnifiedEntry] = []
    for seed in seed_entries:
        forms: list[str] = []
        confidences: dict[str, str] = {}
        for form in seed.surface_forms:
            add_unique_surface(forms, form, confidences, "high")
        unified.append(
            UnifiedEntry(
                canonical_id=seed.canonical_id,
                category=seed.category,
                surface_forms=forms,
                source_seed=True,
                canonical_name=forms[0] if forms else seed.canonical_id,
                protected_surface_keys={normalize_surface_key(form) for form in seed.surface_forms},
                surface_confidences=confidences,
            )
        )
    return unified


def candidate_names_for_matching(byt: BytEntry) -> list[str]:
    """Tạo các tên BYT23 dùng riêng cho fuzzy match với seed, theo hướng thận trọng."""

    candidates = [byt.canonical_name]
    generated = generate_variants_for_form(byt.canonical_name)
    for variants in generated.values():
        candidates.extend(variants)

    normalized_candidates: list[str] = []
    for candidate in candidates:
        normalized = normalize_for_match(candidate)
        if normalized and normalized not in normalized_candidates:
            normalized_candidates.append(normalized)
    return normalized_candidates


def has_whole_phrase_match(haystack: str, needle: str) -> bool:
    """Kiểm tra needle có xuất hiện như một cụm tách bằng khoảng trắng không."""

    return re.search(rf"(^|\s){re.escape(needle)}($|\s)", haystack) is not None


def best_seed_match(byt: BytEntry, seed_choices: dict[str, SeedEntry], threshold: float) -> tuple[SeedEntry, float] | None:
    """Tìm entry seed phù hợp nhất cho một tên canonical BYT23.

    Ở đây cố ý dùng ``fuzz.ratio`` thay vì ``token_set_ratio``. 
    ``token_set_ratio`` thường cho điểm rất cao với các cụm ngắn như
    ``bạch cầu`` nằm trong tên kỹ thuật dài nhưng khác nghĩa, dễ gây gộp nhầm.
    Vì vậy script tạo trước các biến thể prefix/suffix đã duyệt, rồi mới so
    khớp chặt hơn bằng ratio.
    """

    if not seed_choices:
        return None

    best_key: str | None = None
    best_score = 0.0
    best_candidate: str | None = None
    for candidate in candidate_names_for_matching(byt):
        result = process.extractOne(candidate, seed_choices.keys(), scorer=fuzz.ratio)
        if result is None:
            continue
        matched_key, score, _ = result
        if score > best_score:
            best_key = matched_key
            best_score = float(score)
            best_candidate = candidate

    if best_key is None or best_score < threshold:
        return None
    if best_candidate is None or not has_whole_phrase_match(best_candidate, best_key):
        return None
    return seed_choices[best_key], best_score


def merge_sources(
    seed_entries: list[SeedEntry],
    byt_entries: list[BytEntry],
    threshold: float,
) -> tuple[list[UnifiedEntry], list[dict[str, object]], int]:
    """Gộp seed và BYT23 bằng fuzzy matching theo hướng thận trọng."""

    entries = build_seed_entries(seed_entries)
    entry_by_seed_id = {entry.canonical_id: entry for entry in entries}
    seed_choices: dict[str, SeedEntry] = {}
    for seed in seed_entries:
        if seed.category == "vital_sign":
            continue
        for form in seed.surface_forms:
            normalized = normalize_for_match(form)
            if len(normalized) < FUZZY_MIN_SEED_KEY_LENGTH:
                continue
            seed_choices[normalized] = seed

    review_rows: list[dict[str, object]] = []
    used_ids = {entry.canonical_id for entry in entries}
    before_merge_count = len(seed_entries) + len(byt_entries)

    for byt in byt_entries:
        match = best_seed_match(byt, seed_choices, threshold)
        if match is not None:
            seed, score = match
            entry = entry_by_seed_id[seed.canonical_id]
            add_unique_surface(entry.surface_forms, byt.canonical_name, entry.surface_confidences, "high")
            for code in byt.ma_ky_thuat:
                if code not in entry.byt23_ma_ky_thuat:
                    entry.byt23_ma_ky_thuat.append(code)
            review_rows.append(
                {
                    "seed_canonical_id": seed.canonical_id,
                    "seed_name": seed.surface_forms[0] if seed.surface_forms else seed.canonical_id,
                    "byt23_name": byt.canonical_name,
                    "similarity": f"{score:.2f}",
                    "byt23_ma_ky_thuat": ";".join(byt.ma_ky_thuat),
                }
            )
            continue

        base_slug = stable_slug(byt.canonical_name)
        slug = base_slug
        suffix = 2
        while slug in used_ids:
            slug = f"{base_slug}_{suffix}"
            suffix += 1
        used_ids.add(slug)
        entries.append(
            UnifiedEntry(
                canonical_id=slug,
                category=infer_byt23_category(byt),
                surface_forms=[byt.canonical_name],
                source_seed=False,
                byt23_ma_ky_thuat=list(byt.ma_ky_thuat),
                canonical_name=byt.canonical_name,
                surface_confidences={normalize_surface_key(byt.canonical_name): "high"},
            )
        )

    return entries, review_rows, before_merge_count


def strip_rule1_prefix(value: str) -> str | None:
    """Rule 1: bỏ một tiền tố hành động đã được duyệt."""

    value_lower = value.lower()
    for prefix in RULE1_PREFIXES:
        if value_lower.startswith(prefix.lower()):
            stripped = value[len(prefix) :].strip()
            return stripped or None
    return None


def strip_rule2_suffix(value: str) -> str | None:
    """Rule 2: bỏ một hậu tố phương pháp/vị trí đã được duyệt ở cuối tên."""

    value_lower = value.lower().strip()
    for suffix in RULE2_SUFFIXES:
        if value_lower.endswith(suffix.lower()):
            stripped = value[: -len(suffix)].strip(" ,;:-")
            return stripped or None
    return None


def extract_rule3_parentheses(value: str) -> list[str]:
    """Rule 3: lấy phần trong ngoặc nếu có vẻ là thuật ngữ độc lập an toàn."""

    variants: list[str] = []
    for match in re.finditer(r"\(([^()]{2,80})\)", value):
        inner = re.sub(r"\s+", " ", match.group(1).strip())
        normalized = normalize_for_match(inner)
        if not inner:
            continue
        if any(blocked in normalized for blocked in PAREN_VERB_BLOCKLIST):
            continue
        variants.append(inner)
    return variants


def fixed_abbreviation_variants(value: str) -> list[str]:
    """Rule 4: thêm viết tắt chỉ từ bảng tra cố định đã review."""

    normalized = normalize_for_match(value)
    variants: list[str] = []
    for key, abbreviations in FIXED_ABBREVIATIONS.items():
        normalized_key = normalize_for_match(key)
        if normalized_key not in normalized:
            continue
        if normalized_key == "sieu am" and not normalized.startswith("sieu am"):
            continue
        variants.extend(abbreviations)
    return variants


def generate_variants_for_form(form: str) -> dict[str, list[str]]:
    """Sinh biến thể đúng theo bốn nhóm rule đã được duyệt."""

    generated: dict[str, list[str]] = {"rule1": [], "rule2": [], "rule3": [], "rule4": []}

    rule1 = strip_rule1_prefix(form)
    if rule1:
        generated["rule1"].append(rule1)
        rule1_rule2 = strip_rule2_suffix(rule1)
        if rule1_rule2:
            generated["rule2"].append(rule1_rule2)

    rule2 = strip_rule2_suffix(form)
    if rule2:
        generated["rule2"].append(rule2)
        rule2_rule1 = strip_rule1_prefix(rule2)
        if rule2_rule1:
            generated["rule1"].append(rule2_rule1)

    generated["rule3"].extend(extract_rule3_parentheses(form))
    generated["rule4"].extend(fixed_abbreviation_variants(form))
    return generated


def add_generated_variants(entries: list[UnifiedEntry]) -> Counter[str]:
    """Sinh biến thể có giới hạn và trả về số lượng theo từng rule."""

    counts: Counter[str] = Counter()
    for entry in entries:
        original_forms = list(entry.surface_forms)
        for form in original_forms:
            generated = generate_variants_for_form(form)
            for rule_name, variants in generated.items():
                for variant in variants:
                    confidence = "medium" if rule_name == "rule3" else "high"
                    if add_unique_surface(entry.surface_forms, variant, entry.surface_confidences, confidence):
                        counts[rule_name] += 1
    return counts


def remove_short_forms(entries: list[UnifiedEntry], min_length: int = 3) -> None:
    """Loại surface form sinh tự động quá ngắn nếu không nằm trong danh sách an toàn.

    Surface form từ seed được giữ lại dù ngắn vì đã được review tay trước đó.
    Nhờ vậy không làm mất các match cũ như ``k``, ``hr``, ``bp`` và ``rr``.
    Một danh sách nhỏ các viết tắt imaging/vital đã review cũng được giữ lại
    và đánh dấu confidence thấp.
    """

    for entry in entries:
        kept_forms: list[str] = []
        kept_confidences: dict[str, str] = {}
        for form in entry.surface_forms:
            key = normalize_surface_key(form)
            keep = len(form.strip()) >= min_length or key in entry.protected_surface_keys or key in SAFE_SHORT_ABBREVIATIONS
            if not keep:
                continue
            kept_forms.append(form)
            kept_confidences[key] = confidence_for_surface(form, entry.surface_confidences.get(key, "high"))
        entry.surface_forms = kept_forms
        entry.surface_confidences = kept_confidences


def find_surface_conflicts(entries: list[UnifiedEntry]) -> list[dict[str, str]]:
    """Tìm surface form cùng xuất hiện ở nhiều canonical entry khác nhau."""

    owners: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for entry in entries:
        for form in entry.surface_forms:
            owners[normalize_surface_key(form)].append((entry.canonical_id, form))

    conflicts: list[dict[str, str]] = []
    for surface_key, hits in owners.items():
        canonical_ids = sorted({canonical_id for canonical_id, _ in hits})
        if len(canonical_ids) <= 1:
            continue
        conflicts.append(
            {
                "surface_form": surface_key,
                "canonical_ids": ";".join(canonical_ids),
                "forms": ";".join(sorted({form for _, form in hits})),
            }
        )
    return sorted(conflicts, key=lambda item: item["surface_form"])


def remove_conflicting_forms(entries: list[UnifiedEntry], conflicts: list[dict[str, str]]) -> None:
    """Loại surface form bị conflict khỏi tất cả entry liên quan."""

    conflict_keys = {item["surface_form"] for item in conflicts}
    for entry in entries:
        entry.surface_forms = [form for form in entry.surface_forms if normalize_surface_key(form) not in conflict_keys]
        entry.surface_confidences = {
            key: confidence for key, confidence in entry.surface_confidences.items() if key not in conflict_keys
        }


def infer_generic_category(surface_forms: Iterable[str]) -> str:
    """Suy ra category cho entry generic dựa trên surface form."""

    normalized = " ".join(normalize_for_match(form) for form in surface_forms)
    if any(keyword in normalized for keyword in ("sieu am", "chup", "noi soi", "mri", "ct", "x quang")):
        return "imaging"
    return "lab_test"


def upsert_generic_entry(
    entries: list[UnifiedEntry],
    used_ids: set[str],
    canonical_id: str,
    category: str,
    surface_forms: Iterable[str],
) -> bool:
    """Thêm mới hoặc cập nhật entry generic đã curate sau bước lọc conflict."""

    entry = next((candidate for candidate in entries if candidate.canonical_id == canonical_id), None)
    created = False
    if entry is None:
        entry = UnifiedEntry(
            canonical_id=canonical_id,
            category=category,
            surface_forms=[],
            source_seed=False,
            byt23_ma_ky_thuat=[],
            canonical_name=canonical_id,
        )
        entries.append(entry)
        used_ids.add(canonical_id)
        created = True

    entry.category = category
    for form in surface_forms:
        add_unique_surface(entry.surface_forms, form, entry.surface_confidences, "high")
    return created


def add_curated_generic_entries(entries: list[UnifiedEntry]) -> int:
    """Thêm các entry modality generic đã review tay sau dedupe/conflict."""

    used_ids = {entry.canonical_id for entry in entries}
    created = 0
    for item in MODALITY_GENERIC_TERMS:
        if upsert_generic_entry(
            entries,
            used_ids,
            str(item["canonical_id"]),
            str(item["category"]),
            item["surface_forms"],
        ):
            created += 1
    return created


def add_missing_rule4_generic_entries(entries: list[UnifiedEntry]) -> int:
    """Thêm entry generic cho các cặp Rule 4 còn vắng hoàn toàn.

    Đây là lớp bảo hiểm hai chiều cho bảng viết tắt cố định: nếu cả key và
    toàn bộ viết tắt đã review đều chưa xuất hiện ở đâu, tạo một entry nhỏ
    chỉ gồm đúng key và các value tương ứng.
    """

    all_surface_keys = {
        normalize_surface_key(form)
        for entry in entries
        for form in entry.surface_forms
    }
    used_ids = {entry.canonical_id for entry in entries}
    created = 0
    for key, values in FIXED_ABBREVIATIONS.items():
        candidates = (key, *values)
        if any(normalize_surface_key(candidate) in all_surface_keys for candidate in candidates):
            continue
        base_slug = f"generic_{stable_slug(key)}"
        slug = base_slug
        suffix = 2
        while slug in used_ids:
            slug = f"{base_slug}_{suffix}"
            suffix += 1
        if upsert_generic_entry(entries, used_ids, slug, infer_generic_category(candidates), candidates):
            created += 1
            all_surface_keys.update(normalize_surface_key(candidate) for candidate in candidates)
    return created


def log_biliary_endoscopy_surfaces(entries: list[UnifiedEntry]) -> None:
    """Log các entry có nhắc tới Nội soi mật tụy ngược dòng để debug."""

    LOGGER.info("========== ERCP / MẬT TỤY NGƯỢC DÒNG DEBUG ==========")
    found = False
    for entry in entries:
        if any("mật tụy ngược dòng" in form.lower() or "ercp" in form.lower() for form in entry.surface_forms):
            found = True
            LOGGER.info("%s | %s | %s", entry.canonical_id, entry.category, "; ".join(entry.surface_forms))
    if not found:
        LOGGER.info("(no existing ERCP/mật tụy ngược dòng surface forms found before generic upsert)")


def to_jsonable(entries: Iterable[UnifiedEntry]) -> list[dict[str, object]]:
    """Chuyển entry nội bộ sang schema JSON đầu ra."""

    payload: list[dict[str, object]] = []
    for entry in entries:
        if not entry.surface_forms:
            continue
        payload.append(
            {
                "canonical_id": entry.canonical_id,
                "category": entry.category,
                "surface_forms": sorted(entry.surface_forms, key=lambda value: (normalize_surface_key(value), value)),
                "source_refs": {
                    "seed": entry.source_seed,
                    "byt23_ma_ky_thuat": sorted(entry.byt23_ma_ky_thuat),
                },
                "surface_form_confidence": {
                    form: entry.surface_confidences.get(normalize_surface_key(form), confidence_for_surface(form))
                    for form in sorted(entry.surface_forms, key=lambda value: (normalize_surface_key(value), value))
                },
            }
        )
    return sorted(payload, key=lambda item: str(item["canonical_id"]))


def write_json(path: Path, payload: object) -> None:
    """Ghi JSON bằng UTF-8."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Ghi CSV bằng UTF-8."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def log_review_samples(entries: list[dict[str, object]], sample_size: int, seed: int) -> None:
    """Log ngẫu nhiên các entry có nhiều surface form để review nhanh."""

    ranked = sorted(entries, key=lambda item: len(item["surface_forms"]), reverse=True)
    pool = ranked[: min(len(ranked), max(sample_size * 3, sample_size))]
    rng = random.Random(seed)
    LOGGER.info("========== REVIEW SAMPLE: MANY SURFACE FORMS ==========")
    for index, item in enumerate(rng.sample(pool, min(sample_size, len(pool))), start=1):
        LOGGER.info(
            "%02d. %s | %s | %s",
            index,
            item["canonical_id"],
            item["category"],
            "; ".join(str(form) for form in item["surface_forms"]),
        )


def run(
    seed_path: Path,
    byt23_path: Path,
    output_path: Path,
    review_log_path: Path,
    conflict_log_path: Path,
    threshold: float,
    sample_size: int,
    seed: int,
) -> list[dict[str, object]]:
    """Chạy merge, sinh biến thể, loại conflict và in báo cáo."""

    seed_entries = load_seed(seed_path)
    byt_entries = load_byt23(byt23_path)
    entries, review_rows, before_merge_count = merge_sources(seed_entries, byt_entries, threshold)
    after_merge_count = len(entries)
    variant_counts = add_generated_variants(entries)
    remove_short_forms(entries)
    conflicts = find_surface_conflicts(entries)
    remove_conflicting_forms(entries, conflicts)
    log_biliary_endoscopy_surfaces(entries)
    generic_created = add_curated_generic_entries(entries)
    rule4_generic_created = add_missing_rule4_generic_entries(entries)
    dropped_empty_entries = sum(1 for entry in entries if not entry.surface_forms)
    payload = to_jsonable(entries)

    write_json(output_path, payload)
    write_csv(
        review_log_path,
        review_rows,
        ["seed_canonical_id", "seed_name", "byt23_name", "similarity", "byt23_ma_ky_thuat"],
    )
    write_csv(conflict_log_path, conflicts, ["surface_form", "canonical_ids", "forms"])

    LOGGER.info("========== LAB DICTIONARY MERGE ==========")
    LOGGER.info("Seed entries: %s", len(seed_entries))
    LOGGER.info("BYT23 entries: %s", len(byt_entries))
    LOGGER.info("Total entries before fuzzy merge: %s", before_merge_count)
    LOGGER.info("Total entries after fuzzy merge: %s", after_merge_count)
    LOGGER.info("Auto-merged fuzzy pairs: %s (%s)", len(review_rows), review_log_path)
    LOGGER.info("Generated variants rule1: %s", variant_counts["rule1"])
    LOGGER.info("Generated variants rule2: %s", variant_counts["rule2"])
    LOGGER.info("Generated variants rule3: %s", variant_counts["rule3"])
    LOGGER.info("Generated variants rule4: %s", variant_counts["rule4"])
    LOGGER.info("Surface-form conflicts: %s (%s)", len(conflicts), conflict_log_path)
    LOGGER.info("Curated generic entries created: %s", generic_created)
    LOGGER.info("Rule4 missing generic entries created: %s", rule4_generic_created)
    LOGGER.info("Dropped empty-surface entries after conflict removal: %s", dropped_empty_entries)
    LOGGER.info("Final output entries: %s (%s)", len(payload), output_path)
    log_review_samples(payload, sample_size, seed)
    LOGGER.info("Review merge_review_log.csv and surface_form_conflicts.csv before production use.")
    return payload


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""

    parser = argparse.ArgumentParser(description="Merge lab dictionaries and generate limited surface variants.")
    parser.add_argument("--seed", type=Path, default=Path("data/dict/lab_test_seed.json"), help="Curated seed dictionary.")
    parser.add_argument(
        "--byt23",
        type=Path,
        default=Path("data/dict/lab_imaging_from_byt23.json"),
        help="Canonical dictionary extracted from BYT23.",
    )
    parser.add_argument("--output", type=Path, default=Path("data/dict/lab_dict_merged.json"))
    parser.add_argument("--review-log", type=Path, default=Path("data/dict/merge_review_log.csv"))
    parser.add_argument("--conflict-log", type=Path, default=Path("data/dict/surface_form_conflicts.csv"))
    parser.add_argument("--threshold", type=float, default=85.0, help="RapidFuzz similarity threshold.")
    parser.add_argument("--sample-size", type=int, default=20, help="Number of review samples to print.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for report samples.")
    return parser.parse_args()


def main() -> None:
    """Điểm vào khi chạy từ dòng lệnh."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    run(
        seed_path=args.seed,
        byt23_path=args.byt23,
        output_path=args.output,
        review_log_path=args.review_log,
        conflict_log_path=args.conflict_log,
        threshold=args.threshold,
        sample_size=args.sample_size,
        seed=args.random_seed,
    )


if __name__ == "__main__":
    main()
