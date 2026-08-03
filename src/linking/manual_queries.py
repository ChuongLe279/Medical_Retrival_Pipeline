"""Build and apply deterministic, manually curated linking queries.

This module never loads a language model.  It uses one conservative canonical
English query per valid mention and an empty list for unusable NER fragments.
Only ``eng_translation`` is changed when the mapping is applied.
"""

from __future__ import annotations

import argparse
import copy
import json
import unicodedata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON_FOLDER = PROJECT_ROOT / "data" / "new_ner_output"
DEFAULT_MAPPING_PATH = Path(__file__).with_name("manual_queries.json")
SUPPORTED_TYPES = {"THUỐC", "CHẨN_ĐOÁN"}


def normalize_key(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


INVALID_DIAGNOSES = {
    "bình", "bệnh", "bệnh dạ", "bệnh dạibao", "bệnh dạithường",
    "c", "châ", "chậ", "chư", "chứng", "có", "cảm", "dày", "dị",
    "gan", "giãn động", "ho", "hohl", "hoại", "huyết", "hội", "iii",
    "kinh", "loạn", "miễn", "màng cứng", "mô", "mô tế bào", "nhiễm",
    "phế nang", "quai ruột", "rụng", "suy", "sơ sinh", "thấ", "tim",
    "tiêu", "tri", "trị", "trị kiê", "tính", "tăng", "tăng số lươ", "u châ", "u chậ",
    "u trị phu", "ung bướu", "vi", "xuất", "đông", "̉", "̉n giâ", "̣",
}


DIAGNOSIS_OVERRIDES = {
    "bón mãn tính": "Chronic constipation",
    "bạn đay vô căn hộ": "Idiopathic urticaria",
    "bệnh amyloidosis tự miễn dịch": "Amyloidosis",
    "bệnh ba thân động mạch vành": "Three-vessel coronary artery disease",
    "bệnh bàn chân bẹt": "Congenital pes planus",
    "bệnh bạch cầu": "Chronic myeloid leukemia",
    "bệnh bạch cầu dòng tủy mãn tính": "Chronic myeloid leukemia",
    "bệnh dạ dày": "Stomach disease",
    "bệnh dại": "Rabies",
    "bệnh gan do rượu": "Alcoholic liver disease",
    "bệnh gout mạn tính": "Chronic gout",
    "bệnh graves": "Graves disease",
    "bệnh gút": "Gout",
    "bệnh hô hấp": "Respiratory disease",
    "bệnh kawasaki": "Kawasaki disease",
    "bệnh mạch máu ngoại biên": "Peripheral vascular disease",
    "bệnh mạch vành": "Coronary artery disease",
    "bệnh mạn tính": "",
    "bệnh nhiễm trùng": "Infection",
    "bệnh nấm bẹn": "Tinea cruris",
    "bệnh phổi kẽ": "Interstitial lung disease",
    "bệnh phổi tắc nghẽn mạn tính": "Chronic obstructive pulmonary disease",
    "bệnh thoái hóa tinh bột": "Amyloidosis",
    "bệnh thận mạn": "Chronic kidney disease",
    "bệnh thận mạn tính": "Chronic kidney disease",
    "bệnh thủy đậu": "Varicella",
    "bệnh thủy đậu/zona": "Varicella or herpes zoster",
    "bệnh tim mạch": "Cardiovascular disease",
    "bệnh tiểu đường": "Diabetes mellitus",
    "bệnh trĩ": "Hemorrhoids",
    "bệnh tăng ha vô căn": "Essential hypertension",
    "bệnh tăng đông máu": "Thrombophilia",
    "bệnh u nang tuyến vú": "Breast cyst",
    "bệnh ung thư": "Cancer",
    "bệnh ung thư máu": "Leukemia",
    "bệnh viêm mạch máu": "Vasculitis",
    "bệnh viêm sung huyết hang vị dạ dày": "Antral gastritis",
    "bệnh viêm tuỷ xương": "Osteomyelitis",
    "bệnh vảy nến": "Psoriasis",
    "bệnh đau nửa đầu migraine": "Migraine",
    "bệnh động mạch vành": "Coronary artery disease",
    "dẫn mật": "Choledocholithiasis",
    "dị ứng": "Allergy",
    "giãn phế quản": "Bronchiectasis",
    "giãn thừng tinh": "Varicocele",
    "giãn thừng tinh 2": "Varicocele",
    "giả gout": "Pseudogout",
    "giật": "Seizure",
    "gout": "Gout",
    "ho đái tháo đường": "Diabetes mellitus",
    "huyết khối tĩnh mạch sâu": "Deep vein thrombosis",
    "huyết áp": "Hypertension",
    "hẹp": "Cervical spinal stenosis",
    "hẹp tắc mạch vành": "Coronary artery stenosis",
    "hẹp ống sống": "Spinal stenosis",
    "hội chứng bàn chân bẹt bẩm sinh": "Congenital pes planus",
    "hội chứng nghiện rượu": "Alcohol dependence syndrome",
    "hội chứng parkinson": "Parkinsonism",
    "hội chứng ruột kích thích": "Irritable bowel syndrome",
    "hội chứng thận hư": "Nephrotic syndrome",
    "kinh ngoại biên": "Peripheral neuropathy",
    "lành về tuyến vú": "Benign breast neoplasm",
    "men gan": "Elevated liver enzymes",
    "mạch vành": "Coronary artery aneurysm",
    "mày đay vô căn": "Idiopathic urticaria",
    "nang màng nhện": "Arachnoid cyst",
    "ngoài tử cung": "Ectopic pregnancy",
    "nhiễm khuẩn": "Bacterial infection",
    "nhiễm khuẩn huyết": "Sepsis",
    "nhiễm khuẩn tiết niệu": "Urinary tract infection",
    "nhiễm khuẩn đường": "Urinary tract infection",
    "nhiễm khuẩn đường tiết": "Urinary tract infection",
    "nhiễm khuẩn đường tiết niệu": "Urinary tract infection",
    "nhiễm trùng": "Infection",
    "nhiễm trùng chi": "Limb infection",
    "nhiễm trùng chi dưới": "Lower limb infection",
    "nhiễm trùng huyết": "Sepsis",
    "nhiễm trùng nhiễm độc": "Infection",
    "nhiễm trùng đường hô hấp": "Respiratory tract infection",
    "nhiễm trùng đường tiết niệu": "Urinary tract infection",
    "nhồi": "Myocardial infarction",
    "nhồi máu không st": "Non-ST elevation myocardial infarction",
    "não úng thuỷ": "Hydrocephalus",
    "não úng thủy": "Hydrocephalus",
    "não úng tuỷ": "Hydrocephalus",
    "nấm bẹn": "Tinea cruris",
    "phình giãn động mạch vành": "Coronary artery aneurysm",
    "phù gai thị": "Papilledema",
    "quản": "Esophagitis",
    "rung": "Atrial fibrillation",
    "rung nhĩ": "Atrial fibrillation",
    "rối loạn chuyển hóa tinh bột": "Amyloidosis",
    "suy thoái võng mạc": "Retinal degeneration",
    "suy thận": "Renal failure",
    "suy thận cấp": "Acute kidney injury",
    "suy thận mạn": "Chronic kidney disease",
    "suy vành": "Ischemic heart disease",
    "sâu": "Dental caries",
    "sỏi": "Calculus",
    "sỏi niệu quản": "Ureteral calculus",
    "sỏi thận": "Nephrolithiasis",
    "sỏi ống dẫn mật": "Choledocholithiasis",
    "sỏi ống mật": "Choledocholithiasis",
    "thiếu men": "G6PD deficiency",
    "thiếu men g6pd": "G6PD deficiency",
    "thiếu máu cơ tim": "Myocardial ischemia",
    "thiếu máu cơ tim cục bộ": "Myocardial ischemia",
    "thiếu máu tan huyết": "Hemolytic anemia",
    "thiểu sản vành tai": "Microtia",
    "thrombophilia": "Thrombophilia",
    "thuyên tắc phổi": "Pulmonary embolism",
    "thuyên tắc phổi hai bên": "Bilateral pulmonary embolism",
    "thận hư": "Nephrotic syndrome",
    "thực quản": "Esophagitis",
    "tiền sản giật": "Preeclampsia",
    "tiểu đường": "Diabetes mellitus",
    "trùng răng miệng": "Oral infection",
    "tâm phế mạn": "Chronic cor pulmonale",
    "tăng cholesterol máu đơn": "Hypercholesterolemia",
    "tăng cholesterol máu đơn thuần": "Pure hypercholesterolemia",
    "tăng ha": "Hypertension",
    "tăng huyết áp": "Hypertension",
    "tăng huyết áp nguyên phát": "Essential hypertension",
    "tăng huyết áp vô căn": "Essential hypertension",
    "tăng lipid máu": "Hyperlipidemia",
    "tăng nhãn áp": "Glaucoma",
    "tăng số lươ": "",
    "tăng sản tuyến tiền liệt": "Benign prostatic hyperplasia",
    "tắc ống dẫn trứng": "Fallopian tube obstruction",
    "tụ dịch": "Fluid collection",
    "tụ máu dưới màng cứng mạn tính": "Chronic subdural hematoma",
    "tụ máu ngoài màng cứng": "Epidural hematoma",
    "tụ máu ngoài màng cứng phải": "Right epidural hematoma",
    "tụy": "Pancreatitis",
    "u nang tuyến vú": "Breast cyst",
    "u trực tràng": "Rectal tumor",
    "u tuyến": "Adenoma",
    "u xơ tuyến vú": "Breast fibroadenoma",
    "u ác trực tràng": "Rectal cancer",
    "ung thư biểu mô tuyến": "Adenocarcinoma",
    "ung thư biểu mô tuyến đại tràng": "Colon adenocarcinoma",
    "ung thư biểu mô tế bào mật": "Cholangiocarcinoma",
    "ung thư biểu mô tế bào vảy": "Squamous cell carcinoma",
    "ung thư tuyến đại tràng": "Colon cancer",
    "ung thư đường mật": "Biliary tract cancer",
    "ung thư đại tràng": "Colon cancer",
    "utis": "Urinary tract infection",
    "viêm bao tử": "Gastritis",
    "viêm bể thận": "Pyelonephritis",
    "viêm dạ dày ruột": "Gastroenteritis",
    "viêm hang vị sung huyết": "Antral gastritis",
    "viêm mô tế bào": "Cellulitis",
    "viêm nang lông": "Folliculitis",
    "viêm quanh răng": "Periodontitis",
    "viêm sung huyết hang vị dạ dày": "Antral gastritis",
    "viêm thực quản": "Esophagitis",
    "viêm tuyến mồ hôi": "Hidradenitis suppurativa",
    "viêm tủy xương": "Osteomyelitis",
    "viêm tủy xương mãn tính": "Chronic osteomyelitis",
    "viêm xương tủy": "Osteomyelitis",
    "vp": "Pneumonia",
    "vpht": "Necrotizing pneumonia",
    "xhth": "Gastrointestinal hemorrhage",
    "xuất huyết tiêu": "Gastrointestinal hemorrhage",
    "xơ gan": "Cirrhosis of liver",
    "xơ gan do rượu": "Alcoholic cirrhosis",
    "xơ vữa": "Atherosclerosis",
    "xơ vữa động mạch vành": "Coronary atherosclerosis",
    "xẹp phổi": "Atelectasis",
    "áp xe": "Abscess",
    "áp xe phổi": "Lung abscess",
    "áp-xe phổi": "Lung abscess",
    "ăng huyết áp": "Hypertension",
    "đa u tủy": "Multiple myeloma",
    "đay": "Idiopathic urticaria",
    "đay vô căn": "Idiopathic urticaria",
    "đay vô căn hộ": "Idiopathic urticaria",
    "đtd": "Type 2 diabetes mellitus",
    "đái tháo đường": "Diabetes mellitus",
    "đái tháo đường typ ii": "Type 2 diabetes mellitus",
    "đái tháo đường típ 2": "Type 2 diabetes mellitus",
    "đái tháo đườngđái tháo đường": "Diabetes mellitus",
    "đường": "Urinary tract infection",
    "ứ mật": "Cholestasis",
}


DRUG_QUERIES = {
    "acetaminophen": "acetaminophen",
    "acetaminophen 500mg": "acetaminophen 500 mg",
    "acne": "", "ad": "", "bc": "", "cam": "", "copd": "", "neut": "",
    "stress": "", "tham": "",
    "albuterol": "albuterol",
    "allopurinol": "allopurinol",
    "amoxicillin": "amoxicillin",
    "aspirin": "aspirin",
    "aspirin 325mg": "aspirin 325 mg",
    "atenolol": "atenolol",
    "augmentin": "Augmentin",
    "azithromycin": "azithromycin",
    "bactrim": "Bactrim",
    "bumetanide 2mg iv": "bumetanide 2 mg injection",
    "caffeine": "caffeine",
    "carvedilol": "carvedilol",
    "ceftriaxone 1g": "ceftriaxone 1 g injection",
    "citrate 40mg": "alverine citrate 40 mg",
    "clonidine": "clonidine",
    "compazine": "Compazine",
    "coumadin": "Coumadin",
    "coumadin 3.0 mg": "Coumadin 3 mg",
    "crestor": "Crestor",
    "dextrose": "dextrose",
    "dilaudid": "Dilaudid",
    "doxycycline": "doxycycline",
    "eliquis": "Eliquis",
    "folate": "folic acid",
    "furosemide": "furosemide",
    "gleevec": "Gleevec",
    "implanon": "Implanon",
    "insulin glargine": "insulin glargine",
    "lasix": "Lasix",
    "lasix 40 mg once": "furosemide 40 mg injection",
    "levofloxacin 750mg iv": "levofloxacin 750 mg injection",
    "lorazepam": "lorazepam",
    "medrol 16mg": "Medrol 16 mg",
    "methylprednisolone 125mg iv": "methylprednisolone 125 mg injection",
    "metoclopramide 10mg": "metoclopramide 10 mg",
    "metoprolol 25mg po bid": "metoprolol 25 mg oral tablet",
    "mucinex d": "Mucinex D",
    "nac": "acetylcysteine",
    "nitroglycerin": "nitroglycerin",
    "omeprazole": "omeprazole",
    "paracetamol": "acetaminophen",
    "seroquel": "Seroquel",
    "suboxone": "Suboxone",
    "torsemide": "torsemide",
    "tretinoin": "tretinoin topical",
    "tylenol": "Tylenol",
    "vancomycin": "vancomycin",
    "vitamin b12": "vitamin B12",
    "vitamin c": "ascorbic acid",
    "vitamin k": "vitamin K",
    "zestril 10mg": "Zestril 10 mg",
    "zosyn": "Zosyn",
}


# Source mentions sometimes use decomposed Vietnamese Unicode. Normalize the
# curation tables as well as entity text so visually identical keys still match.
NORMALIZED_INVALID_DIAGNOSES = {normalize_key(key) for key in INVALID_DIAGNOSES}
NORMALIZED_DIAGNOSIS_OVERRIDES = {
    normalize_key(key): value for key, value in DIAGNOSIS_OVERRIDES.items()
}
NORMALIZED_DRUG_QUERIES = {
    normalize_key(key): value for key, value in DRUG_QUERIES.items()
}


def _json_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.json"), key=lambda path: int(path.stem))


def build_mapping(json_folder: Path = DEFAULT_JSON_FOLDER) -> dict[str, dict[str, list[str]]]:
    mapping = {entity_type: {} for entity_type in sorted(SUPPORTED_TYPES)}
    for path in _json_files(json_folder):
        entities = json.loads(path.read_text(encoding="utf-8"))
        for entity in entities:
            entity_type = entity.get("type")
            if entity_type not in SUPPORTED_TYPES:
                continue
            key = normalize_key(entity.get("text", ""))
            if key in mapping[entity_type]:
                continue

            if entity_type == "THUỐC":
                if key not in NORMALIZED_DRUG_QUERIES:
                    raise KeyError(f"Missing manual medication query for {key!r}")
                query = NORMALIZED_DRUG_QUERIES[key]
            elif key in NORMALIZED_INVALID_DIAGNOSES:
                query = ""
            elif key in NORMALIZED_DIAGNOSIS_OVERRIDES:
                query = NORMALIZED_DIAGNOSIS_OVERRIDES[key]
            else:
                existing = entity.get("eng_translation")
                if not isinstance(existing, list) or not existing or not isinstance(existing[0], str):
                    raise KeyError(f"Missing manual diagnosis query for {key!r}")
                query = existing[0].strip()

            mapping[entity_type][key] = [query] if query else []
    return mapping


def write_mapping(mapping: dict, path: Path = DEFAULT_MAPPING_PATH) -> Path:
    path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def apply_mapping(json_folder: Path, mapping: dict) -> tuple[int, int]:
    entity_count = 0
    changed_count = 0
    for path in _json_files(json_folder):
        entities = json.loads(path.read_text(encoding="utf-8"))
        original = copy.deepcopy(entities)
        for entity in entities:
            entity_type = entity.get("type")
            if entity_type not in SUPPORTED_TYPES:
                continue
            key = normalize_key(entity.get("text", ""))
            try:
                queries = mapping[entity_type][key]
            except KeyError as exc:
                raise KeyError(f"No manual query for {path.name}: {entity_type} {key!r}") from exc
            entity_count += 1
            if entity.get("eng_translation") != queries:
                entity["eng_translation"] = list(queries)
                changed_count += 1

        for before, after in zip(original, entities):
            before.pop("eng_translation", None)
            after_without_query = dict(after)
            after_without_query.pop("eng_translation", None)
            if before != after_without_query:
                raise AssertionError(f"Non-query field changed in {path.name}")

        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(entities, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    return entity_count, changed_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-folder", type=Path, default=DEFAULT_JSON_FOLDER)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING_PATH)
    args = parser.parse_args()

    mapping = build_mapping(args.json_folder)
    write_mapping(mapping, args.mapping)
    entity_count, changed_count = apply_mapping(args.json_folder, mapping)
    print(f"Wrote {sum(len(values) for values in mapping.values())} manual mappings")
    print(f"Checked {entity_count} linkable entities; updated {changed_count}")


if __name__ == "__main__":
    main()
