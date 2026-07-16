import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

JSON_FOLDER = PROJECT_ROOT / "data" / "output_merged"
FINAL_JSON_FOLDER = PROJECT_ROOT / "data" / "output_merged_validation"


def final_formating(json_folder, final_json_folder):
    json_folder = Path(json_folder)
    final_json_folder = Path(final_json_folder)
    final_json_folder.mkdir(parents=True, exist_ok=True)

    valid_assertions = {
        "isNegated",
        "isFamily",
        "isHistorical",
    }

    assertion_types = {
        "CHẨN_ĐOÁN",
        "THUỐC",
        "TRIỆU_CHỨNG",
    }

    candidate_types = {
        "CHẨN_ĐOÁN",
        "THUỐC",
    }

    for json_path in json_folder.glob("*.json"):
        with json_path.open("r", encoding="utf-8") as file:
            entities = json.load(file)

        formatted_entities = []

        for entity in entities:
            entity_type = entity["type"]
            tag = entity.get("assertion_result", {}).get("tag")

            assertions = (
                [tag]
                if entity_type in assertion_types
                and tag in valid_assertions
                else []
            )

            formatted_entity = {
                "text": entity["text"],
                "type": entity_type,
                "assertions": assertions,
                "position": [int(value) for value in entity["pos"]],
            }

            if entity_type in candidate_types:
                candidate = entity.get("candidate")

                if (
                    isinstance(candidate, dict)
                    and candidate.get("code") is not None
                ):
                    formatted_entity["candidates"] = [
                        str(candidate["code"])
                    ]
                else:
                    formatted_entity["candidates"] = []

            formatted_entities.append(formatted_entity)

        output_path = final_json_folder / json_path.name

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                formatted_entities,
                file,
                ensure_ascii=False,
                indent=4,
            )


if __name__ == "__main__":
    final_formating(JSON_FOLDER, FINAL_JSON_FOLDER)
    print(f"Submission files created at: {FINAL_JSON_FOLDER}")