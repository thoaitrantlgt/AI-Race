from __future__ import annotations

import argparse
import json
import zipfile


ALLOWED_ASSERTION_TYPES = {"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"}


def entity_key(entity: dict) -> tuple:
    return tuple(entity["position"]), entity["type"], entity["text"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--strict", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels", nargs="+", default=["isFamily"])
    args = parser.parse_args()

    with zipfile.ZipFile(args.base) as base_zip, zipfile.ZipFile(args.strict) as strict_zip, zipfile.ZipFile(
        args.output, "w", zipfile.ZIP_DEFLATED
    ) as output_zip:
        names = [name for name in base_zip.namelist() if name.endswith(".json")]
        for name in names:
            entities = json.loads(base_zip.read(name).decode("utf-8"))
            strict_entities = json.loads(strict_zip.read(name).decode("utf-8"))
            strict_by_key = {entity_key(entity): entity for entity in strict_entities}
            for entity in entities:
                if entity["type"] not in ALLOWED_ASSERTION_TYPES:
                    entity["assertions"] = []
                    continue
                strict_assertions = strict_by_key.get(entity_key(entity), {}).get("assertions", [])
                entity["assertions"] = [
                    value
                    for value in entity.get("assertions", [])
                    if value not in args.labels or value in strict_assertions
                ]
            output_zip.writestr(name, json.dumps(entities, ensure_ascii=False, indent=2).encode("utf-8"))
    print(f"Wrote {len(names)} JSON files to {args.output}")


if __name__ == "__main__":
    main()
