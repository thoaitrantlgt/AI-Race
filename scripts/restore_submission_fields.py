from __future__ import annotations

import argparse
import json
import zipfile


def entity_key(entity: dict) -> tuple:
    return tuple(entity["position"]), entity["type"], entity["text"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fields", nargs="+", choices=["assertions", "candidates"], required=True)
    args = parser.parse_args()

    changed = 0
    with zipfile.ZipFile(args.base) as base_zip, zipfile.ZipFile(args.reference) as reference_zip, zipfile.ZipFile(
        args.output, "w", zipfile.ZIP_DEFLATED
    ) as output_zip:
        names = [name for name in base_zip.namelist() if name.endswith(".json")]
        for name in names:
            entities = json.loads(base_zip.read(name).decode("utf-8"))
            reference = json.loads(reference_zip.read(name).decode("utf-8"))
            reference_by_key = {entity_key(entity): entity for entity in reference}
            for entity in entities:
                source = reference_by_key.get(entity_key(entity))
                if source is None:
                    continue
                for field in args.fields:
                    if entity.get(field, []) != source.get(field, []):
                        entity[field] = list(source.get(field, []))
                        changed += 1
            output_zip.writestr(name, json.dumps(entities, ensure_ascii=False, indent=2).encode("utf-8"))
    print(f"Wrote {len(names)} JSON files with {changed} restored field values to {args.output}")


if __name__ == "__main__":
    main()
