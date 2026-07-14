from __future__ import annotations

import argparse
import json
import zipfile


def entity_key(entity: dict) -> tuple:
    return tuple(entity["position"]), entity["type"], entity["text"]


def containment_match(entity: dict, references: list[dict]) -> dict | None:
    start, end = entity["position"]
    candidates = []
    for reference in references:
        if reference["type"] != entity["type"]:
            continue
        ref_start, ref_end = reference["position"]
        if (start <= ref_start and ref_end <= end) or (ref_start <= start and end <= ref_end):
            overlap = min(end, ref_end) - max(start, ref_start)
            if overlap > 0:
                candidates.append((overlap, -abs((end - start) - (ref_end - ref_start)), reference))
    return max(candidates, default=(0, 0, None), key=lambda item: (item[0], item[1]))[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preserve-missing-types", nargs="*", default=["THUỐC"])
    args = parser.parse_args()

    exact_count = 0
    overlap_count = 0
    cleared_count = 0
    with zipfile.ZipFile(args.base) as base_zip, zipfile.ZipFile(args.reference) as reference_zip, zipfile.ZipFile(
        args.output, "w", zipfile.ZIP_DEFLATED
    ) as output_zip:
        names = [name for name in base_zip.namelist() if name.endswith(".json")]
        for name in names:
            entities = json.loads(base_zip.read(name).decode("utf-8"))
            references = json.loads(reference_zip.read(name).decode("utf-8"))
            reference_by_key = {entity_key(entity): entity for entity in references}
            for entity in entities:
                if entity["type"] not in {"CHẨN_ĐOÁN", "THUỐC"}:
                    entity["candidates"] = []
                    continue
                reference = reference_by_key.get(entity_key(entity))
                if reference is not None:
                    entity["candidates"] = list(reference.get("candidates", []))
                    exact_count += 1
                    continue
                reference = containment_match(entity, references)
                if reference is not None:
                    entity["candidates"] = list(reference.get("candidates", []))
                    overlap_count += 1
                    continue
                if entity["type"] not in args.preserve_missing_types:
                    if entity.get("candidates"):
                        cleared_count += 1
                    entity["candidates"] = []
            output_zip.writestr(name, json.dumps(entities, ensure_ascii=False, indent=2).encode("utf-8"))

    print(
        f"Projected candidates for {exact_count} exact and {overlap_count} contained entities; "
        f"cleared {cleared_count} unsupported candidate lists"
    )


if __name__ == "__main__":
    main()
