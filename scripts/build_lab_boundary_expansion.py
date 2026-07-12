from __future__ import annotations

import argparse
import json
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-extension", type=int, default=24)
    args = parser.parse_args()

    changed: list[tuple[str, str, str]] = []
    with zipfile.ZipFile(args.base) as base_zip, zipfile.ZipFile(args.reference) as reference_zip, zipfile.ZipFile(
        args.output, "w", zipfile.ZIP_DEFLATED
    ) as output_zip:
        names = [name for name in base_zip.namelist() if name.endswith(".json")]
        for name in names:
            entities = json.loads(base_zip.read(name).decode("utf-8"))
            reference = json.loads(reference_zip.read(name).decode("utf-8"))
            reference_labs = [entity for entity in reference if entity["type"] == "TÊN_XÉT_NGHIỆM"]
            for entity in entities:
                if entity["type"] != "TÊN_XÉT_NGHIỆM":
                    continue
                start, end = entity["position"]
                candidates = [
                    candidate
                    for candidate in reference_labs
                    if candidate["position"][0] == start
                    and end < candidate["position"][1] <= end + args.max_extension
                    and candidate["text"].lower().startswith(entity["text"].lower())
                    and not any(mark in candidate["text"] for mark in "\n;:")
                ]
                if not candidates:
                    continue
                winner = min(candidates, key=lambda candidate: candidate["position"][1])
                changed.append((name, entity["text"], winner["text"]))
                entity["text"] = winner["text"]
                entity["position"] = winner["position"]
            deduplicated: list[dict] = []
            seen: set[tuple] = set()
            for entity in entities:
                key = tuple(entity["position"]), entity["type"]
                if key in seen:
                    continue
                seen.add(key)
                deduplicated.append(entity)
            output_zip.writestr(name, json.dumps(deduplicated, ensure_ascii=False, indent=2).encode("utf-8"))
    print(f"Wrote {len(names)} JSON files with {len(changed)} expansions to {args.output}")
    for item in changed:
        print(item)


if __name__ == "__main__":
    main()
