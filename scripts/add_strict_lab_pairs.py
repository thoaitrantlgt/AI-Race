from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.extraction.lab_extractor import LAB_NAMES


UNIT_RE = (
    r"(?:%|mg/dL|mmol/L|µmol/L|umol/L|g/L|g/dL|U/L|IU/L|ng/L|ng/mL|pg/mL|"
    r"mEq/L|mmHg|10\^9/L|10\^12/L|K/uL|M/uL)"
)
VALUE_RE = rf"[<>]?[ \t]*\d+(?:[,.]\d+)?(?:[ \t]*{UNIT_RE})?"
STRICT_NUMERIC_LAB_ALIASES = {
    "ag",
    "anion gap",
    "ap",
    "bạch cầu",
    "bicarbonate",
    "bilirubin trực tiếp",
    "bilirubin toàn phần",
    "canxi",
    "canxi ion hóa",
    "canxi toàn phần",
    "chloride",
    "cr",
    "creatinin",
    "egfr",
    "hco3",
    "hemoglobin",
    "huyết cầu tố",
    "huyết sắc tố",
    "kali",
    "lactate",
    "lipase",
    "lymphocyte",
    "magie",
    "magnesium",
    "natri",
    "neutrophil",
    "phosphate",
    "phosphatase kiềm",
    "platelets",
    "potassium",
    "sodium",
    "tbili",
    "tiểu cầu",
}
NAME_BODY = "|".join(
    re.escape(name) for name in sorted(set(LAB_NAMES) | STRICT_NUMERIC_LAB_ALIASES, key=len, reverse=True)
)
STRICT_PAIR_RE = re.compile(
    rf"(?<![A-Za-z0-9])(?P<name>(?:{NAME_BODY})(?:[ \t]*\([^():;\n]{{1,90}}\))?)"
    rf"[ \t]*(?:(?:là|bằng|đạt|:|：|=)[ \t]*)?(?P<value>{VALUE_RE})(?![.,]\d|[A-Za-z0-9])",
    re.IGNORECASE,
)


def is_covered(entities: list[dict], typ: str, start: int, end: int) -> bool:
    return any(
        entity["type"] == typ
        and entity["position"][0] <= start
        and end <= entity["position"][1]
        for entity in entities
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--input-dir", default="data/input")
    args = parser.parse_args()

    added_names: list[tuple[str, str, list[int]]] = []
    added_results: list[tuple[str, str, list[int]]] = []
    with zipfile.ZipFile(args.input) as source_zip, zipfile.ZipFile(
        args.output, "w", zipfile.ZIP_DEFLATED
    ) as output_zip:
        names = [name for name in source_zip.namelist() if name.endswith(".json")]
        for name in names:
            entities = json.loads(source_zip.read(name).decode("utf-8"))
            raw_path = Path(args.input_dir) / f"{Path(name).stem}.txt"
            raw = raw_path.read_text(encoding="utf-8")
            additions: list[dict] = []
            for match in STRICT_PAIR_RE.finditer(raw):
                name_start, name_end = match.span("name")
                value_start, value_end = match.span("value")
                if not is_covered(entities + additions, "TÊN_XÉT_NGHIỆM", name_start, name_end):
                    addition = {
                        "text": raw[name_start:name_end],
                        "position": [name_start, name_end],
                        "type": "TÊN_XÉT_NGHIỆM",
                        "assertions": [],
                        "candidates": [],
                    }
                    additions.append(addition)
                    added_names.append((name, addition["text"], addition["position"]))
                if not is_covered(entities + additions, "KẾT_QUẢ_XÉT_NGHIỆM", value_start, value_end):
                    addition = {
                        "text": raw[value_start:value_end],
                        "position": [value_start, value_end],
                        "type": "KẾT_QUẢ_XÉT_NGHIỆM",
                        "assertions": [],
                        "candidates": [],
                    }
                    additions.append(addition)
                    added_results.append((name, addition["text"], addition["position"]))

            entities.extend(additions)
            entities.sort(key=lambda entity: (entity["position"][0], entity["position"][1], entity["type"]))
            output_zip.writestr(name, json.dumps(entities, ensure_ascii=False, indent=2).encode("utf-8"))

    print(f"Added {len(added_names)} lab names and {len(added_results)} lab results")
    for item in added_names:
        print("name", item)
    for item in added_results:
        print("result", item)


if __name__ == "__main__":
    main()
