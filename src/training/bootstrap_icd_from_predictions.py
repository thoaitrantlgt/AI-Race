from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def bootstrap_icd(label_zip: str | Path, output_path: str | Path) -> None:
    rows: dict[tuple[str, str], tuple[str, str, str, str, str]] = {}
    with zipfile.ZipFile(label_zip) as zf:
        for name in zf.namelist():
            if not name.endswith(".json"):
                continue
            data = json.loads(zf.read(name).decode("utf-8"))
            for item in data:
                if item.get("type") != "CHẨN_ĐOÁN":
                    continue
                text = str(item.get("text", "")).strip()
                for code in item.get("candidates", []) or []:
                    code = str(code).strip()
                    if text and code:
                        rows[(text.lower(), code)] = (text, text, f"ICD10:{code}", "ICD10_VI", "diagnosis")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# source=output_best bootstrap\n")
        for row in sorted(rows.values(), key=lambda item: (item[2], item[0].lower())):
            handle.write("\t".join(row) + "\n")
    print(f"Wrote {len(rows)} ICD aliases to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_zip", default="output_best.zip")
    parser.add_argument("--output", default="data/terminology/custom_aliases/icd10_bootstrap_from_best.tsv")
    args = parser.parse_args()
    bootstrap_icd(args.label_zip, args.output)


if __name__ == "__main__":
    main()
