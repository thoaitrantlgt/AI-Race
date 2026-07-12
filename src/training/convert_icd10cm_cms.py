from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def convert_cms_icd10cm(zip_path: str | Path, output_path: str | Path) -> None:
    rows = []
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        source_name = "icd10cm_codes_2026.txt" if "icd10cm_codes_2026.txt" in names else next(
            name for name in names if name.endswith("_codes_2026.txt")
        )
        text = zf.read(source_name).decode("utf-8", errors="ignore")
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        code, description = parts[0].strip(), parts[1].strip()
        if not code or not description:
            continue
        rows.append((description, description, f"ICD10:{code}", "ICD10_CM_2026", "diagnosis"))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# source=CMS 2026 ICD-10-CM code descriptions\n")
        handle.write("# url=https://www.cms.gov/files/zip/2026-code-descriptions-tabular-order.zip\n")
        for row in rows:
            handle.write("\t".join(row) + "\n")
    print(f"Wrote {len(rows)} ICD-10-CM aliases to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default="data/terminology/icd10_vi/2026-code-descriptions-tabular-order.zip")
    parser.add_argument("--output", default="data/terminology/custom_aliases/icd10cm_2026_cms.tsv")
    args = parser.parse_args()
    convert_cms_icd10cm(args.zip, args.output)


if __name__ == "__main__":
    main()
