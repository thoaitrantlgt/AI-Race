from __future__ import annotations

import csv
from pathlib import Path

from src.linking.terminology_store import TermEntry, TerminologyStore
from src.preprocess.text_normalizer import normalize_for_matching


def load_icd10_vi_to_store(path: str | Path, store: TerminologyStore) -> None:
    root = Path(path)
    if not root.exists():
        return
    files = []
    for pattern in ("*.tsv", "*.csv", "*.txt"):
        files.extend(root.glob(pattern))
    for file_path in sorted(files):
        delimiter = "\t" if file_path.suffix.lower() in {".tsv", ".txt"} else ","
        with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.reader(handle, delimiter=delimiter):
                if len(row) < 2 or not row[0].strip() or row[0].lower() == "code":
                    continue
                code, vi_name = row[0].strip(), row[1].strip()
                aliases = [row[2].strip()] if len(row) > 2 and row[2].strip() else []
                store.add_entry(
                    TermEntry(
                        concept_id=f"ICD10:{code}",
                        name=vi_name,
                        normalized_name=normalize_for_matching(vi_name),
                        source="ICD10_VI",
                        semantic_type="diagnosis",
                        aliases=aliases,
                        version="ICD-10 Vietnamese",
                        language="vi",
                    )
                )
