from __future__ import annotations

from pathlib import Path

from src.linking.terminology_store import TermEntry, TerminologyStore
from src.preprocess.text_normalizer import normalize_for_matching


def load_snomed_to_store(path: str | Path, store: TerminologyStore) -> None:
    root = Path(path)
    if not root.exists():
        return
    files = [root] if root.is_file() else sorted(root.glob("*.tsv"))
    for file_path in files:
        for line in file_path.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            semantic_type = parts[2] if len(parts) > 2 else None
            store.add_entry(TermEntry(parts[0], parts[1], normalize_for_matching(parts[1]), "SNOMED", semantic_type, []))
