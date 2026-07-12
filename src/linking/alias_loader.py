from __future__ import annotations

from pathlib import Path

from src.linking.terminology_store import TermEntry, TerminologyStore
from src.preprocess.text_normalizer import normalize_for_matching


BUILTIN_ALIASES = [
    ("tylenol", "acetaminophen", "Rx161", "RxNorm", "drug"),
    ("paracetamol", "acetaminophen", "Rx161", "RxNorm", "drug"),
    ("acetaminophen", "acetaminophen", "Rx161", "RxNorm", "drug"),
    ("aspirin", "aspirin", "", "RxNorm", "drug"),
    ("amlodipine", "amlodipine", "", "RxNorm", "drug"),
    ("metoprolol", "metoprolol", "", "RxNorm", "drug"),
    ("atenolol", "atenolol", "", "RxNorm", "drug"),
    ("doxycycline", "doxycycline", "", "RxNorm", "drug"),
    ("xơ gan", "xơ gan", "ICD10:K74", "ICD10_VI", "diagnosis"),
    ("tăng huyết áp", "tăng huyết áp", "ICD10:I10", "ICD10_VI", "diagnosis"),
    ("đái tháo đường", "đái tháo đường", "ICD10:E11", "ICD10_VI", "diagnosis"),
    ("tiểu đường", "đái tháo đường", "ICD10:E11", "ICD10_VI", "diagnosis"),
    ("viêm phổi", "viêm phổi", "ICD10:J18", "ICD10_VI", "diagnosis"),
    ("đột quỵ", "đột quỵ", "ICD10:I64", "ICD10_VI", "diagnosis"),
    ("trào ngược dạ dày - thực quản", "trào ngược dạ dày - thực quản", "ICD10:K21.9", "ICD10_VI", "diagnosis"),
]


def _add_alias(store: TerminologyStore, alias: str, canonical: str, concept_id: str, source: str, semantic_type: str) -> None:
    if concept_id:
        store.add_entry(
            TermEntry(
                concept_id=concept_id,
                name=canonical or alias,
                normalized_name=normalize_for_matching(canonical or alias),
                source=source or "custom",
                semantic_type=semantic_type or None,
                aliases=[alias],
            )
        )
    elif semantic_type:
        store.add_extraction_alias(alias, semantic_type)


def load_custom_aliases(path: str | Path, store: TerminologyStore) -> None:
    for row in BUILTIN_ALIASES:
        _add_alias(store, *row)
    root = Path(path)
    if not root.exists():
        return
    files = [root] if root.is_file() else sorted(root.glob("*.tsv"))
    for file_path in files:
        for line in file_path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            while len(parts) < 5:
                parts.append("")
            _add_alias(store, parts[0], parts[1], parts[2], parts[3], parts[4])
