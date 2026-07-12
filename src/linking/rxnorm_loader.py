from __future__ import annotations

from pathlib import Path

from src.linking.terminology_store import TermEntry, TerminologyStore
from src.preprocess.text_normalizer import normalize_for_matching


TTY_KEEP = {"IN", "PIN", "BN", "SCD", "SBD", "GPCK", "BPCK"}


def load_rxnorm_to_store(path: str | Path, store: TerminologyStore) -> None:
    root = Path(path)
    conso = root / "RXNCONSO.RRF"
    if not conso.exists():
        return
    if "2026" not in str(root):
        print(f"Warning: configured RxNorm folder does not indicate 2026: {root}")
    with conso.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("|")
            if len(parts) < 15:
                continue
            rxcui, lat, tty, name, suppress = parts[0], parts[1], parts[12], parts[14], parts[16] if len(parts) > 16 else ""
            if lat != "ENG" or tty not in TTY_KEEP or suppress == "Y":
                continue
            store.add_entry(
                TermEntry(
                    concept_id=f"Rx{rxcui}",
                    name=name,
                    normalized_name=normalize_for_matching(name),
                    source="RxNorm",
                    semantic_type="drug",
                    aliases=[],
                    version="2026",
                    language="en",
                )
            )
