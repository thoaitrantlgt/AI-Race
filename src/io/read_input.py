from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class InputRecord:
    record_id: str
    filename: str
    raw_text: str


def _sort_key(path: Path) -> tuple[int, str]:
    try:
        return (int(path.stem), path.stem)
    except ValueError:
        return (10**12, path.stem)


def read_input_dir(input_dir: str | Path) -> list[InputRecord]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory not found: {root}")
    records: list[InputRecord] = []
    for path in sorted(root.glob("*.txt"), key=_sort_key):
        records.append(
            InputRecord(
                record_id=path.stem,
                filename=path.name,
                raw_text=path.read_text(encoding="utf-8"),
            )
        )
    if not records:
        raise ValueError(f"No .txt files found in {root}")
    return records
