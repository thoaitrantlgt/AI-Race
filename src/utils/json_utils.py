from __future__ import annotations

import json
from pathlib import Path


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data, ensure_ascii: bool = False, indent: int = 2) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=ensure_ascii, indent=indent), encoding="utf-8")
