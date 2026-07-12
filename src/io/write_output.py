from __future__ import annotations

import json
import zipfile
from pathlib import Path

from src.io.schema import SpanPrediction, prediction_to_submission_json


def write_record_json(record_id: str, predictions: list[SpanPrediction], output_dir: str | Path, ensure_ascii: bool = False, indent: int = 2) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    ordered = sorted(predictions, key=lambda pred: (pred.start, pred.end, pred.concept_type))
    data = [prediction_to_submission_json(pred) for pred in ordered]
    (root / f"{record_id}.json").write_text(json.dumps(data, ensure_ascii=ensure_ascii, indent=indent), encoding="utf-8")


def write_all_json(all_outputs: dict[str, list[SpanPrediction]], output_dir: str | Path, ensure_ascii: bool = False, indent: int = 2) -> None:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for old_file in root.glob("*.json"):
        old_file.unlink()
    for record_id, predictions in all_outputs.items():
        write_record_json(record_id, predictions, root, ensure_ascii=ensure_ascii, indent=indent)


def create_output_zip(output_dir: str | Path, zip_path: str | Path) -> None:
    root = Path(output_dir)
    zip_file = Path(zip_path)
    if zip_file.exists():
        zip_file.unlink()
    with zipfile.ZipFile(zip_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.glob("*.json"), key=lambda p: (int(p.stem) if p.stem.isdigit() else 10**12, p.name)):
            zf.write(path, arcname=f"{root.name}/{path.name}")
