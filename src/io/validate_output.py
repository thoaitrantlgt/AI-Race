from __future__ import annotations

import json
from pathlib import Path

from src.io.read_input import InputRecord


REQUIRED_FIELDS = {"text", "position", "type", "assertions", "candidates"}
ALLOWED_TYPES = {"TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM", "CHẨN_ĐOÁN", "THUỐC"}
ALLOWED_ASSERTIONS = {"isNegated", "isFamily", "isHistorical"}
CANDIDATE_TYPES = {"CHẨN_ĐOÁN", "THUỐC"}


def validate_output(output_dir: str | Path, records: list[InputRecord], config: dict) -> None:
    root = Path(output_dir)
    by_id = {record.record_id: record for record in records}
    files = sorted(root.glob("*.json"))
    if len(files) != len(records):
        raise ValueError(f"Output JSON count {len(files)} != input count {len(records)}")
    max_candidates = int(config.get("linking", {}).get("max_candidates", 2))
    for path in files:
        record = by_id.get(path.stem)
        if record is None:
            raise ValueError(f"Unexpected output file: {path.name}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{path.name}: root must be a list")
        seen = set()
        for idx, item in enumerate(data):
            if not isinstance(item, dict) or not REQUIRED_FIELDS.issubset(item):
                raise ValueError(f"{path.name}[{idx}]: missing required fields")
            position, text = item["position"], item["text"]
            if not isinstance(position, list) or len(position) != 2:
                raise ValueError(f"{path.name}[{idx}]: position must be a two-element list")
            start, end = position
            if not isinstance(start, int) or not isinstance(end, int):
                raise ValueError(f"{path.name}[{idx}]: position values must be integers")
            if not (0 <= start < end <= len(record.raw_text)) or record.raw_text[start:end] != text:
                raise ValueError(f"{path.name}[{idx}]: invalid offset for {text!r}")
            if len(item.get("candidates", [])) > max_candidates:
                raise ValueError(f"{path.name}[{idx}]: too many candidates")
            if item["type"] not in ALLOWED_TYPES:
                raise ValueError(f"{path.name}[{idx}]: invalid type {item['type']}")
            if not isinstance(item["assertions"], list) or len(item["assertions"]) > 3:
                raise ValueError(f"{path.name}[{idx}]: invalid assertions list")
            if any(assertion not in ALLOWED_ASSERTIONS for assertion in item["assertions"]):
                raise ValueError(f"{path.name}[{idx}]: invalid assertion in {item['assertions']}")
            if item["type"] not in CANDIDATE_TYPES and item.get("candidates"):
                raise ValueError(f"{path.name}[{idx}]: candidates only allowed for diagnosis and drug")
            key = (start, end, item["type"])
            if key in seen:
                raise ValueError(f"{path.name}[{idx}]: duplicate span {key}")
            seen.add(key)
