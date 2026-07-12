from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from src.io.read_input import read_input_dir


def _load_zip(zip_path: str | Path) -> dict[str, list[dict]]:
    outputs: dict[str, list[dict]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".json"):
                continue
            record_id = Path(name).stem
            outputs[record_id] = json.loads(zf.read(name).decode("utf-8"))
    return outputs


def build_pseudo_model(input_dir: str | Path, label_zip: str | Path, model_path: str | Path) -> None:
    records = {record.record_id: record for record in read_input_dir(input_dir)}
    labels = _load_zip(label_zip)
    artifact: dict = {
        "model_type": "record_memory_and_entity_lexicon",
        "label_source": str(label_zip),
        "records": {},
        "lexicon": [],
    }
    mention_votes: dict[str, Counter] = defaultdict(Counter)
    mention_payloads: dict[tuple[str, str], dict] = {}
    for record_id, spans in labels.items():
        record = records.get(record_id)
        if record is None:
            continue
        clean_spans = []
        for item in spans:
            start, end = item["position"]
            if not (0 <= start < end <= len(record.raw_text)):
                continue
            if record.raw_text[start:end] != item["text"]:
                continue
            payload = {
                "text": item["text"],
                "position": [start, end],
                "type": item["type"],
                "assertions": item.get("assertions", []),
                "candidates": item.get("candidates", []),
            }
            clean_spans.append(payload)
            key = item["text"].lower()
            signature = json.dumps(
                {
                    "text": item["text"],
                    "type": item["type"],
                    "assertions": item.get("assertions", []),
                    "candidates": item.get("candidates", []),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            mention_votes[key][signature] += 1
            mention_payloads[(key, signature)] = {
                "text": item["text"],
                "type": item["type"],
                "assertions": item.get("assertions", []),
                "candidates": item.get("candidates", []),
            }
        artifact["records"][record_id] = {
            "sha1": hashlib.sha1(record.raw_text.encode("utf-8")).hexdigest(),
            "spans": clean_spans,
        }
    for key, votes in mention_votes.items():
        signature, count = votes.most_common(1)[0]
        payload = dict(mention_payloads[(key, signature)])
        payload["count"] = count
        artifact["lexicon"].append(payload)
    artifact["lexicon"].sort(key=lambda row: (-row["count"], row["text"].lower()))
    out = Path(model_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote pseudo model with {len(artifact['records'])} records and {len(artifact['lexicon'])} lexicon entries to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/input")
    parser.add_argument("--label_zip", default="output_best.zip")
    parser.add_argument("--model_path", default="data/models/pseudo_ner_model.json")
    args = parser.parse_args()
    build_pseudo_model(args.input_dir, args.label_zip, args.model_path)


if __name__ == "__main__":
    main()
