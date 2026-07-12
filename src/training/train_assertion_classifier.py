from __future__ import annotations

import argparse
import json
import pickle
import zipfile
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.io.read_input import read_input_dir


ASSERTION_LABELS = ["isNegated", "isFamily", "isHistorical"]


def _load_zip(path: str | Path) -> dict[str, list[dict]]:
    outputs = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.endswith(".json"):
                outputs[Path(name).stem] = json.loads(zf.read(name).decode("utf-8"))
    return outputs


def _context(raw_text: str, start: int, end: int, window: int = 180) -> str:
    left = max(0, start - window)
    right = min(len(raw_text), end + window)
    return raw_text[left:start] + " [MENTION] " + raw_text[start:end] + " [/MENTION] " + raw_text[end:right]


def train_assertion_classifier(input_dir: str | Path, label_zip: str | Path, output_path: str | Path) -> None:
    records = {record.record_id: record for record in read_input_dir(input_dir)}
    labels = _load_zip(label_zip)
    texts: list[str] = []
    y_by_label = {label: [] for label in ASSERTION_LABELS}
    for record_id, spans in labels.items():
        record = records.get(record_id)
        if record is None:
            continue
        for item in spans:
            if item.get("type") not in {"CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"}:
                continue
            start, end = item["position"]
            if not (0 <= start < end <= len(record.raw_text)):
                continue
            texts.append(_context(record.raw_text, start, end))
            assertions = set(item.get("assertions", []) or [])
            for label in ASSERTION_LABELS:
                y_by_label[label].append(1 if label in assertions else 0)
    models = {}
    for label, y in y_by_label.items():
        if len(set(y)) < 2:
            continue
        models[label] = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 3), min_df=1, max_features=50000)),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
        models[label].fit(texts, y)
    artifact = {"labels": ASSERTION_LABELS, "models": models, "num_examples": len(texts)}
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as handle:
        pickle.dump(artifact, handle)
    print(f"Wrote assertion classifier with {len(models)} binary models and {len(texts)} examples to {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="data/input")
    parser.add_argument("--label_zip", default="output_best.zip")
    parser.add_argument("--output", default="data/models/assertion_classifier.pkl")
    args = parser.parse_args()
    train_assertion_classifier(args.input_dir, args.label_zip, args.output)


if __name__ == "__main__":
    main()
