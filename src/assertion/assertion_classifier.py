from __future__ import annotations

import pickle
from pathlib import Path


LABEL_MAP = {
    "NEGATED": "isNegated",
    "NEGATION": "isNegated",
    "FAMILY": "isFamily",
    "HISTORICAL": "isHistorical",
    "HISTORY": "isHistorical",
    "PRESENT": "",
}


class AssertionClassifier:
    def __init__(self, model_path: str | None = None, local_files_only: bool = True, device: int = -1):
        self.enabled = False
        self.pipeline = None
        self.sklearn_model = None
        if not model_path:
            return
        if str(model_path).endswith(".pkl") and Path(model_path).exists():
            with Path(model_path).open("rb") as handle:
                self.sklearn_model = pickle.load(handle)
            self.enabled = True
            return
        if local_files_only and not Path(model_path).exists() and "/" not in model_path:
            return
        try:
            from transformers import pipeline

            self.pipeline = pipeline(
                "text-classification",
                model=model_path,
                tokenizer=model_path,
                device=device,
                local_files_only=local_files_only,
            )
            self.enabled = True
        except Exception as exc:
            print(f"Warning: could not load assertion classifier {model_path}: {exc}")

    def predict(self, span, raw_text, section_name):
        predictions = self.predict_many(span, raw_text, section_name)
        return predictions[0] if predictions else None

    def predict_many(self, span, raw_text, section_name):
        if not self.enabled or self.pipeline is None:
            if self.sklearn_model is None:
                return []
            left = max(0, span.start - 180)
            right = min(len(raw_text), span.end + 180)
            text = raw_text[left:span.start] + " [MENTION] " + raw_text[span.start:span.end] + " [/MENTION] " + raw_text[span.end:right]
            labels = []
            for label, pipeline in self.sklearn_model.get("models", {}).items():
                pred = pipeline.predict([text])[0]
                if int(pred) == 1:
                    labels.append(label)
            return labels
        left = max(0, span.start - 160)
        right = min(len(raw_text), span.end + 160)
        marked = raw_text[left:span.start] + " [MENTION] " + raw_text[span.start:span.end] + " [/MENTION] " + raw_text[span.end:right]
        try:
            result = self.pipeline(marked, truncation=True)[0]
        except Exception as exc:
            print(f"Warning: assertion classifier inference failed: {exc}")
            return []
        label = str(result.get("label", "")).upper()
        for key, mapped in LABEL_MAP.items():
            if key in label:
                return [mapped] if mapped else []
        return []
