from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.io.schema import SpanPrediction


LABEL_TO_TYPE = {
    "SYMPTOM": "symptom",
    "DISEASESYMTOM": "symptom",
    "SIGN": "symptom",
    "DISEASE": "diagnosis",
    "DIAGNOSIS": "diagnosis",
    "PROBLEM": "diagnosis",
    "DRUG": "drug",
    "DRUGCHEMICAL": "drug",
    "MEDICATION": "drug",
    "MEDICINE": "drug",
    "TEST": "lab_test",
    "LAB_TEST": "lab_test",
    "DIAGNOSTICS": "lab_test",
    "MEDDEVICETECHNIQUE": "lab_test",
    "RESULT": "lab_result",
    "LAB_RESULT": "lab_result",
    "UNITCALIBRATOR": "lab_result",
}


@dataclass
class TransformerNerConfig:
    model_name_or_path: str
    enabled: bool = True
    device: int = -1
    aggregation_strategy: str = "simple"
    local_files_only: bool = True
    min_score: float = 0.50
    source: str = "transformer_ner"
    chunk_chars: int = 320
    chunk_overlap: int = 48
    batch_size: int = 8


def _label_to_type(label: str) -> str | None:
    cleaned = label.upper().replace("B-", "").replace("I-", "").replace("S-", "").replace("E-", "")
    for key, value in LABEL_TO_TYPE.items():
        if key in cleaned:
            return value
    return None


class TransformerNerExtractor:
    def __init__(self, config: TransformerNerConfig):
        self.config = config
        self.enabled = False
        self.pipeline = None
        if not config.enabled:
            return
        model_path = config.model_name_or_path
        looks_local = Path(model_path).is_absolute() or model_path.startswith((".", "data/", "data\\"))
        if config.local_files_only and looks_local and not Path(model_path).exists():
            print(f"Warning: local transformer NER checkpoint not found, skipping: {model_path}")
            return
        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

            tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, local_files_only=config.local_files_only)
            model = AutoModelForTokenClassification.from_pretrained(config.model_name_or_path, local_files_only=config.local_files_only)

            self.pipeline = pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy=config.aggregation_strategy,
                device=config.device,
            )
            self.enabled = True
        except Exception as exc:
            print(f"Warning: could not load transformer NER model {config.model_name_or_path}: {exc}")

    def _iter_chunks(self, raw_text: str):
        chunk_chars = max(80, self.config.chunk_chars)
        overlap = min(max(0, self.config.chunk_overlap), chunk_chars // 3)
        start = 0
        while start < len(raw_text):
            target_end = min(len(raw_text), start + chunk_chars)
            end = target_end
            if target_end < len(raw_text):
                search_from = start + chunk_chars // 2
                boundaries = [raw_text.rfind(mark, search_from, target_end) for mark in ("\n", ". ", "; ", ", ", " ")]
                boundary = max(boundaries)
                if boundary > start:
                    end = boundary + (0 if raw_text[boundary] == "\n" else 1)
            if end <= start:
                end = target_end
            yield start, raw_text[start:end]
            if end >= len(raw_text):
                break
            next_start = max(start + 1, end - overlap)
            while next_start > start and next_start < len(raw_text) and not raw_text[next_start - 1].isspace():
                next_start -= 1
            start = next_start if next_start > start else end

    def extract(self, raw_text: str) -> list[SpanPrediction]:
        if not self.enabled or self.pipeline is None:
            return []
        spans: list[SpanPrediction] = []
        chunks = list(self._iter_chunks(raw_text))
        try:
            batch_outputs = self.pipeline(
                [chunk_text for _, chunk_text in chunks],
                batch_size=max(1, self.config.batch_size),
            )
        except Exception as exc:
            print(f"Warning: transformer NER batch inference failed: {exc}")
            batch_outputs = [[] for _ in chunks]
        for (chunk_start, _), outputs in zip(chunks, batch_outputs):
            for item in outputs:
                local_start = item.get("start")
                local_end = item.get("end")
                if local_start is None or local_end is None:
                    continue
                start = chunk_start + int(local_start)
                end = chunk_start + int(local_end)
                score = float(item.get("score", 0.0))
                if start < 0 or end <= start or end > len(raw_text) or score < self.config.min_score:
                    continue
                label = str(item.get("entity_group") or item.get("entity") or "")
                concept_type = _label_to_type(label)
                if concept_type is None:
                    continue
                text = raw_text[start:end]
                spans.append(
                    SpanPrediction(
                        text=text,
                        start=start,
                        end=end,
                        concept_type=concept_type,
                        confidence=score,
                        source=self.config.source,
                    )
                )
        dedup = {(span.start, span.end, span.concept_type): span for span in spans}
        ordered = sorted(dedup.values(), key=lambda span: (span.start, span.end))
        merged: list[SpanPrediction] = []
        for span in ordered:
            if merged:
                previous = merged[-1]
                gap = span.start - previous.end
                bridge = raw_text[previous.end:span.start]
                if (
                    span.concept_type == previous.concept_type
                    and 0 <= gap <= 1
                    and (not bridge or bridge.isspace())
                    and span.end - previous.start <= 100
                ):
                    previous.end = span.end
                    previous.text = raw_text[previous.start:previous.end]
                    previous.confidence = min(previous.confidence, span.confidence)
                    continue
            merged.append(span)
        return merged
