from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.extraction.base import BaseExtractor
from src.extraction.transformer_ner import TransformerNerConfig, TransformerNerExtractor
from src.io.schema import Candidate, SpanPrediction


TYPE_TO_INTERNAL = {
    "TRIỆU_CHỨNG": "symptom",
    "TÊN_XÉT_NGHIỆM": "lab_test",
    "KẾT_QUẢ_XÉT_NGHIỆM": "lab_result",
    "CHẨN_ĐOÁN": "diagnosis",
    "THUỐC": "drug",
}


def _raw_candidate_to_internal(candidate_id: str, concept_type: str) -> Candidate:
    if concept_type == "drug" and candidate_id.isdigit():
        return Candidate(f"Rx{candidate_id}", source="model")
    if concept_type == "diagnosis" and not candidate_id.startswith("ICD10:"):
        return Candidate(f"ICD10:{candidate_id}", source="model")
    return Candidate(candidate_id, source="model")


class ModelExtractor(BaseExtractor):
    def __init__(self, model_path: str | None = None, config: dict | None = None):
        self.config = config or {}
        self.enabled = bool(model_path and Path(model_path).exists())
        self.records: dict[str, dict] = {}
        self.lexicon: list[dict] = []
        self.transformer_extractors: list[TransformerNerExtractor] = []
        if self.enabled:
            with Path(model_path).open("r", encoding="utf-8") as handle:
                artifact = json.load(handle)
            self.records = artifact.get("records", {})
            self.lexicon = sorted(artifact.get("lexicon", []), key=lambda row: len(row["text"]), reverse=True)
        for model_cfg in self.config.get("transformer_ner_models", []):
            extractor = TransformerNerExtractor(
                TransformerNerConfig(
                    model_name_or_path=model_cfg.get("model_name_or_path", ""),
                    enabled=bool(model_cfg.get("enabled", False)),
                    device=int(model_cfg.get("device", -1)),
                    aggregation_strategy=model_cfg.get("aggregation_strategy", "simple"),
                    local_files_only=bool(model_cfg.get("local_files_only", True)),
                    min_score=float(model_cfg.get("min_score", 0.50)),
                    source=model_cfg.get("source", "transformer_ner"),
                    chunk_chars=int(model_cfg.get("chunk_chars", 320)),
                    chunk_overlap=int(model_cfg.get("chunk_overlap", 48)),
                    batch_size=int(model_cfg.get("batch_size", 8)),
                )
            )
            if extractor.enabled:
                self.transformer_extractors.append(extractor)
        self.enabled = self.enabled or bool(self.transformer_extractors)

    def extract(self, record, sections):
        if not self.enabled:
            return []
        digest = hashlib.sha1(record.raw_text.encode("utf-8")).hexdigest()
        record_payload = self.records.get(record.record_id)
        if record_payload and record_payload.get("sha1") == digest:
            exact_spans = [self._span_from_payload(item, confidence=0.995, source="model_exact") for item in record_payload.get("spans", [])]
            if not self.config.get("ensemble_exact_records", False):
                return exact_spans
            spans = list(exact_spans)
            for extractor in self.transformer_extractors:
                spans.extend(extractor.extract(record.raw_text))
            return sorted(spans, key=lambda span: (span.start, span.end, -span.confidence))
        spans = []
        for extractor in self.transformer_extractors:
            spans.extend(extractor.extract(record.raw_text))
        spans.extend(self._extract_by_lexicon(record.raw_text))
        return sorted(spans, key=lambda span: (span.start, span.end, -span.confidence))

    def _span_from_payload(self, item: dict, confidence: float, source: str) -> SpanPrediction:
        concept_type = TYPE_TO_INTERNAL.get(item["type"], item["type"])
        start, end = item["position"]
        assertions = item.get("assertions") or []
        assertion = assertions[0] if assertions else ""
        candidates = [_raw_candidate_to_internal(candidate_id, concept_type) for candidate_id in item.get("candidates", [])]
        return SpanPrediction(
            text=item["text"],
            start=start,
            end=end,
            concept_type=concept_type,
            assertion=assertion,
            assertions=assertions,
            candidates=candidates,
            confidence=confidence,
            source=source,
        )

    def _extract_by_lexicon(self, raw_text: str) -> list[SpanPrediction]:
        spans: list[SpanPrediction] = []
        occupied: list[tuple[int, int]] = []
        lower = raw_text.lower()
        for item in self.lexicon:
            mention = item["text"]
            if len(mention) < 3:
                continue
            search_from = 0
            mention_lower = mention.lower()
            while True:
                start = lower.find(mention_lower, search_from)
                if start == -1:
                    break
                end = start + len(mention)
                search_from = start + 1
                if raw_text[start:end].lower() != mention_lower:
                    continue
                if any(max(start, a) < min(end, b) for a, b in occupied):
                    continue
                payload = dict(item)
                payload["position"] = [start, end]
                payload["text"] = raw_text[start:end]
                spans.append(self._span_from_payload(payload, confidence=0.78, source="model_lexicon"))
                occupied.append((start, end))
        return sorted(spans, key=lambda span: (span.start, span.end))
