from __future__ import annotations

from src.io.schema import SpanPrediction
from src.preprocess.offset_mapper import validate_span


def overlap_ratio(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union else 0.0


def _is_transformer(span: SpanPrediction) -> bool:
    return span.source.endswith("_ner") or span.source == "transformer_ner"


def _boundary_close(a: SpanPrediction, b: SpanPrediction, tolerance: int = 2) -> bool:
    return abs(a.start - b.start) <= tolerance and abs(a.end - b.end) <= tolerance


def _apply_boundary_consensus(spans: list[SpanPrediction], raw_text: str) -> list[SpanPrediction]:
    transformer = [span for span in spans if _is_transformer(span)]
    if not transformer:
        return spans
    rules = [span for span in spans if not _is_transformer(span) and span.source != "model_exact"]
    exact = [span for span in spans if span.source == "model_exact"]
    accepted: list[SpanPrediction] = [span for span in spans if not _is_transformer(span)]
    generic = {"thuốc", "triệu chứng", "bệnh", "xét nghiệm", "kết quả"}
    for old in exact:
        if old.concept_type not in {"symptom", "lab_test"}:
            continue
        candidates = [
            model_span
            for model_span in transformer
            if model_span.concept_type == old.concept_type
            and model_span.confidence >= 0.90
            and model_span.text.strip().lower() not in generic
            and len(model_span.text.strip()) >= 3
            and (
                (
                    old.start <= model_span.start
                    and model_span.end <= old.end
                    and (old.end - old.start) - (model_span.end - model_span.start) >= 8
                )
                or (
                    old.concept_type == "lab_test"
                    and model_span.start <= old.start
                    and old.end <= model_span.end
                    and (model_span.end - model_span.start) - (old.end - old.start) >= 3
                )
            )
            and any(
                rule.concept_type == model_span.concept_type and _boundary_close(rule, model_span)
                for rule in rules
            )
        ]
        if not candidates:
            continue
        winner = max(candidates, key=lambda span: (span.end - span.start, span.confidence))
        old.start = winner.start
        old.end = winner.end
        old.text = raw_text[winner.start:winner.end]
    return accepted


def _better(a: SpanPrediction, b: SpanPrediction) -> SpanPrediction:
    if a.concept_type == b.concept_type and abs(a.confidence - b.confidence) <= 0.08:
        return a if (a.end - a.start) >= (b.end - b.start) else b
    if a.concept_type == "drug" and any(ch.isdigit() for ch in a.text):
        return a
    if b.concept_type == "drug" and any(ch.isdigit() for ch in b.text):
        return b
    return a if a.confidence >= b.confidence else b


def merge_spans(spans: list[SpanPrediction], raw_text: str, min_confidence: float = 0.65) -> list[SpanPrediction]:
    spans = _apply_boundary_consensus(spans, raw_text)
    valid = [s for s in spans if s.confidence >= min_confidence and validate_span(raw_text, s.start, s.end, s.text)]
    dedup: dict[tuple[int, int, str], SpanPrediction] = {}
    for span in valid:
        key = (span.start, span.end, span.concept_type)
        if key not in dedup or span.confidence > dedup[key].confidence:
            dedup[key] = span
    selected: list[SpanPrediction] = []
    for span in sorted(dedup.values(), key=lambda s: (s.start, s.end, -s.confidence)):
        replaced = False
        keep = True
        for idx, old in enumerate(selected):
            if overlap_ratio(span.start, span.end, old.start, old.end) > 0.5:
                winner = _better(span, old)
                if winner is span:
                    selected[idx] = span
                    replaced = True
                keep = False
                break
        if keep and not replaced:
            selected.append(span)
    return sorted(selected, key=lambda s: (s.start, s.end))
