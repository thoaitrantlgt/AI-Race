from __future__ import annotations

from src.io.schema import SpanPrediction
from src.preprocess.offset_mapper import validate_span


def _repair(raw_text: str, span: SpanPrediction) -> SpanPrediction | None:
    if validate_span(raw_text, span.start, span.end, span.text):
        return span
    left = max(0, span.start - 20)
    right = min(len(raw_text), span.end + 20)
    window = raw_text[left:right]
    positions = []
    pos = window.find(span.text)
    while pos != -1:
        positions.append(left + pos)
        pos = window.find(span.text, pos + 1)
    if len(positions) == 1:
        span.start = positions[0]
        span.end = span.start + len(span.text)
        return span
    return None


def validate_and_repair_offsets(spans: list[SpanPrediction], raw_text: str, config: dict | None = None) -> list[SpanPrediction]:
    repaired = []
    for span in spans:
        fixed = _repair(raw_text, span)
        if fixed is not None and validate_span(raw_text, fixed.start, fixed.end, fixed.text):
            repaired.append(fixed)
    return repaired
