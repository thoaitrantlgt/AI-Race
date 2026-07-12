from __future__ import annotations

from src.extraction.span_merger import merge_spans
from src.io.schema import SpanPrediction


def resolve_overlaps(spans: list[SpanPrediction], raw_text: str, config: dict) -> list[SpanPrediction]:
    min_conf = float(config.get("extraction", {}).get("min_confidence", 0.65))
    return merge_spans(spans, raw_text, min_confidence=min_conf)
