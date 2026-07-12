from __future__ import annotations

import re

from src.io.schema import SpanPrediction


DOSE_RE = re.compile(r"\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|g|ml|iu|units?|%)(?![A-Za-zÀ-ỹĐđ])", re.IGNORECASE)


def resolve_types(spans: list[SpanPrediction], raw_text: str, sections, config: dict) -> list[SpanPrediction]:
    return spans
