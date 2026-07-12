from __future__ import annotations

import string


TRIM_CHARS = " \t\r\n" + string.punctuation.replace("-", "")


def validate_span(raw_text: str, start: int, end: int, text: str) -> bool:
    return 0 <= start < end <= len(raw_text) and raw_text[start:end] == text


def trim_span_to_valid_boundary(raw_text: str, start: int, end: int) -> tuple[int, int, str]:
    start = max(0, start)
    end = min(len(raw_text), end)
    while start < end and raw_text[start] in TRIM_CHARS:
        start += 1
    while end > start and raw_text[end - 1] in TRIM_CHARS:
        end -= 1
    return start, end, raw_text[start:end]
