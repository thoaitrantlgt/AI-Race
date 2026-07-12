from __future__ import annotations

import re


def split_sentences_with_offsets(raw_text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    for match in re.finditer(r"[.\n]+", raw_text):
        end = match.start()
        if end > start and raw_text[start:end].strip():
            left = start
            right = end
            while left < right and raw_text[left].isspace():
                left += 1
            while right > left and raw_text[right - 1].isspace():
                right -= 1
            spans.append((left, right, raw_text[left:right]))
        start = match.end()
    if start < len(raw_text) and raw_text[start:].strip():
        left = start
        right = len(raw_text)
        while left < right and raw_text[left].isspace():
            left += 1
        while right > left and raw_text[right - 1].isspace():
            right -= 1
        spans.append((left, right, raw_text[left:right]))
    return spans
