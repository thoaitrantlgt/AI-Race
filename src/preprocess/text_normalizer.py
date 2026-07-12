from __future__ import annotations

import re
import unicodedata


def normalize_for_matching(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[“”\"'`]", " ", text)
    text = re.sub(r"[,:;()\[\]{}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fold_for_matching(text: str) -> str:
    text = normalize_for_matching(text).replace("đ", "d")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
