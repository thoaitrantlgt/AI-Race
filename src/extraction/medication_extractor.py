from __future__ import annotations

import re

from src.extraction.base import BaseExtractor
from src.io.read_input import InputRecord
from src.io.schema import SpanPrediction
from src.preprocess.offset_mapper import trim_span_to_valid_boundary
from src.preprocess.section_parser import SectionSpan


DEFAULT_DRUG_ALIASES = [
    "aspirin",
    "amlodipine",
    "metoprolol",
    "atenolol",
    "doxycycline",
    "levofloxacin",
    "tylenol",
    "paracetamol",
    "acetaminophen",
    "insulin",
    "warfarin",
    "heparin",
    "ceftriaxone",
    "omeprazole",
    "atorvastatin",
    "simvastatin",
    "furosemide",
    "losartan",
    "metformin",
    "chlorpheniramine",
    "capsaicin",
    "salbutamol",
    "prednisolone",
    "cefuroxime",
    "azithromycin",
    "amoxicillin",
    "clavulanate",
    "pantoprazole",
    "esomeprazole",
]

UNIT = r"(?:mg|mcg|µg|g|ml|mL|l|iu|IU|units?|đv|%)(?![A-Za-zÀ-ỹĐđ])"
STRENGTH = rf"\d+(?:[.,]\d+)?\s*{UNIT}(?:\s*/\s*{UNIT})?"
ROUTE = r"(?:po|iv|im|sc|uống|tiêm|truyền|ngậm|bôi|xịt|nhỏ)"
FREQ = r"(?:qd|bid|tid|qid|qhs|prn|mỗi ngày|ngày\s*\d+\s*lần|x\s*\d+|sáng|trưa|chiều|tối)"
DRUG_TOKEN = r"[A-Za-z][A-Za-z0-9+/\-]*"


class MedicationExtractor(BaseExtractor):
    def __init__(self, terminology_store=None, config: dict | None = None):
        self.config = config or {}
        aliases = set(DEFAULT_DRUG_ALIASES)
        if terminology_store is not None:
            aliases.update(terminology_store.aliases_by_type("drug"))
        if len(aliases) > 5000:
            aliases = set(DEFAULT_DRUG_ALIASES) | {alias for alias in aliases if len(alias) <= 48 and " " not in alias}
        self.aliases = sorted(aliases, key=len, reverse=True)
        self.default_aliases = {alias.lower() for alias in DEFAULT_DRUG_ALIASES}
        names = "|".join(re.escape(a) for a in self.aliases)
        self.alias_pattern = re.compile(rf"(?<![A-Za-z0-9])({names})(?![A-Za-z0-9])", re.IGNORECASE) if names else None
        self.med_with_dose = re.compile(
            rf"(?<![A-Za-z0-9])({DRUG_TOKEN}(?:\s+{DRUG_TOKEN}){{0,3}}\s+{STRENGTH}(?:\s+{ROUTE})?(?:\s+{FREQ})?)",
            re.IGNORECASE,
        )
        self.rx_like_line = re.compile(
            rf"(?:thuốc|dùng|sử dụng|điều trị|toa thuốc)\s*[:\-]?\s*"
            rf"({DRUG_TOKEN}(?:\s+{DRUG_TOKEN}){{0,3}}(?:\s+{STRENGTH})?)",
            re.IGNORECASE,
        )

    def _looks_like_known_drug(self, text: str) -> bool:
        lowered = text.lower()
        return any(lowered.startswith(alias.lower()) for alias in self.aliases)

    def _add(self, spans: list[SpanPrediction], raw: str, start: int, end: int, confidence: float, source: str) -> None:
        start, end, text = trim_span_to_valid_boundary(raw, start, end)
        if len(text) >= 2:
            spans.append(SpanPrediction(text, start, end, "drug", confidence=confidence, source=source))

    def extract(self, record: InputRecord, sections: list[SectionSpan]) -> list[SpanPrediction]:
        spans: list[SpanPrediction] = []
        raw = record.raw_text
        for match in self.med_with_dose.finditer(raw):
            text = raw[match.start(1) : match.end(1)]
            if not self._looks_like_known_drug(text):
                continue
            self._add(spans, raw, match.start(1), match.end(1), 0.92, "med_regex")

        if self.alias_pattern is not None:
            for match in self.alias_pattern.finditer(raw):
                start, end = match.start(1), match.end(1)
                alias_text = raw[start:end].lower()
                tail = raw[end : min(len(raw), end + 70)]
                dose = re.match(rf"\s+{STRENGTH}(?:\s+{ROUTE})?(?:\s+{FREQ})?", tail, re.IGNORECASE)
                if dose:
                    end += dose.end()
                    conf = 0.92
                else:
                    left = raw[max(0, start - 80) : start].lower()
                    if alias_text not in self.default_aliases and not re.search(r"(thuốc|toa|dùng|sử dụng|điều trị)\W*$", left):
                        continue
                    conf = 0.88
                self._add(spans, raw, start, end, conf, "med_alias")

        for match in self.rx_like_line.finditer(raw):
            text = raw[match.start(1) : match.end(1)]
            if self._looks_like_known_drug(text) or re.search(STRENGTH, text, re.IGNORECASE):
                self._add(spans, raw, match.start(1), match.end(1), 0.70, "med_cue")
        return spans
