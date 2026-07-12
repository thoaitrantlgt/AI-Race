from __future__ import annotations

import re

from src.extraction.base import BaseExtractor
from src.io.read_input import InputRecord
from src.io.schema import SpanPrediction
from src.preprocess.offset_mapper import trim_span_to_valid_boundary
from src.preprocess.section_parser import SectionSpan, get_section_for_offset


SYMPTOMS = [
    "sốt",
    "ho",
    "ho khan",
    "ho đờm",
    "ho đờm xanh",
    "khó thở",
    "thở khò khè",
    "đau ngực",
    "tức ngực",
    "đau bụng",
    "đau thượng vị",
    "buồn nôn",
    "nôn",
    "ói",
    "chóng mặt",
    "mệt",
    "mệt mỏi",
    "phù",
    "đau đầu",
    "tiêu chảy",
    "táo bón",
    "sụt cân",
    "chán ăn",
    "đau lưng",
    "đau họng",
    "khàn tiếng",
    "hồi hộp",
    "đánh trống ngực",
    "ngất",
    "ngất xỉu",
    "co giật",
    "yếu liệt",
    "tê bì",
    "tê tay",
    "tê chân",
    "cảm giác thắt chặt ngực",
    "khó chịu vùng ngực",
    "ợ hơi",
    "ợ chua",
    "chảy máu",
    "tiểu buốt",
    "tiểu rắt",
    "tiểu máu",
    "vàng da",
    "ngứa",
    "nổi ban",
    "phát ban",
]

DIAGNOSES = [
    "tăng huyết áp",
    "đái tháo đường",
    "tiểu đường",
    "suy tim",
    "nhồi máu cơ tim",
    "bệnh mạch vành",
    "viêm phổi",
    "hen phế quản",
    "copd",
    "bệnh phổi tắc nghẽn mạn tính",
    "suy thận",
    "bệnh thận mạn",
    "xơ gan",
    "viêm gan",
    "tai biến mạch máu não",
    "đột quỵ",
    "nhiễm trùng",
    "nhiễm khuẩn",
    "rối loạn lipid máu",
    "ung thư",
    "thiếu máu",
    "hội chứng não gan",
    "bệnh trào ngược dạ dày - thực quản",
    "trào ngược dạ dày - thực quản",
    "bệnh trào ngược dạ dày- thực quản",
]

SYMPTOM_HEADS = (
    "đau",
    "sốt",
    "ho",
    "khó thở",
    "mệt",
    "nôn",
    "buồn nôn",
    "chóng mặt",
    "phù",
    "ngất",
    "tê",
    "yếu",
    "tiêu chảy",
    "táo bón",
    "ợ",
    "tiểu",
    "chảy máu",
)
GENERIC_DIAGNOSIS_ALIASES = {
    "bệnh",
    "bệnh hiện tại",
    "tiền sử bệnh",
    "triệu chứng",
    "triệu chứng hiện tại",
    "chẩn đoán",
    "điều trị",
    "theo dõi",
    "khác",
    "không xác định",
}

DIAGNOSIS_CUE_RE = re.compile(
    r"(?:chẩn đoán|được chẩn đoán|kết luận|theo dõi|nghĩ nhiều|mắc bệnh|bị bệnh)\s*[:\-]?\s*"
    r"([^.;\n]{3,140})",
    re.IGNORECASE,
)
HISTORY_DIAGNOSIS_RE = re.compile(
    r"(?:tiền sử|ts|bệnh sử)\s+(?:bị|mắc|có|được chẩn đoán)?\s*([^.;\n]{3,120})",
    re.IGNORECASE,
)
COMPLAINT_RE = re.compile(
    r"(?:vào viện vì|lý do vào viện|than phiền|triệu chứng)\s*[:\-]?\s*([^.;\n]{3,160})",
    re.IGNORECASE,
)
SPLIT_RE = re.compile(r"\s*(?:,|;|\bvà\b|\bkèm\b|\s+-\s+)\s*", re.IGNORECASE)


def _compile_terms(terms: list[str]) -> re.Pattern | None:
    clean_terms = [term.strip() for term in terms if term and len(term.strip()) >= 2]
    if not clean_terms:
        return None
    body = "|".join(re.escape(term) for term in sorted(set(clean_terms), key=len, reverse=True))
    return re.compile(rf"(?<!\w)({body})(?!\w)", re.IGNORECASE)


def _looks_vietnamese(text: str) -> bool:
    return any("À" <= ch <= "ỹ" or ch == "đ" or ch == "Đ" for ch in text)


def _usable_diagnosis_alias(alias: str) -> bool:
    alias = alias.strip()
    if not (4 <= len(alias) <= 90):
        return False
    lowered = alias.lower()
    if lowered in GENERIC_DIAGNOSIS_ALIASES:
        return False
    if lowered.startswith(("bệnh ", "các ", "những ")) and len(lowered.split()) <= 3:
        return False
    if any(lowered == head or lowered.startswith(head + " ") for head in SYMPTOM_HEADS):
        return False
    if not _looks_vietnamese(alias):
        return False
    if re.search(r"\d{2,}|[/\\{}\[\]]", alias):
        return False
    return True


def _iter_phrase_parts(text: str) -> list[tuple[int, int]]:
    parts: list[tuple[int, int]] = []
    cursor = 0
    for piece in SPLIT_RE.split(text):
        start = text.find(piece, cursor)
        if start < 0:
            continue
        end = start + len(piece)
        cursor = end
        piece = piece.strip(" :-()[]")
        if 3 <= len(piece) <= 90:
            left_trim = len(text[start:end]) - len(text[start:end].lstrip(" :-()[]"))
            right_trim = len(text[start:end].rstrip(" :-()[]"))
            parts.append((start + left_trim, start + right_trim))
    return parts


class DiagnosisSymptomExtractor(BaseExtractor):
    def __init__(self, terminology_store=None, config: dict | None = None):
        diag = set(DIAGNOSES)
        if terminology_store is not None:
            aliases = [alias for alias in terminology_store.aliases_by_type("diagnosis") if _usable_diagnosis_alias(alias)]
            # Keep the regex bounded. Longer names are more precise and usually
            # carry more private-test value than short generic headings.
            aliases = sorted(set(aliases), key=lambda item: (len(item), item), reverse=True)[:30000]
            diag.update(aliases)
        self.symptom_pattern = _compile_terms(SYMPTOMS)
        self.diagnosis_pattern = _compile_terms(list(diag))

    def _add_span(
        self,
        spans: list[SpanPrediction],
        raw: str,
        start: int,
        end: int,
        concept_type: str,
        confidence: float,
        source: str,
    ) -> None:
        start, end, text = trim_span_to_valid_boundary(raw, start, end)
        text = text.strip(" :-")
        while start < end and raw[start] in " :-":
            start += 1
        while end > start and raw[end - 1] in " :-":
            end -= 1
        if len(text) < 2 or end <= start:
            return
        spans.append(SpanPrediction(raw[start:end], start, end, concept_type, confidence=confidence, source=source))

    def _extract_cued_diagnoses(self, raw: str, spans: list[SpanPrediction]) -> None:
        for pattern, confidence in ((DIAGNOSIS_CUE_RE, 0.80), (HISTORY_DIAGNOSIS_RE, 0.74)):
            for match in pattern.finditer(raw):
                phrase = match.group(1)
                base = match.start(1)
                for rel_start, rel_end in _iter_phrase_parts(phrase):
                    part = phrase[rel_start:rel_end].strip()
                    if len(part) < 4 or any(head in part.lower() for head in SYMPTOM_HEADS):
                        continue
                    self._add_span(spans, raw, base + rel_start, base + rel_end, "diagnosis", confidence, "diagnosis_cue")

    def _extract_cued_symptoms(self, raw: str, spans: list[SpanPrediction]) -> None:
        for match in COMPLAINT_RE.finditer(raw):
            phrase = match.group(1)
            base = match.start(1)
            for rel_start, rel_end in _iter_phrase_parts(phrase):
                part = phrase[rel_start:rel_end].lower()
                if any(part.startswith(head) for head in SYMPTOM_HEADS):
                    self._add_span(spans, raw, base + rel_start, base + rel_end, "symptom", 0.72, "symptom_cue")

    def extract(self, record: InputRecord, sections: list[SectionSpan]) -> list[SpanPrediction]:
        raw = record.raw_text
        spans: list[SpanPrediction] = []
        for pattern, typ in ((self.diagnosis_pattern, "diagnosis"), (self.symptom_pattern, "symptom")):
            if pattern is None:
                continue
            for match in pattern.finditer(raw):
                start, end, text = trim_span_to_valid_boundary(raw, match.start(1), match.end(1))
                section = get_section_for_offset(sections, start, end)
                if section == "medication_history" and typ != "diagnosis":
                    continue
                if typ == "diagnosis":
                    conf = 0.88 if section == "diagnosis" else 0.82 if section == "past_history" else 0.76
                else:
                    conf = 0.84 if section in {"current_illness", "admission_reason"} else 0.68
                spans.append(SpanPrediction(text, start, end, typ, confidence=conf, source=f"{typ}_lexicon"))
        self._extract_cued_diagnoses(raw, spans)
        self._extract_cued_symptoms(raw, spans)
        return spans
