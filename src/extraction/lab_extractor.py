from __future__ import annotations

import re

from src.extraction.base import BaseExtractor
from src.io.read_input import InputRecord
from src.io.schema import SpanPrediction
from src.preprocess.offset_mapper import trim_span_to_valid_boundary
from src.preprocess.section_parser import SectionSpan


LAB_NAMES = [
    "WBC",
    "TWBC",
    "NEUT%",
    "LYPH%",
    "LYMPH%",
    "RBC",
    "HGB",
    "Hb",
    "HCT",
    "PLT",
    "MCV",
    "MCH",
    "MCHC",
    "ALT",
    "AST",
    "GGT",
    "ALP",
    "CRP",
    "ESR",
    "Creatinine",
    "Ure",
    "BUN",
    "Glucose",
    "Troponin",
    "BNP",
    "NT-proBNP",
    "INR",
    "PT",
    "aPTT",
    "Na",
    "K",
    "Cl",
    "HbA1c",
    "Bilirubin",
    "Albumin",
    "Protein",
    "LDH",
    "CK",
    "CK-MB",
    "D-dimer",
    "Ferritin",
    "Procalcitonin",
    "TSH",
    "FT3",
    "FT4",
    "T3",
    "T4",
]

LAB_VALUE_RE = (
    r"[<>]?\s*\d+(?:[,.]\d+)?"
    r"(?:\s*(?:mg/dL|mmol/L|µmol/L|umol/L|g/L|g/dL|U/L|IU/L|ng/L|ng/mL|pg/mL|mEq/L|mmHg|10\^9/L|10\^12/L|%))?"
)
GENERIC_LAB_NAME_RE = r"[A-Za-zÀ-ỹĐđ][A-Za-zÀ-ỹĐđ0-9%/\- ]{1,80}"
BAD_GENERIC_NAMES = {
    "tuổi",
    "mạch",
    "nhiệt độ",
    "huyết áp",
    "địa chỉ",
    "số điện thoại",
    "ngày",
    "tháng",
    "năm",
    "nam",
    "nữ",
}


class LabExtractor(BaseExtractor):
    def __init__(self, terminology_store=None, config: dict | None = None):
        names = "|".join(re.escape(name) for name in sorted(LAB_NAMES, key=len, reverse=True))
        known_name = rf"(?:{names})(?:\s*\([^:\n;]{{1,90}}\))?"
        self.known_name_pattern = re.compile(rf"(?<![A-Za-z0-9])({known_name})(?![A-Za-z0-9])")
        self.name_value_pattern = re.compile(rf"(?<![A-Za-z0-9])({known_name})\s*[:：]\s*({LAB_VALUE_RE})", re.IGNORECASE)
        self.generic_name_value_pattern = re.compile(rf"(?<![A-Za-z0-9])({GENERIC_LAB_NAME_RE})\s*[:：]\s*({LAB_VALUE_RE})")
        self.long_test_pattern = re.compile(
            r"(tổng phân tích tế bào máu(?:\s+bằng\s+máy\s+lazer)?(?:\s*\([^)]+\))?|"
            r"công thức máu|sinh hóa máu|khí máu động mạch|phân tích nước tiểu|tổng phân tích nước tiểu|"
            r"x-quang ngực|xquang ngực|điện tâm đồ|siêu âm tim|siêu âm ổ bụng|ct scanner|chụp cắt lớp|mri|monitor holter)",
            re.IGNORECASE,
        )

    def extract(self, record: InputRecord, sections: list[SectionSpan]) -> list[SpanPrediction]:
        raw = record.raw_text
        spans: list[SpanPrediction] = []
        seen: set[tuple[int, int, str]] = set()

        def add(text_type: str, start: int, end: int, confidence: float, source: str) -> None:
            if text_type == "lab_test":
                start2, end2 = start, end
                while start2 < end2 and raw[start2].isspace():
                    start2 += 1
                while end2 > start2 and raw[end2 - 1].isspace():
                    end2 -= 1
                text = raw[start2:end2]
            else:
                start2, end2, text = trim_span_to_valid_boundary(raw, start, end)
            if not text:
                return
            key = (start2, end2, text_type)
            if key not in seen:
                seen.add(key)
                spans.append(SpanPrediction(text, start2, end2, text_type, confidence=confidence, source=source))

        for match in self.name_value_pattern.finditer(raw):
            name = match.group(1).strip()
            if len(name) > 120:
                continue
            add("lab_test", match.start(1), match.end(1), 0.86, "lab_name_value")
            add("lab_result", match.start(2), match.end(2), 0.84, "lab_value")

        for match in self.generic_name_value_pattern.finditer(raw):
            name = re.sub(r"\s+", " ", match.group(1).strip())
            leading_cue = re.match(r"(?i)^(?:xét nghiệm|xn|kết quả)\s+", name)
            name_start = match.start(1)
            if leading_cue:
                name_start += leading_cue.end()
                name = name[leading_cue.end() :].strip()
            if len(name) > 80 or name.lower() in BAD_GENERIC_NAMES:
                continue
            if not any(ch.isalpha() for ch in name):
                continue
            add("lab_test", name_start, match.end(1), 0.72, "lab_generic_name_value")
            add("lab_result", match.start(2), match.end(2), 0.78, "lab_generic_value")

        for match in self.known_name_pattern.finditer(raw):
            add("lab_test", match.start(1), match.end(1), 0.76, "lab_name")

        for match in self.long_test_pattern.finditer(raw):
            add("lab_test", match.start(1), match.end(1), 0.78, "lab_phrase")

        return spans
