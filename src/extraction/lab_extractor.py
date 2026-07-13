from __future__ import annotations

import re

from src.extraction.base import BaseExtractor
from src.io.read_input import InputRecord
from src.io.schema import SpanPrediction
from src.preprocess.offset_mapper import trim_span_to_valid_boundary
from src.preprocess.section_parser import SectionSpan
from src.preprocess.text_normalizer import fold_for_matching


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
    "cân nặng",
    "chiều cao",
    "nhịp thở",
    "spo2",
}
LAB_BLOCK_HEADINGS = (
    "ket qua xet nghiem",
    "xet nghiem can lam sang",
    "laboratory results",
    "laboratory",
)
LAB_BLOCK_END_HEADINGS = (
    "ket qua chan doan hinh anh",
    "chan doan hinh anh",
    "chan doan",
    "danh gia",
    "dieu tri",
    "thuoc",
    "thu thuat",
    "ke hoach",
    "cac phat hien chan doan",
    "cac phat hien khac",
)
BLOCK_NAME_CONTEXT_SUFFIX_RE = re.compile(
    r"[ \t]+(?:vào[ \t]+ngày|lặp[ \t]+lại|trả[ \t]+về|vẫn|dai[ \t]+dẳng|tăng[ \t]+từ|"
    r"bắt[ \t]+đầu|cải[ \t]+thiện|giảm[ \t]+xuống|tăng[ \t]+lên|tăng)\b.*$",
    re.IGNORECASE,
)
BAD_BLOCK_NAME_CUES = (
    "binh thuong",
    "rung nhi",
    "dap ung that",
    "nhip tim",
    "huyet ap",
)
BLOCK_NAME_RE = r"[A-Za-zÀ-ỹĐđ][^\n:;=]{1,70}?"
BLOCK_VALUE_RE = (
    r"[<>]?[ \t]*\d+(?:[,.]\d+)?"
    r"(?:[ \t]*(?:%|mg/dL|mmol/L|µmol/L|umol/L|g/L|g/dL|U/L|IU/L|ng/L|ng/mL|pg/mL|"
    r"mEq/L|mmHg|10\^9/L|10\^12/L|K/uL|M/uL))?"
)
BLOCK_EXPLICIT_PAIR_RE = re.compile(
    rf"^[ \t]*(?:[-*•][ \t]*)?(?:xét nghiệm[ \t]+)?(?P<name>{BLOCK_NAME_RE})"
    rf"[ \t]+(?:là|bằng|đạt|tăng lên|giảm còn|cải thiện thành)[ \t]+(?P<value>{BLOCK_VALUE_RE})",
    re.IGNORECASE,
)
BLOCK_COLON_PAIR_RE = re.compile(
    rf"^[ \t]*(?:[-*•][ \t]*)?(?:xét nghiệm[ \t]+)?(?P<name>{BLOCK_NAME_RE})"
    rf"[ \t]*[:=][ \t]*(?P<value>{BLOCK_VALUE_RE})",
    re.IGNORECASE,
)
BLOCK_BARE_PAIR_RE = re.compile(
    rf"^[ \t]*(?:[-*•][ \t]*)?(?:xét nghiệm[ \t]+)?(?P<name>{BLOCK_NAME_RE})"
    rf"[ \t]+(?P<value>{BLOCK_VALUE_RE})(?![.,]\d|[A-Za-z0-9])",
    re.IGNORECASE,
)


class LabExtractor(BaseExtractor):
    def __init__(self, terminology_store=None, config: dict | None = None):
        names = "|".join(re.escape(name) for name in sorted(LAB_NAMES, key=len, reverse=True))
        known_name = rf"(?:{names})(?:\s*\([^:\n;]{{1,90}}\))?"
        word_chars = "A-Za-zÀ-ỹĐđ0-9"
        self.known_name_pattern = re.compile(rf"(?<![{word_chars}])({known_name})(?![{word_chars}])")
        self.name_value_pattern = re.compile(
            rf"(?<![{word_chars}])({known_name})\s*[:：]\s*({LAB_VALUE_RE})", re.IGNORECASE
        )
        self.generic_name_value_pattern = re.compile(
            rf"(?<![{word_chars}])({GENERIC_LAB_NAME_RE})\s*[:：]\s*({LAB_VALUE_RE})"
        )
        self.long_test_pattern = re.compile(
            r"(tổng phân tích tế bào máu(?:\s+bằng\s+máy\s+lazer)?(?:\s*\([^)]+\))?|"
            r"công thức máu|sinh hóa máu|khí máu động mạch|phân tích nước tiểu|tổng phân tích nước tiểu|"
            r"x-quang ngực|xquang ngực|điện tâm đồ|siêu âm tim|siêu âm ổ bụng|ct scanner|chụp cắt lớp|mri|monitor holter)",
            re.IGNORECASE,
        )

    @staticmethod
    def _clean_block_name(name: str) -> tuple[str, int]:
        leading_trim = len(name) - len(name.lstrip(" -*\t"))
        cleaned = name.lstrip(" -*\t")
        prefix = re.match(r"(?i)kết[ \t]+quả[ \t]+", cleaned)
        if prefix:
            leading_trim += prefix.end()
            cleaned = cleaned[prefix.end() :]
        suffix = BLOCK_NAME_CONTEXT_SUFFIX_RE.search(cleaned)
        if suffix:
            cleaned = cleaned[: suffix.start()]
        return cleaned.rstrip(" -*\t"), leading_trim

    @staticmethod
    def _usable_block_name(name: str) -> bool:
        cleaned = re.sub(r"\s+", " ", name.strip(" -*\t"))
        folded = fold_for_matching(cleaned)
        base_name = cleaned.split("(", 1)[0].strip()
        if not (2 <= len(cleaned) <= 70) or len(base_name.split()) > 5:
            return False
        if not any(char.isalpha() for char in cleaned):
            return False
        if folded in {fold_for_matching(value) for value in BAD_GENERIC_NAMES}:
            return False
        if any(mark in cleaned for mark in ".!?/"):
            return False
        if any(cue in folded for cue in BAD_BLOCK_NAME_CUES):
            return False
        return True

    def _extract_structured_lab_blocks(self, raw: str, add) -> None:
        in_lab_block = False
        for line_match in re.finditer(r"(?m)^.*$", raw):
            line = line_match.group(0)
            folded = fold_for_matching(line.strip(" -*\t:"))
            if any(folded.startswith(heading) for heading in LAB_BLOCK_HEADINGS):
                in_lab_block = True
                continue
            if in_lab_block and any(folded.startswith(heading) for heading in LAB_BLOCK_END_HEADINGS):
                in_lab_block = False
            if not in_lab_block or not line.strip():
                continue

            pair = BLOCK_EXPLICIT_PAIR_RE.match(line) or BLOCK_COLON_PAIR_RE.match(line) or BLOCK_BARE_PAIR_RE.match(line)
            if pair is None:
                continue
            name, name_leading_trim = self._clean_block_name(pair.group("name"))
            if not self._usable_block_name(name):
                continue
            name_start = line_match.start() + pair.start("name") + name_leading_trim
            name_end = name_start + len(name)
            value_start = line_match.start() + pair.start("value")
            value_end = line_match.start() + pair.end("value")
            add("lab_test", name_start, name_end, 0.84, "lab_structured_name")
            add("lab_result", value_start, value_end, 0.84, "lab_structured_value")

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

        self._extract_structured_lab_blocks(raw, add)

        return spans
