from __future__ import annotations

import re
from dataclasses import dataclass

from src.preprocess.text_normalizer import normalize_for_matching


SECTION_PATTERNS = {
    "admission_reason": ["lý do nhập viện", "lí do nhập viện"],
    "current_illness": [
        "bệnh sử hiện tại",
        "triệu chứng hiện tại",
        "diễn tiến bệnh",
        "diễn biến",
        "thời điểm khởi phát triệu chứng",
        "các triệu chứng hiện tại",
    ],
    "past_history": ["tiền sử", "tiền căn", "tiền sử bệnh", "tiền sử bệnh nội khoa"],
    "medication_history": [
        "tiền sử dùng thuốc",
        "thuốc đang dùng",
        "thuốc trước khi nhập viện",
        "toa thuốc",
        "medication",
        "medications",
    ],
    "diagnosis": ["chẩn đoán", "đánh giá", "assessment", "diagnosis"],
    "treatment": ["điều trị", "kế hoạch", "plan"],
    "family_history": ["tiền sử gia đình", "gia đình"],
}


@dataclass
class SectionSpan:
    name: str
    start: int
    end: int
    heading_text: str


def _classify_heading(line: str) -> str | None:
    cleaned = re.sub(r"^\s*(?:[#*\-]+|\d+[.)])\s*", "", line)
    cleaned = cleaned.split(":", 1)[0]
    norm = normalize_for_matching(cleaned)
    if not norm or len(norm) > 90:
        return None
    for name, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            pnorm = normalize_for_matching(pattern)
            if norm == pnorm or norm.startswith(pnorm) or pnorm in norm:
                return name
    return None


def parse_sections(raw_text: str) -> list[SectionSpan]:
    matches: list[tuple[int, int, str, str]] = []
    for match in re.finditer(r"(?m)^[ \t]*(?:[#*\-]+|\d+[.)])?[ \t]*([^\n:]{2,90})(?::)?", raw_text):
        line = match.group(0).strip()
        name = _classify_heading(line)
        if name:
            content_start = match.end()
            if content_start < len(raw_text) and raw_text[content_start : content_start + 1] == "\n":
                content_start += 1
            matches.append((match.start(), content_start, name, line))
    if not matches:
        return [SectionSpan("unknown", 0, len(raw_text), "")]
    sections: list[SectionSpan] = []
    for idx, (heading_start, content_start, name, heading) in enumerate(matches):
        end = matches[idx + 1][0] if idx + 1 < len(matches) else len(raw_text)
        sections.append(SectionSpan(name=name, start=heading_start, end=end, heading_text=heading))
    if sections[0].start > 0:
        sections.insert(0, SectionSpan("unknown", 0, sections[0].start, ""))
    return sections


def get_section_for_offset(sections: list[SectionSpan], char_start: int, char_end: int) -> str:
    best_name = "unknown"
    best_overlap = 0
    for section in sections:
        overlap = max(0, min(section.end, char_end) - max(section.start, char_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_name = section.name
    return best_name
