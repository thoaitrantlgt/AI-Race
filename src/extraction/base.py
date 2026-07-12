from __future__ import annotations

from src.io.read_input import InputRecord
from src.io.schema import SpanPrediction
from src.preprocess.section_parser import SectionSpan


class BaseExtractor:
    def extract(self, record: InputRecord, sections: list[SectionSpan]) -> list[SpanPrediction]:
        raise NotImplementedError
