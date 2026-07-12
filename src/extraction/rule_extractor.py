from __future__ import annotations

from src.extraction.base import BaseExtractor
from src.extraction.diagnosis_symptom_extractor import DiagnosisSymptomExtractor
from src.extraction.lab_extractor import LabExtractor
from src.extraction.medication_extractor import MedicationExtractor
from src.io.read_input import InputRecord
from src.io.schema import SpanPrediction
from src.preprocess.section_parser import SectionSpan


class RuleExtractor(BaseExtractor):
    def __init__(self, terminology_store=None, config: dict | None = None):
        self.med_extractor = MedicationExtractor(terminology_store, config)
        self.problem_extractor = DiagnosisSymptomExtractor(terminology_store, config)
        self.lab_extractor = LabExtractor(terminology_store, config)

    def extract(self, record: InputRecord, sections: list[SectionSpan]) -> list[SpanPrediction]:
        return (
            self.med_extractor.extract(record, sections)
            + self.problem_extractor.extract(record, sections)
            + self.lab_extractor.extract(record, sections)
        )
