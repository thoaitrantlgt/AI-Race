from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Candidate:
    id: str
    score: float | None = None
    source: str | None = None


@dataclass
class SpanPrediction:
    text: str
    start: int
    end: int
    concept_type: str
    assertion: str = "isPresent"
    assertions: list[str] | None = None
    candidates: list[Candidate] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "rule"


OUTPUT_TYPE_MAP = {
    "symptom": "TRIỆU_CHỨNG",
    "lab_test": "TÊN_XÉT_NGHIỆM",
    "lab_result": "KẾT_QUẢ_XÉT_NGHIỆM",
    "diagnosis": "CHẨN_ĐOÁN",
    "drug": "THUỐC",
}

OUTPUT_ASSERTIONS = {"isNegated", "isFamily", "isHistorical"}
CANDIDATE_TYPES = {"CHẨN_ĐOÁN", "THUỐC"}


def _submission_candidate_id(candidate_id: str) -> str:
    if candidate_id.startswith("Rx"):
        return candidate_id[2:]
    if candidate_id.startswith("ICD10:"):
        return candidate_id.split(":", 1)[1]
    return candidate_id


def prediction_to_submission_json(pred: SpanPrediction) -> dict:
    output_type = OUTPUT_TYPE_MAP.get(pred.concept_type, pred.concept_type)
    if pred.assertions is not None:
        assertions = [assertion for assertion in pred.assertions if assertion in OUTPUT_ASSERTIONS]
    else:
        assertions = [pred.assertion] if pred.assertion in OUTPUT_ASSERTIONS else []
    candidates = [_submission_candidate_id(candidate.id) for candidate in pred.candidates] if output_type in CANDIDATE_TYPES else []
    return {
        "text": pred.text,
        "position": [pred.start, pred.end],
        "type": output_type,
        "assertions": assertions,
        "candidates": candidates,
    }
