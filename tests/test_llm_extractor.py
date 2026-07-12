from src.extraction.llm_extractor import LlmExtractor
from src.io.read_input import InputRecord
from src.main import validate_parameter_budget


def test_llm_extractor_parses_and_aligns_exact_mentions():
    extractor = LlmExtractor({"enabled": True})
    extractor._request = lambda _: """{
      "entities": [
        {"text":"không ho", "position":[14,22], "type":"TRIỆU_CHỨNG", "assertions":["isNegated"]},
        {"text":"aspirin", "position":[28,35], "type":"THUỐC", "assertions":[]}
      ]
    }"""
    record = InputRecord("1", "1.txt", "Bệnh nhân nói không ho và dùng aspirin.")
    spans = extractor.extract(record, [])
    assert [(span.text, span.concept_type) for span in spans] == [("không ho", "symptom"), ("aspirin", "drug")]
    assert spans[0].assertions == ["isNegated"]


def test_parameter_budget_accepts_qwen3_and_xlmr():
    config = {
        "models": {
            "parameter_budget_billion": 9.0,
            "llm": {"enabled": True, "active_parameters_billion": 8.2},
            "transformer_ner_models": [{"enabled": True, "active_parameters_billion": 0.278}],
        }
    }
    assert validate_parameter_budget(config) == 8.478
