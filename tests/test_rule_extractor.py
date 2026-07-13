from src.extraction.rule_extractor import RuleExtractor
from src.extraction.lab_extractor import LabExtractor
from src.io.read_input import InputRecord
from src.preprocess.section_parser import parse_sections


def test_rule_extractor_medications():
    raw = "BN Ä‘ang dÃ¹ng metoprolol 25mg po bid vÃ  aspirin."
    record = InputRecord("x", "x.txt", raw)
    spans = RuleExtractor().extract(record, parse_sections(raw))
    texts = {span.text for span in spans}
    assert "metoprolol 25mg po bid" in texts
    assert "aspirin" in texts


def test_structured_lab_block_supports_unknown_test_names():
    raw = (
        "Kết quả xét nghiệm\n"
        "- Interleukin 6: 42 pg/mL\n"
        "- Cystatin C 1.4 mg/dL\n"
        "Kết quả chẩn đoán hình ảnh\n"
        "- Khối 12 cm"
    )
    record = InputRecord("x", "x.txt", raw)
    spans = LabExtractor().extract(record, parse_sections(raw))
    extracted = {(span.text, span.concept_type) for span in spans}

    assert ("Interleukin 6", "lab_test") in extracted
    assert ("42 pg/mL", "lab_result") in extracted
    assert ("Cystatin C", "lab_test") in extracted
    assert ("1.4 mg/dL", "lab_result") in extracted
    assert ("K", "lab_test") not in extracted
    assert ("Khối", "lab_test") not in extracted
