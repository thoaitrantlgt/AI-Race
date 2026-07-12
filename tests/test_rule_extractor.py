from src.extraction.rule_extractor import RuleExtractor
from src.io.read_input import InputRecord
from src.preprocess.section_parser import parse_sections


def test_rule_extractor_medications():
    raw = "BN Ä‘ang dÃ¹ng metoprolol 25mg po bid vÃ  aspirin."
    record = InputRecord("x", "x.txt", raw)
    spans = RuleExtractor().extract(record, parse_sections(raw))
    texts = {span.text for span in spans}
    assert "metoprolol 25mg po bid" in texts
    assert "aspirin" in texts
