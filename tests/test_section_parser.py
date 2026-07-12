from src.preprocess.section_parser import get_section_for_offset, parse_sections


def test_section_parser_detects_mojibake_headings():
    raw = "LÃ½ do nháº­p viá»‡n:\nÄau ngá»±c\n\nTiá»n sá»­ bá»‡nh:\nTÄƒng huyáº¿t Ã¡p"
    sections = parse_sections(raw)
    names = [section.name for section in sections]
    assert "admission_reason" in names
    assert "past_history" in names
    start = raw.index("TÄƒng")
    assert get_section_for_offset(sections, start, start + 5) == "past_history"
