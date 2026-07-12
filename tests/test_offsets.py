from src.preprocess.offset_mapper import trim_span_to_valid_boundary, validate_span


def test_validate_span():
    raw = "Bá»‡nh nhÃ¢n dÃ¹ng amlodipine 10mg má»—i ngÃ y."
    span = "amlodipine 10mg"
    start = raw.index(span)
    end = start + len(span)
    assert validate_span(raw, start, end, span)


def test_trim_span_to_valid_boundary():
    raw = "  aspirin 325mg, "
    start, end, text = trim_span_to_valid_boundary(raw, 0, len(raw))
    assert text == "aspirin 325mg"
    assert raw[start:end] == text
