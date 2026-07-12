from src.io.read_input import InputRecord
from src.io.schema import SpanPrediction
from src.io.validate_output import validate_output
from src.io.write_output import write_all_json


def test_json_schema(tmp_path):
    raw = "aspirin"
    records = [InputRecord("1", "1.txt", raw)]
    outputs = {"1": [SpanPrediction("aspirin", 0, 7, "drug", "isPresent", [])]}
    write_all_json(outputs, tmp_path)
    validate_output(tmp_path, records, {"linking": {"max_candidates": 2}})
