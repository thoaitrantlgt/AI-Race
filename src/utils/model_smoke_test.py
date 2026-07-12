from __future__ import annotations

import argparse
import json

from src.extraction.model_extractor import ModelExtractor
from src.io.read_input import read_input_dir
from src.linking.candidate_generator import CandidateMatch
from src.linking.sapbert_ranker import get_sapbert_reranker
from src.main import load_config


def run_smoke(config_path: str, input_dir: str) -> dict:
    config = load_config(config_path)
    records = read_input_dir(input_dir)
    model = ModelExtractor(config.get("pipeline", {}).get("model_path"), config.get("models", {}))
    first = records[0]
    spans = model.extract(first, [])
    result = {
        "pseudo_or_transformer_model_enabled": model.enabled,
        "transformer_ner_loaded": len(model.transformer_extractors),
        "first_record_model_spans": len(spans),
        "sapbert_loaded": False,
        "sapbert_top_candidate": None,
    }
    sapbert_cfg = config.get("models", {}).get("sapbert", {})
    if sapbert_cfg.get("enabled", False):
        reranker = get_sapbert_reranker(
            sapbert_cfg.get("model_name_or_path", "BAAI/bge-reranker-v2-m3"),
            local_files_only=bool(sapbert_cfg.get("local_files_only", True)),
        )
        candidates = [
            CandidateMatch("Rx6918", "metoprolol", "RxNorm", 0.75, "smoke", "drug"),
            CandidateMatch("Rx1191", "aspirin", "RxNorm", 0.75, "smoke", "drug"),
        ]
        ranked = reranker.rerank("metoprolol 25mg po bid", candidates)
        result["sapbert_loaded"] = reranker.enabled
        result["sapbert_top_candidate"] = ranked[0].concept_id if ranked else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input_dir", default="data/input")
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.config, args.input_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
