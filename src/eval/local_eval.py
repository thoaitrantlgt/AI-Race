from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.eval.metrics import jaccard


def _load_dir(path: str | Path) -> dict[str, list[dict]]:
    root = Path(path)
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in root.glob("*.json")}


def evaluate(gold_dir: str, pred_dir: str) -> dict:
    gold = _load_dir(gold_dir)
    pred = _load_dir(pred_dir)
    gold_spans = {(rid, x["start"], x["end"], x["type"]) for rid, rows in gold.items() for x in rows}
    pred_spans = {(rid, x["start"], x["end"], x["type"]) for rid, rows in pred.items() for x in rows}
    tp = len(gold_spans & pred_spans)
    precision = tp / len(pred_spans) if pred_spans else 0.0
    recall = tp / len(gold_spans) if gold_spans else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "span_jaccard": jaccard(gold_spans, pred_spans)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.gold, args.pred), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
