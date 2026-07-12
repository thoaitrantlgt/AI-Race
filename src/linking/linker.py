from __future__ import annotations

from src.io.schema import Candidate, SpanPrediction
from src.linking.candidate_generator import generate_candidates
from src.linking.candidate_ranker import rank_candidates
from src.linking.sapbert_ranker import get_sapbert_reranker
from src.linking.terminology_store import TerminologyStore
from src.postprocess.candidate_filter import filter_candidates


def get_context_window(raw_text: str, start: int, end: int, window_chars: int = 120) -> str:
    return raw_text[max(0, start - window_chars) : min(len(raw_text), end + window_chars)]


def link_spans(spans: list[SpanPrediction], raw_text: str, sections, terminology_store: TerminologyStore, config: dict) -> list[SpanPrediction]:
    for span in spans:
        augment_model = bool(config.get("linking", {}).get("augment_model_candidates", False))
        if span.source.startswith("model") and (span.candidates or not augment_model):
            continue
        if span.concept_type not in {"diagnosis", "drug"}:
            continue
        context = get_context_window(raw_text, span.start, span.end)
        generated = generate_candidates(span.text, span.concept_type, context, terminology_store)
        ranked = rank_candidates(
            span.text,
            span.concept_type,
            generated,
            context,
            use_context=bool(config.get("linking", {}).get("contextual_reranking", False)),
        )
        sapbert_cfg = config.get("models", {}).get("sapbert", {})
        if sapbert_cfg.get("enabled", False):
            reranker = get_sapbert_reranker(
                sapbert_cfg.get("model_name_or_path", "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"),
                local_files_only=bool(sapbert_cfg.get("local_files_only", True)),
            )
            ranked = reranker.rerank(span.text, ranked, weight=float(sapbert_cfg.get("weight", 0.18)))
        filtered = filter_candidates(ranked, config)
        existing = {candidate.id for candidate in span.candidates}
        for item in filtered:
            if item.concept_id not in existing:
                span.candidates.append(Candidate(item.concept_id, item.score, item.source))
                existing.add(item.concept_id)
        max_candidates = int(config.get("linking", {}).get("max_candidates", 2))
        span.candidates = span.candidates[:max_candidates]
    return spans
