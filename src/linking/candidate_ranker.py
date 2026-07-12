from __future__ import annotations

from difflib import SequenceMatcher

from src.linking.candidate_generator import CandidateMatch
from src.preprocess.text_normalizer import normalize_for_matching


SOURCE_PRIORITY = {"RxNorm": 4, "ICD10_TT06": 4, "ICD10_VI": 3, "custom": 2, "SNOMED": 1}


def _tokens(text: str) -> set[str]:
    return {token for token in normalize_for_matching(text).split() if len(token) > 1}


def _contextual_score(mention: str, candidate: CandidateMatch, context_window: str) -> float:
    mention_norm = normalize_for_matching(mention)
    name_norm = normalize_for_matching(candidate.name)
    lexical = SequenceMatcher(None, mention_norm, name_norm).ratio()
    mention_tokens = _tokens(mention)
    name_tokens = _tokens(candidate.name)
    context_tokens = _tokens(context_window)
    mention_coverage = len(mention_tokens & name_tokens) / max(1, len(mention_tokens))
    context_coverage = len(name_tokens & context_tokens) / max(1, len(name_tokens))
    return candidate.score + 0.08 * lexical + 0.05 * mention_coverage + 0.02 * context_coverage


def rank_candidates(
    mention: str,
    concept_type: str,
    candidates: list[CandidateMatch],
    context_window: str,
    use_context: bool = False,
) -> list[CandidateMatch]:
    return sorted(
        candidates,
        key=lambda c: (
            _contextual_score(mention, c, context_window) if use_context else c.score,
            SOURCE_PRIORITY.get(c.source, 0),
            -abs(len(c.name) - len(mention)),
        ),
        reverse=True,
    )
