from __future__ import annotations

from src.linking.candidate_generator import CandidateMatch


def filter_candidates(ranked_candidates: list[CandidateMatch], config: dict) -> list[CandidateMatch]:
    if not ranked_candidates:
        return []
    linking = config.get("linking", {})
    high = float(linking.get("high_confidence_threshold", 0.88))
    medium = float(linking.get("medium_confidence_threshold", 0.72))
    max_candidates = int(linking.get("max_candidates", 2))
    top = ranked_candidates[0]
    if top.score >= high:
        return [top][:max_candidates]
    if top.score >= medium:
        if len(ranked_candidates) >= 2 and abs(top.score - ranked_candidates[1].score) <= 0.05:
            return ranked_candidates[:max_candidates]
        return [top]
    return []
