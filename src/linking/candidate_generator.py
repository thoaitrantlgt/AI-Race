from __future__ import annotations

import re
from dataclasses import dataclass

from src.linking.terminology_store import TermEntry, TerminologyStore
from src.preprocess.text_normalizer import normalize_for_matching


@dataclass
class CandidateMatch:
    concept_id: str
    name: str
    source: str
    score: float
    match_type: str
    semantic_type: str | None = None


def _compatible(entry: TermEntry, concept_type: str) -> bool:
    if concept_type == "drug":
        return entry.semantic_type == "drug" or entry.source == "RxNorm"
    if concept_type in {"diagnosis", "symptom"}:
        return entry.semantic_type in {"diagnosis", "problem", "symptom", None} and entry.source != "RxNorm"
    return True


def _base_mentions(mention: str, concept_type: str) -> list[str]:
    mentions = [mention]
    if concept_type == "drug":
        stripped = re.split(r"\s+\d", mention, maxsplit=1)[0]
        if stripped and stripped != mention:
            mentions.append(stripped)
    return mentions


def generate_candidates(mention_text: str, concept_type: str, context_window: str, store: TerminologyStore) -> list[CandidateMatch]:
    if concept_type not in {"drug", "diagnosis"}:
        return []
    found: dict[str, CandidateMatch] = {}
    for mention in _base_mentions(mention_text, concept_type):
        for entry in store.lookup_exact(mention):
            if _compatible(entry, concept_type):
                found.setdefault(entry.concept_id, CandidateMatch(entry.concept_id, entry.name, entry.source, 1.0, "exact", entry.semantic_type))
        for entry in store.lookup_alias(mention):
            if _compatible(entry, concept_type):
                found.setdefault(entry.concept_id, CandidateMatch(entry.concept_id, entry.name, entry.source, 0.95, "alias", entry.semantic_type))
        for entry in store.lookup_folded_exact(mention):
            if _compatible(entry, concept_type):
                found.setdefault(entry.concept_id, CandidateMatch(entry.concept_id, entry.name, entry.source, 0.92, "folded_exact", entry.semantic_type))
    if found:
        return list(found.values())
    for entry in store.lookup_fuzzy(mention_text, limit=20):
        if _compatible(entry, concept_type):
            score = 0.82
            if normalize_for_matching(mention_text) in entry.normalized_name or entry.normalized_name in normalize_for_matching(mention_text):
                score = 0.88
            found.setdefault(entry.concept_id, CandidateMatch(entry.concept_id, entry.name, entry.source, score, "fuzzy", entry.semantic_type))
    if found:
        return list(found.values())
    for entry in store.lookup_by_tokens(mention_text, limit=20):
        if _compatible(entry, concept_type):
            found.setdefault(entry.concept_id, CandidateMatch(entry.concept_id, entry.name, entry.source, 0.70, "token", entry.semantic_type))
    return list(found.values())
