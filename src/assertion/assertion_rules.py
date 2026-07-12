from __future__ import annotations

import re

from src.assertion.cue_lexicons import FAMILY_CUES, HISTORICAL_CUES, NEGATION_CUES, SPECULATION_CUES
from src.assertion.assertion_classifier import AssertionClassifier
from src.io.schema import SpanPrediction
from src.preprocess.section_parser import get_section_for_offset
from src.preprocess.text_normalizer import normalize_for_matching


def _labels(config: dict) -> dict:
    return config.get("assertion", {}).get(
        "assertion_labels",
        {"present": "isPresent", "historical": "isHistorical", "negated": "isNegated", "possible": "isPossible", "family": "isFamily"},
    )


def _sentence_window(raw_text: str, start: int, end: int) -> tuple[str, str, str]:
    left_bound = max(raw_text.rfind(".", 0, start), raw_text.rfind("\n", 0, start), 0)
    right_dot = raw_text.find(".", end)
    right_nl = raw_text.find("\n", end)
    candidates = [pos for pos in (right_dot, right_nl) if pos != -1]
    right_bound = min(candidates) if candidates else len(raw_text)
    left = raw_text[max(left_bound, start - 80) : start]
    sent = raw_text[left_bound:right_bound]
    right = raw_text[end : min(right_bound, end + 80)]
    return left, sent, right


def _contains_cue(text: str, cues: list[str]) -> bool:
    normalized = normalize_for_matching(text)
    return any(re.search(rf"(?<!\w){re.escape(normalize_for_matching(cue))}(?!\w)", normalized) for cue in cues)


def _clause_window(raw_text: str, start: int, end: int) -> tuple[str, str]:
    separators = ".;:\n!?"
    left_bound = max((raw_text.rfind(mark, 0, start) for mark in separators), default=-1) + 1
    right_positions = [raw_text.find(mark, end) for mark in separators]
    right_positions = [position for position in right_positions if position >= 0]
    right_bound = min(right_positions) if right_positions else len(raw_text)
    return raw_text[left_bound:start], raw_text[left_bound:right_bound]


def predict_assertion(span: SpanPrediction, raw_text: str, section_name: str, config: dict) -> str:
    labels = _labels(config)
    left, sentence, _ = _sentence_window(raw_text, span.start, span.end)
    left_norm = normalize_for_matching(left[-70:])
    sent_norm = normalize_for_matching(sentence)
    if section_name == "family_history" or _contains_cue(sentence, FAMILY_CUES):
        return labels.get("family", labels.get("present", "isPresent"))
    if any(normalize_for_matching(cue) in left_norm for cue in NEGATION_CUES):
        return labels.get("negated", labels.get("present", "isPresent"))
    if any(normalize_for_matching(cue) in sent_norm for cue in SPECULATION_CUES):
        return labels.get("possible", labels.get("present", "isPresent"))
    if section_name in {"past_history", "medication_history"} or any(normalize_for_matching(cue) in sent_norm for cue in HISTORICAL_CUES):
        return labels.get("historical", labels.get("present", "isPresent"))
    return labels.get("present", "isPresent")


def repair_assertions_by_scope(spans: list[SpanPrediction], raw_text: str, sections, config: dict) -> list[SpanPrediction]:
    if not config.get("assertion", {}).get("scope_repair", False):
        return spans
    for span in spans:
        if span.concept_type not in {"diagnosis", "drug", "symptom"}:
            span.assertions = []
            span.assertion = "isPresent"
            continue
        section = get_section_for_offset(sections, span.start, span.end)
        left_clause, clause = _clause_window(raw_text, span.start, span.end)
        current = list(span.assertions or ([span.assertion] if span.assertion else []))
        repaired: list[str] = []
        repair_negated = bool(config.get("assertion", {}).get("repair_negated", False))
        repair_family = bool(config.get("assertion", {}).get("repair_family", True))
        repair_historical = bool(config.get("assertion", {}).get("repair_historical", False))
        if "isNegated" in current and (not repair_negated or _contains_cue(left_clause[-70:], NEGATION_CUES)):
            repaired.append("isNegated")
        if "isFamily" in current and (
            not repair_family or section == "family_history" or _contains_cue(clause, FAMILY_CUES)
        ):
            repaired.append("isFamily")
        if "isHistorical" in current and (
            not repair_historical
            or section in {"past_history", "medication_history"}
            or _contains_cue(clause, HISTORICAL_CUES)
        ):
            repaired.append("isHistorical")
        span.assertions = repaired
        span.assertion = repaired[0] if repaired else "isPresent"
    return spans


def predict_assertions_for_spans(spans: list[SpanPrediction], raw_text: str, sections, config: dict) -> list[SpanPrediction]:
    classifier = None
    clf_cfg = config.get("models", {}).get("assertion_classifier", {})
    if clf_cfg.get("enabled", False):
        classifier = AssertionClassifier(
            clf_cfg.get("model_name_or_path"),
            local_files_only=bool(clf_cfg.get("local_files_only", True)),
            device=int(clf_cfg.get("device", -1)),
        )
    for span in spans:
        if (span.source.startswith("model") or span.source.startswith("llm_")) and span.assertions:
            continue
        section = get_section_for_offset(sections, span.start, span.end)
        predicted_many = classifier.predict_many(span, raw_text, section) if classifier is not None else []
        if predicted_many:
            span.assertions = predicted_many
            span.assertion = predicted_many[0]
        else:
            span.assertion = predict_assertion(span, raw_text, section, config)
    return repair_assertions_by_scope(spans, raw_text, sections, config)
