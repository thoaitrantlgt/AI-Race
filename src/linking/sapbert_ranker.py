from __future__ import annotations

from functools import lru_cache

import numpy as np

from src.linking.candidate_generator import CandidateMatch


class SapBertReranker:
    def __init__(self, model_name_or_path: str, local_files_only: bool = True):
        self.enabled = False
        self.tokenizer = None
        self.sequence_model = None
        self.encoder = None
        self.mode = "disabled"
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only)
            self.sequence_model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, local_files_only=local_files_only)
            self.sequence_model.eval()
            self.enabled = True
            self.mode = "cross_encoder"
            return
        except Exception as exc:
            print(f"Warning: could not load sequence reranker {model_name_or_path}: {exc}")
        try:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(model_name_or_path, local_files_only=local_files_only)
            self.enabled = True
            self.mode = "bi_encoder"
        except Exception as exc:
            print(f"Warning: could not load SapBERT reranker {model_name_or_path}: {exc}")

    def _sequence_scores(self, mention: str, candidates: list[CandidateMatch]) -> list[float]:
        assert self.tokenizer is not None and self.sequence_model is not None
        pairs_left = [mention] * len(candidates)
        pairs_right = [candidate.name for candidate in candidates]
        inputs = self.tokenizer(pairs_left, pairs_right, padding=True, truncation=True, return_tensors="pt", max_length=256)
        with self._torch.no_grad():
            logits = self.sequence_model(**inputs).logits
        if logits.ndim == 2 and logits.shape[1] > 1:
            scores = logits[:, -1]
        else:
            scores = logits.reshape(-1)
        return [float(score) for score in scores.detach().cpu().tolist()]

    def _bi_scores(self, mention: str, candidates: list[CandidateMatch]) -> list[float]:
        assert self.encoder is not None
        names = [candidate.name for candidate in candidates]
        vectors = np.asarray(self.encoder.encode([mention] + names, normalize_embeddings=True, show_progress_bar=False))
        mention_vec = vectors[0]
        candidate_vecs = vectors[1:]
        return [float(score) for score in candidate_vecs @ mention_vec]

    def rerank(self, mention: str, candidates: list[CandidateMatch], weight: float = 0.18) -> list[CandidateMatch]:
        if not self.enabled or not candidates:
            return candidates
        try:
            model_scores = self._sequence_scores(mention, candidates) if self.sequence_model is not None else self._bi_scores(mention, candidates)
        except Exception as exc:
            print(f"Warning: SapBERT reranker inference failed: {exc}")
            return candidates
        min_score = min(model_scores)
        max_score = max(model_scores)
        denom = max(max_score - min_score, 1e-6)
        reranked: list[CandidateMatch] = []
        for candidate, model_score in zip(candidates, model_scores):
            normalized = (model_score - min_score) / denom
            candidate.score = min(1.0, max(candidate.score, candidate.score * (1.0 - weight) + normalized * weight))
            candidate.match_type = f"{candidate.match_type}+{self.mode}"
            reranked.append(candidate)
        return sorted(reranked, key=lambda item: item.score, reverse=True)


@lru_cache(maxsize=2)
def get_sapbert_reranker(model_name_or_path: str, local_files_only: bool = True) -> SapBertReranker:
    return SapBertReranker(model_name_or_path, local_files_only=local_files_only)
