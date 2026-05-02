"""Cross-encoder reranking module using BGE-reranker-v2-m3.

Provides:
- BGE-reranker-v2-m3 model loading
- Cross-encoder scoring for candidate citations
- Score normalization to [0, 1] range (sigmoid)
- Batch reranking for efficiency
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-encoder reranker using BGE-reranker-v2-m3."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        """Initialize reranker.

        Args:
            model_name: HuggingFace model name
            device: Device to run on ("cpu" or "cuda")
            batch_size: Batch size for reranking
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        """Load BGE-reranker-v2-m3 model and tokenizer."""
        logger.info(f"Loading reranker model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Reranker model loaded on {self.device}")

    def _ensure_model_loaded(self) -> None:
        """Ensure model and tokenizer are loaded."""
        if self.model is None or self.tokenizer is None:
            self.load_model()

    def score(
        self,
        query: str,
        documents: list[dict],
    ) -> list[float]:
        """Score documents using cross-encoder.

        Args:
            query: Query text
            documents: List of document dicts with 'text' key

        Returns:
            List of relevance scores (logits)
        """
        if not documents:
            return []

        self._ensure_model_loaded()

        texts = [doc.get("text", "") for doc in documents]
        scores = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_pairs = [[query, text] for text in batch_texts]

            # Tokenize
            inputs = self.tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Get logits for relevance (shape: [batch_size, num_labels])
                batch_scores = outputs.logits.squeeze(-1).cpu().tolist()

                # Handle single item case
                if not isinstance(batch_scores, list):
                    batch_scores = [batch_scores]

                scores.extend(batch_scores)

        return scores

    def normalize_scores(
        self,
        scores: list[float],
    ) -> list[float]:
        """Normalize scores to [0, 1] range using sigmoid.

        Args:
            scores: Raw logit scores

        Returns:
            Normalized scores in [0, 1]
        """
        if not scores:
            return []

        scores_array = np.array(scores)
        normalized = 1 / (1 + np.exp(-scores_array))
        return normalized.tolist()

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """Rerank documents and return sorted list.

        Args:
            query: Query text
            documents: List of document dicts
            top_k: Number of top results to return (None = all)

        Returns:
            List of documents with 'reranker_score' added, sorted by score
        """
        if not documents:
            return []

        # Score documents
        raw_scores = self.score(query, documents)

        # Normalize scores
        normalized_scores = self.normalize_scores(raw_scores)

        # Add scores to documents
        scored_docs = []
        for doc, score in zip(documents, normalized_scores):
            doc_copy = doc.copy()
            doc_copy["reranker_score"] = score
            scored_docs.append(doc_copy)

        # Sort by score descending
        scored_docs.sort(key=lambda x: x["reranker_score"], reverse=True)

        # Return top_k if specified
        if top_k is not None:
            return scored_docs[:top_k]

        return scored_docs
