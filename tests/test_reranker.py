"""Tests for the cross-encoder reranker module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from src.omnilex.retrieval.reranker import Reranker


class TestReranker:
    """Test cases for Reranker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.reranker = Reranker(
            model_name="BAAI/bge-reranker-v2-m3",
            device="cpu",
            batch_size=4,
        )
        self.query = "What are the requirements for contract formation?"
        self.documents = [
            {"text": "A contract requires offer, acceptance, and consideration."},
            {"text": "Tort law deals with civil wrongs and damages."},
            {"text": "Contract formation needs mutual assent and consideration."},
            {"text": "Property law governs ownership rights."},
        ]

    def test_initialization(self):
        """Test reranker initialization."""
        assert self.reranker.model_name == "BAAI/bge-reranker-v2-m3"
        assert self.reranker.device == "cpu"
        assert self.reranker.batch_size == 4
        assert self.reranker.model is None
        assert self.reranker.tokenizer is None

    @patch("src.omnilex.retrieval.reranker.AutoTokenizer")
    @patch("src.omnilex.retrieval.reranker.AutoModelForSequenceClassification")
    def test_load_model(self, mock_model_class, mock_tokenizer_class):
        """Test model loading."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        self.reranker.load_model()

        mock_tokenizer_class.from_pretrained.assert_called_once_with("BAAI/bge-reranker-v2-m3")
        mock_model_class.from_pretrained.assert_called_once_with("BAAI/bge-reranker-v2-m3")
        mock_model.to.assert_called_once_with("cpu")
        mock_model.eval.assert_called_once()
        assert self.reranker.tokenizer == mock_tokenizer
        assert self.reranker.model == mock_model

    def test_normalize_scores(self):
        """Test score normalization with sigmoid."""
        # Test with various logit values
        scores = [-10.0, -1.0, 0.0, 1.0, 10.0]
        normalized = self.reranker.normalize_scores(scores)

        # Check all scores are in [0, 1]
        for score in normalized:
            assert score >= 0.0
            assert score <= 1.0

        # Check sigmoid properties
        assert normalized[0] < normalized[1]  # -10 < -1
        assert normalized[1] < normalized[2]  # -1 < 0
        assert normalized[2] < normalized[3]  # 0 < 1
        assert normalized[3] < normalized[4]  # 1 < 10

        # Check specific values (approximate)
        assert abs(normalized[2] - 0.5) < 10 ** (-3)  # sigmoid(0) = 0.5

    def test_normalize_scores_empty(self):
        """Test normalization with empty scores."""
        result = self.reranker.normalize_scores([])
        assert result == []

    def test_score_empty_documents(self):
        """Test scoring with empty documents list."""
        scores = self.reranker.score("query", [])
        assert scores == []

    @patch.object(Reranker, "_ensure_model_loaded")
    def test_score_with_mock(self, mock_ensure):
        """Test scoring functionality with mocked model."""
        # Setup mock tokenizer
        mock_tokenizer = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.to = MagicMock(return_value=mock_inputs)
        mock_tokenizer.return_value = mock_inputs

        # Setup mock model
        mock_model = MagicMock()
        mock_outputs = MagicMock()
        mock_logits = MagicMock()
        mock_logits.squeeze = MagicMock(return_value=mock_logits)
        mock_logits.cpu = MagicMock(return_value=mock_logits)
        mock_logits.tolist = MagicMock(return_value=[-2.5, 1.2, 0.8, -0.5])
        mock_outputs.logits = mock_logits
        mock_model.return_value = mock_outputs

        with (
            patch.object(self.reranker, "tokenizer", mock_tokenizer),
            patch.object(self.reranker, "model", mock_model),
        ):
            scores = self.reranker.score(self.query, self.documents)

            assert len(scores) == 4
            assert scores == [-2.5, 1.2, 0.8, -0.5]

    def test_rerank_empty_documents(self):
        """Test reranking with empty documents."""
        result = self.reranker.rerank("query", [])
        assert result == []

    @patch.object(Reranker, "_ensure_model_loaded")
    def test_rerank(self, mock_ensure):
        """Test reranking functionality."""
        # Mock the score and normalize methods
        with (
            patch.object(self.reranker, "score", return_value=[-2.5, 1.2, 0.8, -0.5]),
            patch.object(
                self.reranker,
                "normalize_scores",
                return_value=[0.075, 0.769, 0.690, 0.378],
            ),
        ):
            result = self.reranker.rerank(self.query, self.documents)

            assert len(result) == 4
            # Check that documents are sorted by score descending
            assert result[0]["reranker_score"] == 0.769
            assert result[1]["reranker_score"] == 0.690
            assert result[2]["reranker_score"] == 0.378
            assert result[3]["reranker_score"] == 0.075

            # Check that reranker_score is added to each doc
            for doc in result:
                assert "reranker_score" in doc

    @patch.object(Reranker, "_ensure_model_loaded")
    def test_rerank_top_k(self, mock_ensure):
        """Test reranking with top_k parameter."""
        with (
            patch.object(self.reranker, "score", return_value=[-2.5, 1.2, 0.8, -0.5]),
            patch.object(
                self.reranker,
                "normalize_scores",
                return_value=[0.075, 0.769, 0.690, 0.378],
            ),
        ):
            result = self.reranker.rerank(self.query, self.documents, top_k=2)

            assert len(result) == 2
            assert result[0]["reranker_score"] == 0.769
            assert result[1]["reranker_score"] == 0.690

    def test_batch_processing(self):
        """Test that batch processing works correctly."""
        # Create more documents than batch_size
        docs = [{"text": f"Document {i}"} for i in range(10)]
        self.reranker.batch_size = 3

        with (
            patch.object(self.reranker, "model", MagicMock()),
            patch.object(self.reranker, "tokenizer", MagicMock()),
            patch.object(self.reranker, "_ensure_model_loaded"),
        ):
            # Mock score to return appropriate number of scores
            with (
                patch.object(self.reranker, "score", return_value=list(range(10))),
                patch.object(
                    self.reranker,
                    "normalize_scores",
                    return_value=[float(i) / 10 for i in range(10)],
                ),
            ):
                result = self.reranker.rerank(self.query, docs)

                assert len(result) == 10
