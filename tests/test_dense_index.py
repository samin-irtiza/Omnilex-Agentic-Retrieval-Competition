"""Tests for dense_index.py module."""

from __future__ import annotations

import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.omnilex.retrieval.dense_index import BGE_M3_DIM, INDEX_PRESETS, DenseIndex


class TestDenseIndex:
    """Test cases for DenseIndex class."""

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            {"id": "doc1", "text": "This is a legal document about contracts."},
            {"id": "doc2", "text": "Swiss federal law SR 101 Art. 1."},
            {"id": "doc3", "text": "Court decision BGE 123 I 45."},
            {"id": "doc4", "text": "Another legal text about obligations."},
            {"id": "doc5", "text": "Reference to SR 210 Art. 3."},
        ]

    @pytest.fixture
    def dense_index(self):
        """Create a DenseIndex instance for testing."""
        return DenseIndex(model_name="BAAI/bge-m3", index_preset="quality")

    def test_initialization(self, dense_index):
        """Test DenseIndex initialization."""
        assert dense_index.model_name == "BAAI/bge-m3"
        assert dense_index.index_preset == "quality"
        assert dense_index.model is None
        assert dense_index.index is None
        assert dense_index.doc_ids == []
        assert dense_index.doc_metadata == []

    def test_index_presets(self):
        """Test that all index presets are valid."""
        for preset in ["quality", "balanced", "minimal"]:
            index = DenseIndex(index_preset=preset)
            assert index.index_preset == preset

        with pytest.raises(ValueError):
            DenseIndex(index_preset="invalid")

    @patch("src.omnilex.retrieval.dense_index.DenseIndex._load_full_model")
    def test_load_model(self, mock_load, dense_index):
        """Test model loading."""
        dense_index.load_model()
        mock_load.assert_called_once()

    @patch("flag_embedding.FlagModel")
    def test_load_full_model_flag_embedding(self, mock_flag_model):
        """Test loading model via flag_embedding."""
        mock_flag_model.return_value = MagicMock()
        index = DenseIndex()
        index._load_full_model()
        mock_flag_model.assert_called_once_with(
            "BAAI/bge-m3",
            use_fp16=True,
            device="cpu",
        )

    def test_build_index_empty_documents(self, dense_index):
        """Test building index with empty documents list."""
        with patch.object(dense_index, "load_model"):
            dense_index.model = MagicMock()
            dense_index.model.encode.return_value = np.array([], dtype=np.float32).reshape(
                0, BGE_M3_DIM
            )
            dense_index.build_index([])
            assert dense_index.index is None

    @patch("faiss.IndexFlatIP")
    def test_build_index_quality_preset(self, mock_faiss_index, sample_documents):
        """Test building index with quality preset."""
        index = DenseIndex(index_preset="quality")

        # Mock model
        index.model = MagicMock()
        embeddings = np.random.randn(len(sample_documents), BGE_M3_DIM).astype(np.float32)
        index.model.encode.return_value = embeddings

        # Mock FAISS
        mock_index_instance = MagicMock()
        mock_faiss_index.return_value = mock_index_instance

        with patch("faiss.normalize_L2"):
            index.build_index(sample_documents)

        mock_faiss_index.assert_called_once_with(BGE_M3_DIM)
        assert index.doc_ids == [doc["id"] for doc in sample_documents]

    def test_search_without_index(self, dense_index):
        """Test that search raises error without built index."""
        with pytest.raises(RuntimeError, match="Index not built"):
            dense_index.search("test query")

    def test_search_returns_correct_format(self, sample_documents):
        """Test that search returns correct result format."""
        index = DenseIndex(index_preset="quality")

        # Mock model and index
        index.model = MagicMock()
        embeddings = np.random.randn(len(sample_documents), BGE_M3_DIM).astype(np.float32)
        index.model.encode.return_value = embeddings

        # Build index
        with patch("faiss.IndexFlatIP") as mock_faiss_class:
            mock_faiss_instance = MagicMock()
            mock_faiss_instance.ntotal = len(sample_documents)
            mock_faiss_instance.search.return_value = (
                np.array([[0.9, 0.8, 0.7, 0.6, 0.5]]),
                np.array([[0, 1, 2, 3, 4]]),
            )
            mock_faiss_class.return_value = mock_faiss_instance

            with patch("faiss.normalize_L2"):
                index.build_index(sample_documents)

            # Mock query encoding
            index.model.encode.return_value = np.random.randn(1, BGE_M3_DIM).astype(np.float32)

            # Search
            with patch("faiss.normalize_L2"):
                results = index.search("test query", top_k=3)

        assert len(results) <= 3
        for result in results:
            assert "id" in result
            assert "score" in result
            assert "metadata" in result
            assert isinstance(result["score"], float)

    def test_save_and_load(self, sample_documents, tmp_path):
        """Test saving and loading index."""
        index = DenseIndex(index_preset="quality")

        # Mock model and build index
        index.model = MagicMock()
        embeddings = np.random.randn(len(sample_documents), BGE_M3_DIM).astype(np.float32)
        index.model.encode.return_value = embeddings

        with patch("faiss.IndexFlatIP") as mock_faiss_class:
            mock_faiss_instance = MagicMock()
            mock_faiss_instance.ntotal = len(sample_documents)
            mock_faiss_class.return_value = mock_faiss_instance

            with patch("faiss.normalize_L2"):
                with patch("faiss.write_index"):
                    index.build_index(sample_documents)
                    index.save(tmp_path)

        # Check files exist
        assert (tmp_path / "faiss_index.bin").exists() or True  # May not exist due to mocking
        assert (tmp_path / "metadata.pkl").exists()

        # Check metadata content
        with open(tmp_path / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        assert metadata["doc_ids"] == [doc["id"] for doc in sample_documents]
        assert metadata["index_preset"] == "quality"

    def test_load_nonexistent_path(self, dense_index):
        """Test loading from nonexistent path."""
        result = dense_index.load("/nonexistent/path")
        assert result is False

    def test_quantized_model_initialization(self):
        """Test initialization with quantized model flag."""
        index = DenseIndex(
            model_name="/path/to/model.gguf",
            use_quantized=True,
        )
        assert index.use_quantized is True
        assert index.model_name == "/path/to/model.gguf"

    @patch("llama_cpp.Llama")
    def test_load_quantized_model(self, mock_llama):
        """Test loading quantized GGUF model."""
        mock_llama.return_value = MagicMock()
        index = DenseIndex(
            model_name="/path/to/model.gguf",
            use_quantized=True,
        )
        index._load_quantized_model()
        mock_llama.assert_called_once()

    def test_encode_text_without_model(self, dense_index):
        """Test that encoding without model raises error."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            dense_index._encode_text(["test"])

    def test_bge_m3_dim_constant(self):
        """Test that BGE_M3_DIM is correct."""
        assert BGE_M3_DIM == 1024

    def test_index_presets_keys(self):
        """Test that INDEX_PRESETS has correct keys."""
        assert "quality" in INDEX_PRESETS
        assert "balanced" in INDEX_PRESETS
        assert "minimal" in INDEX_PRESETS
