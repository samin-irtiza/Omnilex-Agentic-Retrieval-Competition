"""Integration test to verify all three BM25 indices are built."""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilex.retrieval.ablation.config import ExperimentConfig
from omnilex.retrieval.ablation.runner import ExperimentRunner


@pytest.fixture
def test_config():
    """Create a test config with all corpus paths."""
    return ExperimentConfig(
        name="test_multi_corpus",
        description="Test multi-corpus BM25",
        components={
            "bm25": True,
            "dense": False,
            "graph": False,
            "rrf_fusion": False,
            "reranker": False,
            "verifier": False,
        },
        laws_corpus_path="data/raw/laws_de.csv",
        courts_corpus_path="data/raw/court_considerations.csv",
        non_leading_corpus_path="data/raw/court_decisions.jsonl",
        use_german_stemming=True,
        index_cache_dir="experiments/cache",
    )


class TestMultiCorpusBM25:
    """Test that all three BM25 indices are built when bm25=True."""

    def test_all_three_indices_initialized(self, test_config):
        """Test that all three BM25 indices are initialized."""
        runner = ExperimentRunner(test_config)

        # Verify all three index attributes exist
        assert hasattr(runner, "_bm25_index")
        assert hasattr(runner, "_bm25_courts_index")
        assert hasattr(runner, "_bm25_non_leading_index")

        # Initially all should be None
        assert runner._bm25_index is None
        assert runner._bm25_courts_index is None
        assert runner._bm25_non_leading_index is None

    def test_bm25_laws_index_built(self, test_config):
        """Test that laws BM25 index is built correctly."""
        runner = ExperimentRunner(test_config)

        # Initialize laws index
        runner._bm25_index = runner._init_bm25_index("laws")

        # Verify index is built
        assert runner._bm25_index is not None
        assert runner._bm25_index.index is not None
        assert len(runner._bm25_index.documents) > 0

        # Test search works
        results = runner._bm25_index.search("Gesetz", top_k=5)
        assert len(results) > 0

    def test_bm25_courts_index_built(self, test_config):
        """Test that courts BM25 index is built correctly."""
        runner = ExperimentRunner(test_config)

        # Initialize courts index
        runner._bm25_courts_index = runner._init_bm25_index("courts")

        # Verify index is built
        assert runner._bm25_courts_index is not None
        assert runner._bm25_courts_index.index is not None
        assert len(runner._bm25_courts_index.documents) > 0

        # Test search works
        results = runner._bm25_courts_index.search("Entscheid", top_k=5)
        assert len(results) > 0

    def test_bm25_non_leading_index_built(self, test_config):
        """Test that non-leading decisions BM25 index is built correctly."""
        runner = ExperimentRunner(test_config)

        # Initialize non-leading index
        runner._bm25_non_leading_index = runner._init_bm25_index("non_leading")

        # Verify index is built
        assert runner._bm25_non_leading_index is not None
        assert runner._bm25_non_leading_index.index is not None
        assert len(runner._bm25_non_leading_index.documents) > 0

        # Test search works
        results = runner._bm25_non_leading_index.search("BGE", top_k=5)
        assert len(results) > 0

    def test_run_retrieval_returns_all_signals(self, test_config):
        """Test that _run_retrieval returns signals from all three indices."""
        runner = ExperimentRunner(test_config)

        # Run retrieval with a test query
        signals = runner._run_retrieval("Gesetz", "test_query")

        # Verify all three signals are present
        assert "bm25_laws" in signals
        assert "bm25_courts" in signals
        assert "bm25_non_leading" in signals

        # Verify each signal has results
        assert len(signals["bm25_laws"]) > 0
        assert len(signals["bm25_courts"]) > 0
        assert len(signals["bm25_non_leading"]) > 0

    def test_german_stemming_enabled(self, test_config):
        """Test that German stemming is enabled in all indices."""
        runner = ExperimentRunner(test_config)

        # Build all three indices
        runner._bm25_index = runner._init_bm25_index("laws")
        runner._bm25_courts_index = runner._init_bm25_index("courts")
        runner._bm25_non_leading_index = runner._init_bm25_index("non_leading")

        # Verify stemming is enabled
        assert runner._bm25_index.use_german_stemming is True
        assert runner._bm25_courts_index.use_german_stemming is True
        assert runner._bm25_non_leading_index.use_german_stemming is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
