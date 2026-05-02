"""Tests for SignalFusion module."""

from src.omnilex.retrieval.fusion import RRF_K, SignalFusion


class TestSignalFusion:
    """Test suite for SignalFusion."""

    def test_basic_rrf_two_signals(self):
        """Test basic RRF with two signals."""
        fusion = SignalFusion()
        signals = {
            "bm25": [{"id": "doc1", "score": 0.9}, {"id": "doc2", "score": 0.8}],
            "dense": [{"id": "doc2", "score": 0.85}, {"id": "doc3", "score": 0.7}],
        }
        results = fusion.fuse(signals)
        assert len(results) == 3
        # Check all have fused_score
        for doc in results:
            assert "fused_score" in doc
        # Results should be sorted descending
        assert results[0]["fused_score"] >= results[1]["fused_score"]

    def test_three_signals_different_weights(self):
        """Test three signals with custom weights."""
        weights = {"bm25": 0.3, "dense": 0.5, "graph": 0.2}
        fusion = SignalFusion(weights=weights)
        signals = {
            "bm25": [{"id": "doc1"}],
            "dense": [{"id": "doc1"}],
            "graph": [{"id": "doc1"}],
        }
        results = fusion.fuse(signals)
        assert len(results) == 1
        doc = results[0]
        # Expected: sum(w/(k+1)) for each signal + boost (2 signals extra)
        expected_base = (0.3 / (RRF_K + 1)) + (0.5 / (RRF_K + 1)) + (0.2 / (RRF_K + 1))
        expected_boost = 2 * fusion.boost_factor  # 3 signals, so (3-1)*0.1=0.2
        expected = expected_base + expected_boost
        assert abs(doc["fused_score"] - expected) < 1e-9

    def test_cross_signal_boost(self):
        """Test cross-signal boosting for multiple signals."""
        fusion = SignalFusion(boost_factor=0.1)
        # Doc in 2 signals: boost 0.1
        signals_2 = {
            "bm25": [{"id": "doc1"}],
            "dense": [{"id": "doc1"}],
        }
        results_2 = fusion.fuse(signals_2)
        assert len(results_2) == 1
        score_2 = results_2[0]["fused_score"]
        expected_2 = (1.0 / (RRF_K + 1)) * 2 + 0.1  # 2 signals, boost 0.1
        assert abs(score_2 - expected_2) < 1e-9

        # Doc in 3 signals: boost 0.2
        signals_3 = {
            "bm25": [{"id": "doc1"}],
            "dense": [{"id": "doc1"}],
            "graph": [{"id": "doc1"}],
        }
        results_3 = fusion.fuse(signals_3)
        score_3 = results_3[0]["fused_score"]
        expected_3 = (1.0 / (RRF_K + 1)) * 3 + 0.2  # 3 signals, boost 0.2
        assert abs(score_3 - expected_3) < 1e-9

    def test_doc_in_multiple_signals_merge_info(self):
        """Test that doc info is merged across signals."""
        fusion = SignalFusion()
        signals = {
            "bm25": [{"id": "doc1", "bm25_score": 0.9, "extra": "bm25"}],
            "dense": [{"id": "doc1", "dense_score": 0.85, "extra": "dense"}],
        }
        results = fusion.fuse(signals)
        assert len(results) == 1
        doc = results[0]
        assert doc["bm25_score"] == 0.9
        assert doc["dense_score"] == 0.85
        # Later signal overwrites earlier for same key
        assert doc["extra"] == "dense"

    def test_empty_signals(self):
        """Test empty signals returns empty list."""
        fusion = SignalFusion()
        signals = {}
        results = fusion.fuse(signals)
        assert results == []

    def test_single_signal(self):
        """Test single signal with no boost."""
        fusion = SignalFusion()
        signals = {"bm25": [{"id": "doc1"}, {"id": "doc2"}]}
        results = fusion.fuse(signals)
        assert len(results) == 2
        # No boost since only 1 signal
        for doc in results:
            rank = 1 if doc["id"] == "doc1" else 2
            expected = 1.0 / (RRF_K + rank)
            assert abs(doc["fused_score"] - expected) < 1e-9

    def test_get_final_ranking_top_k(self):
        """Test get_final_ranking with top_k truncation."""
        fusion = SignalFusion()
        fused_results = [
            {"id": "doc1", "fused_score": 0.5},
            {"id": "doc2", "fused_score": 0.3},
            {"id": "doc3", "fused_score": 0.4},
        ]
        # Unsorted input, should sort first
        top_2 = fusion.get_final_ranking(fused_results, top_k=2)
        assert len(top_2) == 2
        assert top_2[0]["id"] == "doc1"
        assert top_2[1]["id"] == "doc3"
