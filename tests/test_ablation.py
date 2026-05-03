"""Tests for ablation framework components."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.omnilex.retrieval.ablation.config import (
    ExperimentConfig,
)
from src.omnilex.retrieval.ablation.metrics import MetricsTracker
from src.omnilex.retrieval.ablation.reporter import ResultsReporter
from src.omnilex.retrieval.ablation.runner import ExperimentRunner
from src.omnilex.retrieval.bm25_index import load_corpus_from_csv

# ---- Config Tests ----


class TestExperimentConfig:
    def test_from_preset_valid(self):
        config = ExperimentConfig.from_preset("exp_baseline")
        assert config.name == "exp_baseline"
        assert config.components["bm25"] is True
        assert config.components["dense"] is False

    def test_from_preset_invalid(self):
        with pytest.raises(ValueError):
            ExperimentConfig.from_preset("invalid_preset")

    def test_validate_valid(self):
        config = ExperimentConfig.from_preset("exp_baseline")
        errors = config.validate()
        assert len(errors) == 0

    def test_validate_invalid_component(self):
        config = ExperimentConfig(name="test", components={"invalid": True})
        errors = config.validate()
        assert any("Unknown component" in e for e in errors)

    def test_validate_invalid_threshold(self):
        config = ExperimentConfig(name="test", verifier_threshold=1.5)
        errors = config.validate()
        assert any("verifier_threshold" in e for e in errors)

    def test_from_yaml(self, tmp_path: Path):
        config_data = {
            "name": "test_exp",
            "description": "Test experiment",
            "components": {"bm25": True, "dense": False},
            "top_k": 100,
        }
        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, "w") as f:
            import yaml

            yaml.dump(config_data, f)

        config = ExperimentConfig.from_yaml(yaml_path)
        assert config.name == "test_exp"
        assert config.top_k == 100

    def test_to_dict(self):
        config = ExperimentConfig.from_preset("exp_dense_only")
        config_dict = config.to_dict()
        assert config_dict["name"] == "exp_dense_only"
        assert "components" in config_dict


# ---- Metrics Tests ----


class TestMetricsTracker:
    def test_track_retrieval(self):
        tracker = MetricsTracker()
        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        gold = ["doc1", "doc3", "doc5"]

        metrics = tracker.track_retrieval("q1", retrieved, gold)
        assert "recall@5" in metrics
        assert abs(metrics["recall@5"] - 1.0) < 0.01

    def test_track_retrieval_recall_at_k(self):
        tracker = MetricsTracker()
        retrieved = ["doc1", "doc2", "doc3"]
        gold = ["doc1", "doc4"]

        metrics = tracker.track_retrieval("q1", retrieved, gold)
        assert metrics["recall@5"] == 0.5  # 1 hit out of 2 gold
        assert metrics["recall@10"] == 0.5

    def test_track_reranker(self):
        tracker = MetricsTracker()
        reranked = ["doc1", "doc2", "doc3"]
        gold = ["doc1", "doc3"]

        metrics = tracker.track_reranker("q1", reranked, gold)
        assert "recall@5" in metrics
        assert "accuracy" in metrics

    def test_track_verifier(self):
        tracker = MetricsTracker()
        verified = ["doc1", "doc2"]
        gold = ["doc1", "doc3"]

        metrics = tracker.track_verifier("q1", verified, gold)
        assert abs(metrics["precision"] - 0.5) < 0.01
        assert abs(metrics["recall"] - 0.5) < 0.01
        assert abs(metrics["f1"] - 0.5) < 0.01

    def test_get_aggregate_metrics(self):
        tracker = MetricsTracker()
        # Add some query metrics
        tracker.track_retrieval("q1", ["d1", "d2"], ["d1"])
        tracker.track_retrieval("q2", ["d1", "d3"], ["d1", "d3"])
        tracker.track_verifier("q1", ["d1"], ["d1"])

        agg = tracker.get_aggregate_metrics()
        assert "retrieval_recall@5" in agg
        assert "verifier_f1" in agg

    def test_reset(self):
        tracker = MetricsTracker()
        tracker.track_retrieval("q1", ["d1"], ["d1"])
        tracker.reset()
        assert len(tracker.query_metrics) == 0


# ---- Reporter Tests ----


class TestResultsReporter:
    def test_add_result(self):
        reporter = ResultsReporter()
        reporter.add_result("exp1", {"name": "exp1"}, {"retrieval_recall@5": 0.8})
        assert len(reporter.results) == 1

    def test_generate_comparison_table(self):
        reporter = ResultsReporter()
        reporter.add_result("exp1", {"name": "exp1"}, {"retrieval_recall@5": 0.8})
        reporter.add_result("exp2", {"name": "exp2"}, {"retrieval_recall@5": 0.9})

        table = reporter.generate_comparison_table()
        assert "exp1" in table
        assert "exp2" in table

    def test_export_csv(self, tmp_path: Path):
        reporter = ResultsReporter(tmp_path)
        reporter.add_result("exp1", {"name": "exp1"}, {"retrieval_recall@5": 0.8})
        path = reporter.export_csv()
        assert path.exists()

        # Verify CSV content

        with open(path) as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert rows[0] == ["experiment_name", "retrieval_recall@5"]
            assert rows[1][0] == "exp1"

    def test_export_json(self, tmp_path: Path):
        reporter = ResultsReporter(tmp_path)
        reporter.add_result("exp1", {"name": "exp1"}, {"retrieval_recall@5": 0.8})
        path = reporter.export_json()
        assert path.exists()

        with open(path) as f:
            data = json.load(f)
            assert len(data) == 1
            assert data[0]["experiment_name"] == "exp1"

    def test_print_summary(self, capsys):
        reporter = ResultsReporter()
        reporter.add_result("exp1", {"name": "exp1"}, {"retrieval_recall@5": 0.8})
        reporter.print_summary()
        captured = capsys.readouterr()
        assert "exp1" in captured.out


# ---- Runner Tests ----


class TestExperimentRunner:
    def test_init(self):
        config = ExperimentConfig.from_preset("exp_baseline")
        runner = ExperimentRunner(config)
        assert runner.config.name == "exp_baseline"

    def test_run_mock(self, tmp_path: Path):
        """Test runner with mock queries (no actual retrieval)."""
        from unittest.mock import patch

        config = ExperimentConfig.from_preset("exp_baseline")
        runner = ExperimentRunner(config, output_dir=tmp_path)

        # Mock queries
        queries = [
            {"id": "q1", "query": "test query 1"},
            {"id": "q2", "query": "test query 2"},
        ]

        # Mock ground truth
        ground_truth = {
            "q1": [{"id": "doc1"}, {"id": "doc2"}],
            "q2": [{"id": "doc3"}],
        }

        # Mock _run_retrieval to return empty signals (no actual indices needed)
        with patch.object(runner, "_run_retrieval", return_value={}):
            results = runner.run(queries, ground_truth)
            assert "results" in results
            assert "metrics" in results

    def test_normalize_gold_ids_strings(self):
        """Test _normalize_gold_ids with list of strings."""
        config = ExperimentConfig.from_preset("exp_baseline")
        runner = ExperimentRunner(config)

        gold_docs = ["SR 123.1 Art. 5", "BGE 123 II 456"]
        result = runner._normalize_gold_ids(gold_docs)
        assert result == ["SR 123.1 Art. 5", "BGE 123 II 456"]

    def test_normalize_gold_ids_dicts_with_id(self):
        """Test _normalize_gold_ids with list of dicts with 'id' key."""
        config = ExperimentConfig.from_preset("exp_baseline")
        runner = ExperimentRunner(config)

        gold_docs = [{"id": "SR 123.1 Art. 5"}, {"id": "BGE 123 II 456"}]
        result = runner._normalize_gold_ids(gold_docs)
        assert result == ["SR 123.1 Art. 5", "BGE 123 II 456"]

    def test_normalize_gold_ids_dicts_with_citation(self):
        """Test _normalize_gold_ids with list of dicts with 'citation' key."""
        config = ExperimentConfig.from_preset("exp_baseline")
        runner = ExperimentRunner(config)

        gold_docs = [{"citation": "SR 123.1 Art. 5"}]
        result = runner._normalize_gold_ids(gold_docs)
        assert result == ["SR 123.1 Art. 5"]

    def test_normalize_gold_ids_mixed_format(self):
        """Test _normalize_gold_ids with mixed formats in same list."""
        config = ExperimentConfig.from_preset("exp_baseline")
        runner = ExperimentRunner(config)

        gold_docs = ["SR 123.1 Art. 5", {"id": "BGE 123 II 456"}]
        result = runner._normalize_gold_ids(gold_docs)
        assert result == ["SR 123.1 Art. 5", "BGE 123 II 456"]

    def test_normalize_gold_ids_missing_keys(self, caplog):
        """Test _normalize_gold_ids with dict missing 'id' and 'citation' keys."""
        import logging

        config = ExperimentConfig.from_preset("exp_baseline")
        runner = ExperimentRunner(config, verbose=True)  # Enable INFO logging to capture warnings

        # Enable propagation so caplog can capture messages
        # Logger name is "src.omnilex.retrieval.ablation.runner" (module __name__)
        logging.getLogger("src.omnilex.retrieval.ablation.runner").propagate = True

        gold_docs = [{"foo": "bar"}]
        result = runner._normalize_gold_ids(gold_docs)
        assert result == [""]
        assert "neither 'id' nor 'citation' key" in caplog.text

    def test_run_with_string_ground_truth(self, tmp_path: Path):
        """Test run() with ground_truth as list of strings."""
        config = ExperimentConfig(
            name="test",
            components={
                "bm25": False,
                "dense": False,
                "graph": False,
                "reranker": False,
                "verifier": False,
                "rrf_fusion": False,
            },
        )
        runner = ExperimentRunner(config, output_dir=tmp_path)

        queries = [{"id": "q1", "query": "test query"}]
        ground_truth = {
            "q1": ["SR 123.1 Art. 5", "BGE 123 II 456"],
        }

        results = runner.run(queries, ground_truth)
        assert "results" in results
        assert "metrics" in results

    def test_run_with_mixed_ground_truth(self, tmp_path: Path):
        """Test run() with ground_truth in mixed format."""
        config = ExperimentConfig(
            name="test",
            components={
                "bm25": False,
                "dense": False,
                "graph": False,
                "reranker": False,
                "verifier": False,
                "rrf_fusion": False,
            },
        )
        runner = ExperimentRunner(config, output_dir=tmp_path)

        queries = [{"id": "q1", "query": "test query"}]
        ground_truth = {
            "q1": ["SR 123.1 Art. 5", {"id": "BGE 123 II 456"}],
        }

        results = runner.run(queries, ground_truth)
        assert "results" in results
        assert "metrics" in results

    def test_run_with_citation_key_ground_truth(self, tmp_path: Path):
        """Test run() with ground_truth using 'citation' key in dicts."""
        config = ExperimentConfig(
            name="test",
            components={
                "bm25": False,
                "dense": False,
                "graph": False,
                "reranker": False,
                "verifier": False,
                "rrf_fusion": False,
            },
        )
        runner = ExperimentRunner(config, output_dir=tmp_path)

        queries = [{"id": "q1", "query": "test query"}]
        ground_truth = {
            "q1": [{"citation": "SR 123.1 Art. 5"}],
        }

        results = runner.run(queries, ground_truth)
        assert "results" in results
        assert "metrics" in results


# ---- BM25 CSV Loading Tests ----


class TestLoadCorpusFromCsv:
    def test_load_basic_csv(self, tmp_path: Path):
        """Test loading a basic CSV with citation and text columns."""
        csv_path = tmp_path / "test_corpus.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("citation,text\n")
            f.write("SR 123.1 Art. 5,This is the text for article 5\n")
            f.write("BGE 123 II 456,This is a court decision\n")

        documents = load_corpus_from_csv(csv_path)
        assert len(documents) == 2
        assert documents[0]["citation"] == "SR 123.1 Art. 5"
        assert documents[0]["text"] == "This is the text for article 5"
        assert documents[1]["citation"] == "BGE 123 II 456"

    def test_load_with_max_rows(self, tmp_path: Path):
        """Test max_rows parameter limits loaded rows."""
        csv_path = tmp_path / "test_corpus.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("citation,text\n")
            f.write("SR 123.1 Art. 5,Text 1\n")
            f.write("SR 123.1 Art. 6,Text 2\n")
            f.write("SR 123.1 Art. 7,Text 3\n")

        documents = load_corpus_from_csv(csv_path, max_rows=2)
        assert len(documents) == 2
        assert documents[1]["citation"] == "SR 123.1 Art. 6"

    def test_load_with_custom_columns(self, tmp_path: Path):
        """Test loading with custom column names."""
        csv_path = tmp_path / "test_corpus.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("doc_id,content\n")
            f.write("SR 123.1 Art. 5,Some content here\n")

        documents = load_corpus_from_csv(csv_path, citation_col="doc_id", text_col="content")
        assert len(documents) == 1
        assert documents[0]["citation"] == "SR 123.1 Art. 5"
        assert documents[0]["text"] == "Some content here"

    def test_load_missing_columns(self, tmp_path: Path):
        """Test handling of missing columns (should return empty string)."""
        csv_path = tmp_path / "test_corpus.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("citation,other\n")
            f.write("SR 123.1 Art. 5,some other data\n")

        documents = load_corpus_from_csv(csv_path, text_col="nonexistent")
        assert len(documents) == 1
        assert documents[0]["citation"] == "SR 123.1 Art. 5"
        assert documents[0]["text"] == ""  # Missing column returns empty string

    def test_load_empty_csv(self, tmp_path: Path):
        """Test loading an empty CSV (only headers)."""
        csv_path = tmp_path / "test_corpus.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("citation,text\n")

        documents = load_corpus_from_csv(csv_path)
        assert len(documents) == 0

    def test_load_nonexistent_file(self):
        """Test that loading a nonexistent file raises an error."""
        with pytest.raises(FileNotFoundError):
            load_corpus_from_csv("/nonexistent/path.csv")


# ---- Runner BM25 Index Initialization Tests ----


class TestRunnerBM25Init:
    """Tests for ExperimentRunner BM25 index initialization."""

    def test_init_laws_index_fast_path(self, tmp_path: Path):
        """Test loading pre-built index from cache (fast path)."""
        # Create a test CSV
        csv_path = tmp_path / "laws_de.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("citation,text\n")
            f.write("SR 123.1 Art. 5,Law text 1\n")
            f.write("SR 123.1 Art. 6,Law text 2\n")

        # Create config with paths
        config = ExperimentConfig(
            name="test",
            components={
                "bm25": True,
                "dense": False,
                "graph": False,
                "reranker": False,
                "verifier": False,
                "rrf_fusion": False,
            },
            laws_corpus_path=str(csv_path),
            index_cache_dir=str(tmp_path / "cache"),
        )
        runner = ExperimentRunner(config, output_dir=tmp_path)

        # First call should build and cache
        index = runner._init_bm25_index("laws")
        assert index is not None
        assert len(index.documents) == 2

        # Verify cache was created
        cache_path = tmp_path / "cache" / "bm25_laws.pkl"
        assert cache_path.exists()

        # Second call should load from cache (fast path)
        # Create a new runner to test loading from cache
        runner2 = ExperimentRunner(config, output_dir=tmp_path)
        index2 = runner2._init_bm25_index("laws")
        assert index2 is not None
        assert len(index2.documents) == 2

    def test_init_laws_index_slow_path(self, tmp_path: Path):
        """Test building index from CSV when no cache exists (slow path)."""
        # Create a test CSV
        csv_path = tmp_path / "laws_de.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("citation,text\n")
            f.write("SR 123.1 Art. 5,Law text 1\n")
            f.write("SR 123.1 Art. 6,Law text 2\n")

        # Create config without cache dir (slow path only)
        config = ExperimentConfig(
            name="test",
            components={
                "bm25": True,
                "dense": False,
                "graph": False,
                "reranker": False,
                "verifier": False,
                "rrf_fusion": False,
            },
            laws_corpus_path=str(csv_path),
        )
        runner = ExperimentRunner(config, output_dir=tmp_path)

        # Should build from CSV
        index = runner._init_bm25_index("laws")
        assert index is not None
        assert len(index.documents) == 2

        # Test search works
        results = index.search("Law text", top_k=10)
        assert len(results) > 0

    def test_init_courts_index(self, tmp_path: Path):
        """Test initializing courts BM25 index."""
        # Create a test CSV for courts
        csv_path = tmp_path / "court_considerations.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("citation,text\n")
            f.write("BGE 123 II 456,Court text 1\n")
            f.write("BGE 124 II 789,Court text 2\n")

        config = ExperimentConfig(
            name="test",
            components={
                "bm25": False,
                "dense": False,
                "graph": False,
                "reranker": False,
                "verifier": False,
                "rrf_fusion": False,
            },
            courts_corpus_path=str(csv_path),
            index_cache_dir=str(tmp_path / "cache"),
        )
        runner = ExperimentRunner(config, output_dir=tmp_path)

        # Initialize courts index
        index = runner._init_bm25_index("courts")
        assert index is not None
        assert len(index.documents) == 2

        # Test search works
        results = index.search("Court text", top_k=10)
        assert len(results) > 0

    def test_init_both_indices(self, tmp_path: Path):
        """Test that both laws and courts indices can be initialized independently."""
        # Create test CSVs
        laws_csv = tmp_path / "laws_de.csv"
        with open(laws_csv, "w", encoding="utf-8") as f:
            f.write("citation,text\n")
            f.write("SR 123.1 Art. 5,Law text\n")

        courts_csv = tmp_path / "court_considerations.csv"
        with open(courts_csv, "w", encoding="utf-8") as f:
            f.write("citation,text\n")
            f.write("BGE 123 II 456,Court text\n")

        config = ExperimentConfig(
            name="test",
            components={
                "bm25": True,
                "dense": False,
                "graph": False,
                "reranker": False,
                "verifier": False,
                "rrf_fusion": False,
            },
            laws_corpus_path=str(laws_csv),
            courts_corpus_path=str(courts_csv),
            index_cache_dir=str(tmp_path / "cache"),
        )
        runner = ExperimentRunner(config, output_dir=tmp_path)

        # Initialize both indices
        laws_index = runner._init_bm25_index("laws")
        courts_index = runner._init_bm25_index("courts")

        assert laws_index is not None
        assert courts_index is not None
        assert len(laws_index.documents) == 1
        assert len(courts_index.documents) == 1

    def test_missing_corpus_path_raises_error(self, tmp_path: Path):
        """Test that missing corpus path raises a clear error."""
        config = ExperimentConfig(
            name="test",
            components={
                "bm25": True,
                "dense": False,
                "graph": False,
                "reranker": False,
                "verifier": False,
                "rrf_fusion": False,
            },
            # No laws_corpus_path set
        )
        runner = ExperimentRunner(config, output_dir=tmp_path)

        with pytest.raises(ValueError) as exc_info:
            runner._init_bm25_index("laws")
        assert "laws_corpus_path is not configured" in str(exc_info.value)

    def test_nonexistent_corpus_file_raises_error(self, tmp_path: Path):
        """Test that nonexistent corpus file raises an error."""
        config = ExperimentConfig(
            name="test",
            components={
                "bm25": True,
                "dense": False,
                "graph": False,
                "reranker": False,
                "verifier": False,
                "rrf_fusion": False,
            },
            laws_corpus_path=str(tmp_path / "nonexistent.csv"),
        )
        runner = ExperimentRunner(config, output_dir=tmp_path)

        with pytest.raises(FileNotFoundError) as exc_info:
            runner._init_bm25_index("laws")
        assert "not found" in str(exc_info.value).lower()

    def test_run_retrieval_with_bm25(self, tmp_path: Path):
        """Test that _run_retrieval works with BM25 initialization."""
        # Create a test CSV
        csv_path = tmp_path / "laws_de.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("citation,text\n")
            f.write("SR 123.1 Art. 5,Law text about something\n")

        config = ExperimentConfig(
            name="test",
            components={
                "bm25": True,
                "dense": False,
                "graph": False,
                "reranker": False,
                "verifier": False,
                "rrf_fusion": False,
            },
            laws_corpus_path=str(csv_path),
            index_cache_dir=str(tmp_path / "cache"),
        )
        runner = ExperimentRunner(config, output_dir=tmp_path)

        # Run retrieval
        signals = runner._run_retrieval("something", "q1")
        assert "bm25" in signals
        assert len(signals["bm25"]) > 0
