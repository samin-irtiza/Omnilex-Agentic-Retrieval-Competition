"""Tests for ablation framework components."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional

import pytest

from src.omnilex.retrieval.ablation.config import (
    ExperimentConfig,
    EXPERIMENT_PRESETS,
)
from src.omnilex.retrieval.ablation.metrics import MetricsTracker
from src.omnilex.retrieval.ablation.reporter import ResultsReporter
from src.omnilex.retrieval.ablation.runner import ExperimentRunner


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
        import csv

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

        # Run (will use empty results since no actual indices)
        results = runner.run(queries, ground_truth)
        assert "results" in results
        assert "metrics" in results
