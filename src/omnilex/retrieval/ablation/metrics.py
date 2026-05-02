"""Component-level metrics tracking for ablation framework.

Tracks retrieval recall@K, reranker accuracy, verifier precision/recall.
"""

from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Tracks metrics at each pipeline stage."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.stage_metrics: dict[str, dict] = defaultdict(dict)
        self.query_metrics: list[dict] = []

    def track_retrieval(
        self,
        query_id: str,
        retrieved_ids: list[str],
        gold_ids: list[str],
        k_values: list[int] | None = None,
    ) -> dict[str, float]:
        """Track retrieval recall@K.

        Args:
            query_id: Query identifier
            retrieved_ids: Retrieved document IDs
            gold_ids: Gold document IDs
            k_values: K values to compute recall for

        Returns:
            Dict mapping f"recall@{k}" to value
        """
        k_values = k_values or [5, 10, 20, 50]
        gold_set = set(gold_ids)
        metrics = {}

        for k in k_values:
            top_k = retrieved_ids[:k]
            hits = len(gold_set.intersection(top_k))
            recall = hits / len(gold_set) if gold_set else 0.0
            metrics[f"recall@{k}"] = recall

        self.query_metrics.append({"query_id": query_id, "stage": "retrieval", "metrics": metrics})
        return metrics

    def track_reranker(
        self,
        query_id: str,
        reranked_ids: list[str],
        gold_ids: list[str],
        k_values: list[int] | None = None,
    ) -> dict[str, float]:
        """Track reranker accuracy.

        Args:
            query_id: Query identifier
            reranked_ids: Reranked document IDs
            gold_ids: Gold document IDs
            k_values: K values to compute metrics for

        Returns:
            Dict with reranker metrics
        """
        k_values = k_values or [5, 10, 20, 50]
        gold_set = set(gold_ids)
        metrics = {}

        for k in k_values:
            top_k = reranked_ids[:k]
            hits = len(gold_set.intersection(top_k))
            recall = hits / len(gold_set) if gold_set else 0.0
            metrics[f"recall@{k}"] = recall

        # Add average accuracy
        metrics["accuracy"] = sum(metrics.values()) / len(metrics) if metrics else 0.0

        self.query_metrics.append({"query_id": query_id, "stage": "reranker", "metrics": metrics})
        return metrics

    def track_verifier(
        self,
        query_id: str,
        verified_ids: list[str],
        gold_ids: list[str],
    ) -> dict[str, float]:
        """Track verifier precision/recall.

        Args:
            query_id: Query identifier
            verified_ids: Verified document IDs
            gold_ids: Gold document IDs

        Returns:
            Dict with precision, recall, f1
        """
        gold_set = set(gold_ids)
        verified_set = set(verified_ids)

        tp = len(gold_set.intersection(verified_set))
        precision = tp / len(verified_set) if verified_set else 0.0
        recall = tp / len(gold_set) if gold_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics = {"precision": precision, "recall": recall, "f1": f1}

        self.query_metrics.append({"query_id": query_id, "stage": "verifier", "metrics": metrics})
        return metrics

    def get_aggregate_metrics(self) -> dict[str, float]:
        """Get aggregate metrics across all queries.

        Returns:
            Dict with aggregate metrics
        """
        # Group metrics by stage and metric name
        stage_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

        for qm in self.query_metrics:
            stage = qm["stage"]
            for metric_name, value in qm["metrics"].items():
                stage_metrics[stage][metric_name].append(value)

        # Compute averages
        result = {}
        for stage, metrics in stage_metrics.items():
            for metric_name, values in metrics.items():
                avg = sum(values) / len(values) if values else 0.0
                result[f"{stage}_{metric_name}"] = avg

        return result

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.stage_metrics.clear()
        self.query_metrics.clear()
