"""Signal fusion module using Weighted Reciprocal Rank Fusion.

Provides:
- Weighted Reciprocal Rank Fusion (WRRF)
- Cross-signal boosting for documents in multiple signals
- Configurable signal weights (BM25, dense, graph)
- Final ranked output after fusion
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Default RRF constant
RRF_K = 60


class SignalFusion:
    """Weighted Reciprocal Rank Fusion for combining retrieval signals."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        rrf_k: int = RRF_K,
        boost_factor: float = 0.1,
    ):
        """Initialize signal fusion.

        Args:
            weights: Dict mapping signal names to weights (e.g., {"bm25": 0.3, "dense": 0.5})
            rrf_k: RRF constant (default 60)
            boost_factor: Boost per additional signal (default 0.1)
        """
        self.weights = weights or {}
        self.rrf_k = rrf_k
        self.boost_factor = boost_factor

    def fuse(
        self,
        signals: dict[str, list[dict]],
    ) -> list[dict]:
        """Fuse multiple ranked lists using Weighted RRF.

        Args:
            signals: Dict mapping signal names to ranked lists of docs with 'id' key

        Returns:
            Single ranked list after fusion, with 'fused_score' added
        """
        per_doc_data: dict[str, dict] = {}

        # First pass: collect per-document info and per-signal scores
        for signal_name, docs in signals.items():
            for rank, doc in enumerate(docs, start=1):  # rank is 1-indexed
                doc_id = doc["id"]
                if doc_id not in per_doc_data:
                    per_doc_data[doc_id] = {
                        "doc_info": {},
                        "per_signal_scores": {},
                        "signal_count": 0,
                    }
                # Merge document info from all signals
                per_doc_data[doc_id]["doc_info"].update(doc)
                # Compute RRF contribution for this signal
                rrf_score = self._compute_rrf_score(signal_name, rank)
                per_doc_data[doc_id]["per_signal_scores"][signal_name] = rrf_score
                per_doc_data[doc_id]["signal_count"] += 1

        # Second pass: compute fused scores with cross-signal boosting
        fused_results = []
        for doc_id, data in per_doc_data.items():
            base_score = sum(data["per_signal_scores"].values())
            boosted_score = self._apply_cross_signal_boost(
                doc_id, data["per_signal_scores"], base_score
            )
            result_doc = data["doc_info"].copy()
            result_doc["fused_score"] = boosted_score
            fused_results.append(result_doc)

        # Sort by fused_score descending
        fused_results.sort(key=lambda x: x["fused_score"], reverse=True)
        return fused_results

    def _compute_rrf_score(
        self,
        signal_name: str,
        rank: int,
    ) -> float:
        """Compute RRF score for a given rank.

        Args:
            signal_name: Name of the signal (for weight lookup)
            rank: Rank position (1-indexed)

        Returns:
            Weighted RRF score
        """
        weight = self.weights.get(signal_name, 1.0)
        return weight / (self.rrf_k + rank)

    def _apply_cross_signal_boost(
        self,
        doc_id: str,
        scores: dict[str, float],
        fused_score: float,
    ) -> float:
        """Apply boost for documents appearing in multiple signals.

        Args:
            doc_id: Document ID
            scores: Per-signal scores for this document
            fused_score: Current fused score

        Returns:
            Boosted score
        """
        num_signals = len(scores)
        if num_signals >= 2:
            boost = (num_signals - 1) * self.boost_factor
            return fused_score + boost
        return fused_score

    def get_final_ranking(
        self,
        fused_results: list[dict],
        top_k: int | None = None,
    ) -> list[dict]:
        """Get final ranked output after fusion.

        Args:
            fused_results: Results with 'fused_score'
            top_k: Number of top results to return

        Returns:
            Sorted and optionally truncated results
        """
        # Sort by fused_score descending (ensure order)
        sorted_results = sorted(fused_results, key=lambda x: x["fused_score"], reverse=True)
        if top_k is not None:
            return sorted_results[:top_k]
        return sorted_results
