"""Hybrid retrieval fusion using Reciprocal Rank Fusion (RRF)."""

from collections import defaultdict
from typing import List, Tuple, Optional


def rrf_fusion(
    result_lists: List[List[Tuple[int, float]]],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """Reciprocal Rank Fusion (RRF) for combining search results.
    
    Combines multiple ranked retrieval results using the formula:
    score(d) = Σ 1 / (k + rank(d))
    
    Args:
        result_lists: List of ranked result lists. Each list contains
                     (document_id, score) tuples sorted by rank.
        k: RRF parameter (default 60). Higher values give more weight
           to lower-ranked results.
    
    Returns:
        Combined ranked list of (document_id, rrf_score) tuples.
    
    Example:
        >>> bm25_results = [(0, 10.5), (1, 8.2), (2, 5.0)]
        >>> dense_results = [(2, 0.95), (0, 0.88), (3, 0.75)]
        >>> combined = rrf_fusion([bm25_results, dense_results])
    """
    if not result_lists:
        return []
    
    rrf_scores = defaultdict(float)
    
    for result_list in result_lists:
        for rank, (doc_id, _) in enumerate(result_list, 1):
            rrf_scores[doc_id] += 1 / (k + rank)
    
    # Sort by RRF score (descending)
    sorted_results = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return sorted_results


def convex_fusion(
    result_lists: List[List[Tuple[int, float]]],
    weights: Optional[List[float]] = None,
) -> List[Tuple[int, float]]:
    """Convex combination fusion.
    
    Combines scores using weighted linear combination.
    Requires normalized scores across result lists.
    
    Args:
        result_lists: List of ranked result lists
        weights: Optional weights for each list (default: equal weights)
    
    Returns:
        Combined ranked list of (document_id, fused_score) tuples
    """
    if not result_lists:
        return []
    
    if weights is None:
        weights = [1.0 / len(result_lists)] * len(result_lists)
    else:
        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
    
    # Combine scores
    combined_scores = defaultdict(float)
    
    for result_list, weight in zip(result_lists, weights):
        for doc_id, score in result_list:
            combined_scores[doc_id] += weight * score
    
    # Sort by combined score
    sorted_results = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    return sorted_results


def score_normalize(
    result_list: List[Tuple[int, float]],
    method: str = "minmax",
) -> List[Tuple[int, float]]:
    """Normalize scores to [0, 1] range.
    
    Args:
        result_list: List of (doc_id, score) tuples
        method: Normalization method ('minmax', 'zscore', 'rank')
    
    Returns:
        List of (doc_id, normalized_score) tuples
    """
    if not result_list:
        return []
    
    doc_ids = [doc_id for doc_id, _ in result_list]
    scores = [score for _, score in result_list]
    
    if method == "minmax":
        min_s = min(scores)
        max_s = max(scores)
        if max_s == min_s:
            return [(doc_ids[i], 1.0) for i in range(len(doc_ids))]
        normalized = [(s - min_s) / (max_s - min_s) for s in scores]
    elif method == "zscore":
        import numpy as np
        mean_s = np.mean(scores)
        std_s = np.std(scores)
        if std_s == 0:
            return [(doc_ids[i], 0.0) for i in range(len(doc_ids))]
        normalized = [(s - mean_s) / std_s for s in scores]
    elif method == "rank":
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        normalized = [0.0] * len(scores)
        for rank, idx in enumerate(sorted_indices, 1):
            normalized[idx] = 1.0 - (rank - 1) / len(scores)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return list(zip(doc_ids, normalized))


def hybrid_search(
    bm25_results: List[Tuple[int, float]],
    dense_results: List[Tuple[int, float]],
    k: int = 60,
    normalize: bool = True,
) -> List[Tuple[int, float]]:
    """Combine BM25 and dense search results using RRF.
    
    Args:
        bm25_results: BM25 search results as (doc_id, score) tuples
        dense_results: Dense search results as (doc_id, score) tuples
        k: RRF parameter (default 60)
        normalize: Whether to normalize scores before fusion
    
    Returns:
        Combined ranked list
    """
    if normalize:
        bm25_norm = score_normalize(bm25_results, method="rank")
        dense_norm = score_normalize(dense_results, method="rank")
        return rrf_fusion([bm25_norm, dense_norm], k=k)
    else:
        return rrf_fusion([bm25_results, dense_results], k=k)


# Re-export for convenience
__all__ = [
    "rrf_fusion",
    "convex_fusion",
    "score_normalize",
    "hybrid_search",
]