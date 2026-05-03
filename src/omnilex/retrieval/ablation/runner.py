"""Experiment runner for ablation framework.

Executes pipeline with config toggles and tracks results.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from omnilex.citations.normalizer import CitationNormalizer
from omnilex.retrieval.bm25_index import BM25Index
from omnilex.retrieval.dense_index import DenseIndex
from omnilex.retrieval.graph_index import CitationGraph

from .config import ExperimentConfig
from .metrics import MetricsTracker
from .reporter import ResultsReporter

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Runs ablation experiments with different component configurations."""

    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: str | Path = "experiments",
    ):
        """Initialize experiment runner.

        Args:
            config: Experiment configuration
            output_dir: Directory for experiment outputs
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.metrics = MetricsTracker()
        self.reporter = ResultsReporter(self.output_dir)

        # Pipeline components (lazily initialized)
        self._bm25_index = None
        self._bm25_courts_index = None  # Separate index for courts corpus
        self._bm25_non_leading_index = None  # Separate index for non-leading decisions
        self._dense_index = None
        self._graph_index = None
        self._reranker = None
        self._verifier = None
        self._fusion = None

    def _normalize_gold_ids(self, gold_docs: list) -> list[str]:
        """Normalize ground truth documents to list of citation strings.

        Handles multiple formats:
        - List of strings: ["SR 123.1 Art. 5"]
        - List of dicts with "id" key: [{"id": "SR 123.1 Art. 5"}]
        - List of dicts with "citation" key: [{"citation": "SR 123.1 Art. 5"}]

        Args:
            gold_docs: List of ground truth documents in various formats

        Returns:
            List of citation strings
        """
        gold_ids = []
        for doc in gold_docs:
            if isinstance(doc, str):
                gold_ids.append(doc)
            elif isinstance(doc, dict):
                # Try "id" key first, then "citation" as fallback
                doc_id = doc.get("id")
                if doc_id is None:
                    doc_id = doc.get("citation")
                if doc_id is None:
                    logger.warning(f"Ground truth doc has neither 'id' nor 'citation' key: {doc}")
                    doc_id = ""
                gold_ids.append(doc_id)
            else:
                logger.warning(f"Unexpected ground truth format: {type(doc)} - {doc}")
                gold_ids.append(str(doc))
        return gold_ids

    def _init_bm25_index(self, corpus_type: str) -> BM25Index:
        """Initialize BM25 index for a given corpus type.

        Checks for pre-built index cache first, then builds from CSV/JSONL if needed.

        Args:
            corpus_type: One of "laws", "courts", or "non_leading"

        Returns:
            Initialized BM25Index

        Raises:
            ValueError: If corpus paths are not configured or file not found
        """
        # Determine paths based on corpus type
        if corpus_type == "laws":
            corpus_path = self.config.laws_corpus_path
        elif corpus_type == "courts":
            corpus_path = self.config.courts_corpus_path
        elif corpus_type == "non_leading":
            corpus_path = self.config.non_leading_corpus_path
        else:
            raise ValueError(
                f"Unknown corpus_type: {corpus_type}. Must be 'laws', 'courts', or 'non_leading'."
            )

        # Convert to Path if it's a string
        if corpus_path and isinstance(corpus_path, str):
            corpus_path = Path(corpus_path)

        # Check if corpus path is configured
        if corpus_path is None:
            raise ValueError(
                f"BM25 index initialization failed: {corpus_type}_corpus_path is not configured. "
                f"Set {corpus_type}_corpus_path in config or preset. "
                f"Current config: laws_corpus_path={self.config.laws_corpus_path}, "
                f"courts_corpus_path={self.config.courts_corpus_path}"
            )

        if not corpus_path.exists():
            raise FileNotFoundError(
                f"Corpus file not found: {corpus_path}. "
                f"Please check the {corpus_type}_corpus_path configuration."
            )

        # Determine cache path
        cache_path = None
        if self.config.index_cache_dir is not None:
            self.config.index_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self.config.index_cache_dir / f"bm25_{corpus_type}.pkl"

        # Get use_german_stemming from config
        use_stemming = self.config.use_german_stemming

        # Fast path: load from cache if exists
        if cache_path and cache_path.exists():
            logger.info(f"Loading cached BM25 index from {cache_path}")
            try:
                return BM25Index.load(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cached index: {e}. Rebuilding...")

        # Slow path: build from CSV or JSONL
        logger.info(f"Building BM25 index from {corpus_path}")

        if corpus_path.suffix == ".jsonl":
            from omnilex.retrieval.bm25_index import load_jsonl_corpus

            documents = load_jsonl_corpus(corpus_path)
        else:
            from omnilex.retrieval.bm25_index import load_corpus_from_csv

            documents = load_corpus_from_csv(corpus_path)

        if not documents:
            logger.warning(f"No documents loaded from {corpus_path}")

        index = BM25Index(use_german_stemming=use_stemming)
        index.build(documents)

        # Save to cache
        if cache_path:
            logger.info(f"Saving BM25 index to cache: {cache_path}")
            index.save(cache_path)

        return index

    def run(
        self,
        queries: list[dict],
        ground_truth: dict[str, list] | None = None,
    ) -> dict:
        """Run experiment pipeline.

        Args:
            queries: List of query dicts with 'id', 'query', 'citations' keys
            ground_truth: Optional dict mapping query_id to gold citations.
                Accepted formats for gold citations:
                - List of strings: ["SR 123.1 Art. 5", "BGE 123 II 456"]
                - List of dicts with "id" key: [{"id": "SR 123.1 Art. 5"}]
                - List of dicts with "citation" key: [{"citation": "SR 123.1 Art. 5"}]
                - Mixed formats are also supported

        Returns:
            Dict with results and metrics
        """
        results = []

        for query in queries:
            query_id = query["id"]
            query_text = query["query"]

            # Stage 1: Retrieval
            signals = self._run_retrieval(query_text, query_id)

            # Track retrieval metrics if ground truth exists
            if ground_truth and query_id in ground_truth:
                gold_ids = self._normalize_gold_ids(ground_truth[query_id])
                retrieved_ids = []
                for sig_docs in signals.values():
                    retrieved_ids.extend([doc.get("id", "") for doc in sig_docs[:50]])
                self.metrics.track_retrieval(query_id, retrieved_ids, gold_ids)

            # Stage 2: Fusion
            if self.config.components.get("rrf_fusion"):
                fused = self._run_fusion(signals)
            else:
                fused = self._flatten_signals(signals)

            # Stage 3: Reranking
            if self.config.components.get("reranker"):
                reranked = self._run_reranker(query_text, fused)
                # Track reranker metrics
                if ground_truth and query_id in ground_truth:
                    gold_ids = self._normalize_gold_ids(ground_truth[query_id])
                    reranked_ids = [doc.get("id", "") for doc in reranked]
                    self.metrics.track_reranker(query_id, reranked_ids, gold_ids)
            else:
                reranked = fused

            # Stage 4: Verification
            if self.config.components.get("verifier"):
                verified = self._run_verifier(query_text, reranked)
                # Track verifier metrics
                if ground_truth and query_id in ground_truth:
                    gold_ids = self._normalize_gold_ids(ground_truth[query_id])
                    verified_ids = [doc.get("id", "") for doc in verified]
                    self.metrics.track_verifier(query_id, verified_ids, gold_ids)
            else:
                verified = reranked

            # Collect results
            citations = [doc.get("citation", doc.get("id", "")) for doc in verified]
            # Normalize citations to canonical format
            normalizer = CitationNormalizer()
            citations = normalizer.canonicalize_list(citations)
            results.append({"query_id": query_id, "citations": citations})

        # Prepare output
        output = {
            "results": results,
            "metrics": self.metrics.get_aggregate_metrics(),
        }

        # Save results
        self.save_results(output)

        # Generate submission.csv for validation
        import csv

        submission_path = self.output_dir / "submission.csv"
        with open(submission_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["query_id", "predicted_citations"])
            writer.writeheader()
            for res in results:
                writer.writerow(
                    {"query_id": res["query_id"], "predicted_citations": ";".join(res["citations"])}
                )

        # Run validation if script exists
        import subprocess

        eval_script = Path("scripts/evaluate_submission.py")
        if eval_script.exists():
            try:
                result = subprocess.run(
                    ["python", str(eval_script), str(submission_path)],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                logger.info(f"Validation output: {result.stdout}")
                if result.stderr:
                    logger.warning(f"Validation errors: {result.stderr}")
            except Exception as e:
                logger.warning(f"Failed to run validation: {e}")

        return output

    def _flatten_signals(self, signals: dict[str, list[dict]]) -> list[dict]:
        """Flatten signal dict into single document list.

        Args:
            signals: Dict mapping signal names to retrieved documents

        Returns:
            Deduplicated list of documents
        """
        flattened = []
        for signal_name, docs in signals.items():
            for doc in docs:
                doc_copy = doc.copy()
                doc_copy["signal_source"] = signal_name
                flattened.append(doc_copy)

        # Deduplicate by document ID
        seen_ids = set()
        unique_docs = []
        for doc in flattened:
            doc_id = doc.get("id", "")
            if doc_id and doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(doc)

        return unique_docs

    def _get_or_create_index(
        self,
        attr_name: str,
        cls: type,
        *args,
        **kwargs,
    ) -> Any:
        """Get existing index or create if None (lazy init).

        Args:
            attr_name: Name of the instance attribute (e.g., '_bm25_index')
            cls: Class to instantiate if attribute is None
            *args: Positional arguments for class constructor
            **kwargs: Keyword arguments for class constructor

        Returns:
            The index object (existing or newly created)
        """
        current = getattr(self, attr_name)
        if current is None:
            current = cls(*args, **kwargs)
            setattr(self, attr_name, current)
        return current

    def _run_retrieval(
        self,
        query: str,
        query_id: str,
    ) -> dict[str, list[dict]]:
        """Run retrieval stage based on enabled components.

        Args:
            query: Query text
            query_id: Query identifier

        Returns:
            Dict mapping signal names to retrieved documents
        """
        signals = {}

        if self.config.components.get("bm25"):
            # Initialize ALL three BM25 indices when bm25=True
            # 1. Laws index
            if self._bm25_index is None or self._bm25_index.index is None:
                self._bm25_index = self._init_bm25_index("laws")
            results = self._bm25_index.search(query, top_k=self.config.top_k)
            signals["bm25_laws"] = results

            # 2. Courts index (leading BGE decisions) - NOT gated behind graph=True
            if self._bm25_courts_index is None or self._bm25_courts_index.index is None:
                self._bm25_courts_index = self._init_bm25_index("courts")
            results = self._bm25_courts_index.search(query, top_k=self.config.top_k)
            signals["bm25_courts"] = results

            # 3. Non-leading decisions index
            if self._bm25_non_leading_index is None or self._bm25_non_leading_index.index is None:
                self._bm25_non_leading_index = self._init_bm25_index("non_leading")
            results = self._bm25_non_leading_index.search(query, top_k=self.config.top_k)
            signals["bm25_non_leading"] = results

        if self.config.components.get("dense"):
            index = self._get_or_create_index(
                "_dense_index",
                DenseIndex,
                index_preset=self.config.dense_index_preset,
            )
            results = index.search(query, top_k=self.config.top_k)
            signals["dense"] = results

        if self.config.components.get("graph"):
            # Initialize graph index (may also need courts BM25 index)
            if self._graph_index is None:
                # Courts index should already be initialized if bm25=True
                # If not, initialize it now for graph citation lookups
                if self._bm25_courts_index is None:
                    self._bm25_courts_index = self._init_bm25_index("courts")
                self._graph_index = self._get_or_create_index(
                    "_graph_index",
                    CitationGraph,
                )
            results = self._graph_index.search(query, top_k=self.config.top_k)
            signals["graph"] = results

        return signals

    def _run_fusion(
        self,
        signals: dict[str, list[dict]],
    ) -> list[dict]:
        """Run signal fusion stage.

        Args:
            signals: Dict mapping signal names to retrieved documents

        Returns:
            Fused and ranked list
        """
        from omnilex.retrieval.fusion import SignalFusion

        if self._fusion is None:
            self._fusion = SignalFusion(weights=self.config.signal_weights)
        return self._fusion.fuse(signals)

    def _run_reranker(
        self,
        query: str,
        documents: list[dict],
    ) -> list[dict]:
        """Run reranking stage.

        Args:
            query: Query text
            documents: Documents to rerank

        Returns:
            Reranked documents
        """
        from omnilex.retrieval.reranker import Reranker

        if self._reranker is None:
            self._reranker = Reranker(model_name=self.config.reranker_model)
        return self._reranker.rerank(query, documents, top_k=self.config.reranker_top_k)

    def _run_verifier(
        self,
        query: str,
        documents: list[dict],
    ) -> list[dict]:
        """Run verification stage.

        Args:
            query: Query text
            documents: Documents to verify

        Returns:
            Verified documents
        """
        from omnilex.retrieval.verifier import LLMVerifier

        if self._verifier is None:
            self._verifier = LLMVerifier(model_path=self.config.verifier_model)
        verified_docs = self._verifier.verify(query, documents)
        # Apply threshold filtering
        return self._verifier.filter_by_threshold(
            verified_docs, threshold=self.config.verifier_threshold
        )

    def save_results(self, results: dict) -> Path:
        """Save experiment results.

        Args:
            results: Experiment results

        Returns:
            Path to saved results
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{self.config.name}_results.json"

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        return output_path
