"""Citation graph retrieval module.

Provides:
- Citation edge extraction from court decisions
- Graph construction using networkx
- Personalized PageRank (PPR) for query-based ranking
- Leiden community detection
- Co-citation analysis
- Bibliographic coupling
"""

from __future__ import annotations

import logging
import pickle
import re
from collections import defaultdict
from pathlib import Path

import networkx as nx

from src.omnilex.citations.normalizer import CitationNormalizer

logger = logging.getLogger(__name__)


class CitationGraph:
    """Citation knowledge graph for legal document retrieval."""

    def __init__(self):
        """Initialize citation graph."""
        self.graph = nx.DiGraph()
        self.ppr_scores: dict[str, float] = {}
        self.communities: list[set[str]] = []

    def extract_citation_edges(
        self,
        documents: list[dict],
    ) -> list[tuple[str, str, dict]]:
        """Extract citation edges from court decisions.

        Args:
            documents: List of document dicts with 'id', 'text', 'metadata'

        Returns:
            List of (source_id, target_id, edge_attrs) tuples
        """
        normalizer = CitationNormalizer()
        citation_to_doc = {}

        # Build mapping from normalized citation to document ID
        for doc in documents:
            doc_id = doc["id"]
            normalized = normalizer.normalize(doc_id)
            citation_to_doc[normalized] = doc_id

        edges = []
        bge_pattern = re.compile(r"BGE (\d+) ([IVXLCDM]+) (\d+)")
        sr_pattern = re.compile(r"SR (\d+) Art\. (\d+)")

        for doc in documents:
            source_id = doc["id"]
            text = doc.get("text", "")

            # Extract BGE references
            for match in bge_pattern.finditer(text):
                vol, sect, page = match.groups()
                raw_citation = f"BGE {vol} {sect} {page}"
                normalized = normalizer.normalize(raw_citation)
                if normalized in citation_to_doc:
                    target_id = citation_to_doc[normalized]
                    if target_id != source_id:
                        edges.append(
                            (source_id, target_id, {"type": "bge", "extracted_text": raw_citation})
                        )

            # Extract SR references
            for match in sr_pattern.finditer(text):
                num, art = match.groups()
                raw_citation = f"SR {num} Art. {art}"
                normalized = normalizer.normalize(raw_citation)
                if normalized in citation_to_doc:
                    target_id = citation_to_doc[normalized]
                    if target_id != source_id:
                        edges.append(
                            (source_id, target_id, {"type": "sr", "extracted_text": raw_citation})
                        )

        return edges

    def build_graph(
        self,
        documents: list[dict],
        force_rebuild: bool = False,
    ) -> None:
        """Build citation graph from documents.

        Args:
            documents: List of document dicts
            force_rebuild: Whether to rebuild graph even if cached
        """
        if not force_rebuild and self.graph.number_of_nodes() > 0:
            logger.info("Graph already built, use force_rebuild=True to rebuild")
            return

        self.graph.clear()

        # Add nodes with metadata
        for doc in documents:
            doc_id = doc["id"]
            metadata = doc.get("metadata", {})
            node_attrs = {
                "type": metadata.get("type", "unknown"),
                "text_snippet": doc.get("text", "")[:200],
                **metadata,
            }
            self.graph.add_node(doc_id, **node_attrs)

        # Extract and add edges
        edges = self.extract_citation_edges(documents)
        for source, target, attrs in edges:
            if source in self.graph and target in self.graph:
                self.graph.add_edge(source, target, **attrs)

        logger.info(
            f"Built graph with {self.graph.number_of_nodes()} nodes "
            f"and {self.graph.number_of_edges()} edges"
        )

    def ppr_rank(
        self,
        query_doc_ids: list[str],
        damping: float = 0.85,
        max_iter: int = 100,
    ) -> dict[str, float]:
        """Run Personalized PageRank from query seed set.

        Args:
            query_doc_ids: Seed documents for PPR
            damping: Damping factor (default 0.85)
            max_iter: Maximum iterations

        Returns:
            Dict mapping doc_id to PPR score
        """
        if not query_doc_ids:
            return {}

        valid_queries = [qid for qid in query_doc_ids if qid in self.graph]
        if not valid_queries:
            return {}

        personalization = {qid: 1.0 / len(valid_queries) for qid in valid_queries}

        try:
            ppr_scores = nx.pagerank(
                self.graph,
                personalization=personalization,
                alpha=damping,
                max_iter=max_iter,
            )
        except nx.PowerIterationFailedConvergence:
            logger.warning("PPR failed to converge, returning empty scores")
            return {}

        self.ppr_scores = ppr_scores
        return ppr_scores

    def detect_communities(self) -> list[set[str]]:
        """Detect communities using Leiden algorithm.

        Returns:
            List of communities (sets of doc_ids)
        """
        if self.graph.number_of_nodes() == 0:
            logger.warning("Graph is empty, cannot detect communities")
            return []

        und_graph = self.graph.to_undirected()

        # Try Leiden algorithm first
        logger.info("Attempting community detection with Leiden algorithm")
        try:
            import igraph as ig
            import leidenalg

            ig_graph = ig.Graph.from_networkx(und_graph)
            partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition)
            self.communities = [set(ig_graph.vs[part]["name"]) for part in partition]
            logger.info(f"Detected {len(self.communities)} communities via Leiden")
            return self.communities

        except ImportError:
            logger.warning("leidenalg not available, falling back to Louvain")
            logger.info("Attempting community detection with Louvain algorithm")

        # Fallback to Louvain
        try:
            import community

            partition = community.best_partition(und_graph)
            comm_dict = defaultdict(set)
            for doc_id, comm_id in partition.items():
                comm_dict[comm_id].add(doc_id)
            self.communities = list(comm_dict.values())
            logger.info(f"Detected {len(self.communities)} communities via Louvain")
            return self.communities

        except ImportError:
            logger.warning("Neither leidenalg nor python-louvain available")
            self.communities = []
            return []

    def co_citation_analysis(
        self,
        candidate_ids: list[str],
    ) -> dict[str, float]:
        """Compute co-citation scores for candidates.

        Args:
            candidate_ids: Candidate document IDs

        Returns:
            Dict mapping doc_id to co-citation score
        """
        co_citation_counts = defaultdict(int)

        for source in self.graph.nodes():
            cited = list(self.graph.successors(source))
            # Count co-citations for pairs of cited documents
            for i in range(len(cited)):
                for j in range(i + 1, len(cited)):
                    a, b = cited[i], cited[j]
                    co_citation_counts[a] += 1
                    co_citation_counts[b] += 1

        return {cand: float(co_citation_counts.get(cand, 0)) for cand in candidate_ids}

    def bibliographic_coupling(
        self,
        candidate_ids: list[str],
    ) -> dict[str, float]:
        """Compute bibliographic coupling scores.

        Args:
            candidate_ids: Candidate document IDs

        Returns:
            Dict mapping doc_id to coupling score
        """
        # Get cited documents for each candidate
        cited_by = {}
        for cand in candidate_ids:
            if cand in self.graph:
                cited_by[cand] = set(self.graph.successors(cand))

        coupling_scores = defaultdict(float)
        candidates = list(cited_by.keys())

        for i in range(len(candidates)):
            cand1 = candidates[i]
            for j in range(i + 1, len(candidates)):
                cand2 = candidates[j]
                intersection = len(cited_by[cand1] & cited_by[cand2])
                coupling_scores[cand1] += intersection
                coupling_scores[cand2] += intersection

        return {cand: coupling_scores.get(cand, 0.0) for cand in candidate_ids}

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search graph using query-derived seeds and PPR ranking.

        Args:
            query: Query text to find seed documents
            top_k: Number of results to return

        Returns:
            List of document dicts with 'id', 'citation', 'ppr_score'
        """
        if self.graph.number_of_nodes() == 0:
            return []

        # Simple keyword search on node text snippets to find seeds
        query_tokens = set(re.findall(r"\w+", query.lower()))
        seed_scores = defaultdict(float)

        for node_id, node_attrs in self.graph.nodes(data=True):
            text = node_attrs.get("text_snippet", "").lower()
            node_tokens = set(re.findall(r"\w+", text))
            overlap = len(query_tokens & node_tokens)
            if overlap > 0:
                seed_scores[node_id] = overlap

        # Get top seeds (max 20)
        top_seeds = sorted(seed_scores.items(), key=lambda x: x[1], reverse=True)[:20]
        seed_ids = [sid for sid, _ in top_seeds]

        if not seed_ids:
            return []

        # Run PPR from seeds
        ppr_scores = self.ppr_rank(seed_ids)

        # Sort documents by PPR score
        sorted_docs = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Format results
        results = []
        for doc_id, score in sorted_docs:
            node_attrs = self.graph.nodes[doc_id]
            results.append({"id": doc_id, "citation": doc_id, "ppr_score": score, **node_attrs})

        return results

    def save(self, path: str | Path) -> None:
        """Save graph to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".gexf":
            nx.write_gexf(self.graph, path)
        elif path.suffix == ".graphml":
            nx.write_graphml(self.graph, path)
        else:
            with open(path, "wb") as f:
                pickle.dump(
                    {"graph": self.graph, "communities": self.communities},
                    f,
                )

        logger.info(f"Saved graph to {path}")

    def load(self, path: str | Path) -> bool:
        """Load graph from disk."""
        path = Path(path)
        if not path.exists():
            logger.error(f"Path {path} does not exist")
            return False

        try:
            if path.suffix == ".gexf":
                self.graph = nx.read_gexf(path)
            elif path.suffix == ".graphml":
                self.graph = nx.read_graphml(path)
            else:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                    self.graph = data["graph"]
                    self.communities = data.get("communities", [])

            logger.info(f"Loaded graph from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
            return False
