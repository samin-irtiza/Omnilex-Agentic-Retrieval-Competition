"""Tests for graph_index.py citation graph module."""

import tempfile

import pytest

from src.omnilex.retrieval.graph_index import CitationGraph


@pytest.fixture
def sample_documents():
    """Sample legal documents for testing."""
    return [
        {
            "id": "BGE 127 III 248",
            "text": "This decision cites BGE 126 II 100 and SR 210 Art. 1. Also references SR 210 Art. 2.",
            "metadata": {"type": "decision", "citation": "BGE 127 III 248"},
        },
        {
            "id": "BGE 126 II 100",
            "text": "Cites SR 210 Art. 1 and SR 210 Art. 2.",
            "metadata": {"type": "decision", "citation": "BGE 126 II 100"},
        },
        {
            "id": "SR 210 Art. 1",
            "text": "Swiss Civil Code Article 1.",
            "metadata": {"type": "law", "citation": "SR 210 Art. 1"},
        },
        {
            "id": "SR 210 Art. 2",
            "text": "Swiss Civil Code Article 2.",
            "metadata": {"type": "law", "citation": "SR 210 Art. 2"},
        },
    ]


@pytest.fixture
def graph(sample_documents):
    """Pre-built citation graph."""
    g = CitationGraph()
    g.build_graph(sample_documents)
    return g


class TestCitationEdgeExtraction:
    def test_extract_bge_edges(self, sample_documents):
        g = CitationGraph()
        edges = g.extract_citation_edges(sample_documents)

        # Check BGE 127 III 248 cites BGE 126 II 100
        bge_edges = [e for e in edges if e[2]["type"] == "bge"]
        assert len(bge_edges) >= 1
        assert ("BGE 127 III 248", "BGE 126 II 100", ...) in bge_edges

    def test_extract_sr_edges(self, sample_documents):
        g = CitationGraph()
        edges = g.extract_citation_edges(sample_documents)

        # Check SR edges exist
        sr_edges = [e for e in edges if e[2]["type"] == "sr"]
        assert len(sr_edges) >= 2  # BGE 127 III 248 cites SR 210 Art. 1 and 2

    def test_no_self_citations(self, sample_documents):
        g = CitationGraph()
        edges = g.extract_citation_edges(sample_documents)
        for source, target, _ in edges:
            assert source != target


class TestGraphConstruction:
    def test_node_count(self, graph, sample_documents):
        assert graph.graph.number_of_nodes() == len(sample_documents)

    def test_edge_count(self, graph):
        # BGE 127 III 248 -> BGE 126 II 100, SR 210 Art.1, SR 210 Art.2
        # BGE 126 II 100 -> SR 210 Art.1, SR 210 Art.2
        assert graph.graph.number_of_edges() >= 5

    def test_node_metadata(self, graph):
        node = graph.graph.nodes["BGE 127 III 248"]
        assert node["type"] == "decision"
        assert "text_snippet" in node


class TestPPRRanking:
    def test_ppr_returns_scores(self, graph):
        scores = graph.ppr_rank(["BGE 127 III 248"])
        assert len(scores) > 0
        assert "BGE 127 III 248" in scores

    def test_ppr_empty_query(self, graph):
        scores = graph.ppr_rank([])
        assert scores == {}

    def test_ppr_invalid_query(self, graph):
        scores = graph.ppr_rank(["invalid_id"])
        assert scores == {}


class TestCommunityDetection:
    def test_community_detection_runs(self, graph):
        communities = graph.detect_communities()
        # Should return list of sets (even if empty)
        assert isinstance(communities, list)


class TestCoCitationAnalysis:
    def test_co_citation_scores(self, graph):
        scores = graph.co_citation_analysis(["SR 210 Art. 1", "SR 210 Art. 2"])
        # SR 210 Art.1 and 2 are co-cited by both BGE docs
        assert scores["SR 210 Art. 1"] > 0
        assert scores["SR 210 Art. 2"] > 0


class TestBibliographicCoupling:
    def test_bibliographic_coupling_scores(self, graph):
        scores = graph.bibliographic_coupling(["BGE 127 III 248", "BGE 126 II 100"])
        # Both cite SR 210 Art.1 and 2, so coupling score > 0
        assert scores["BGE 127 III 248"] > 0
        assert scores["BGE 126 II 100"] > 0


class TestPersistence:
    def test_save_load_pickle(self, graph):
        with tempfile.NamedTemporaryFile(suffix=".pkl") as f:
            graph.save(f.name)
            new_graph = CitationGraph()
            assert new_graph.load(f.name)
            assert new_graph.graph.number_of_nodes() == graph.graph.number_of_nodes()

    def test_save_load_gexf(self, graph):
        with tempfile.NamedTemporaryFile(suffix=".gexf") as f:
            graph.save(f.name)
            new_graph = CitationGraph()
            assert new_graph.load(f.name)
            assert new_graph.graph.number_of_edges() == graph.graph.number_of_edges()
