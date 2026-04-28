"""Retrieval tools and indexing for Swiss legal documents."""

from .bm25_index import BM25Index, build_index, load_jsonl_corpus, search
from .dense_index import DenseIndex
from .fusion import rrf_fusion, hybrid_search
from .citation_extractor import CitationExtractor, extract_citations
from .tools import CourtSearchTool, LawSearchTool

__all__ = [
    "BM25Index",
    "build_index",
    "load_jsonl_corpus",
    "search",
    "DenseIndex",
    "rrf_fusion",
    "hybrid_search",
    "CitationExtractor",
    "extract_citations",
    "LawSearchTool",
    "CourtSearchTool",
]
