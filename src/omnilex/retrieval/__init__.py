"""Retrieval tools and indexing for Swiss legal documents."""

from .bm25_index import BM25Index, build_index, load_jsonl_corpus, search
from .bm25_index_gpu import BM25SparseIndex, build_sparse_index
from .bm25_faiss_gpu import BM25FAISSIndex, build_faiss_index
from .dense_index import DenseIndex
from .fusion import rrf_fusion, hybrid_search
from .citation_extractor import CitationExtractor, extract_citations
from .tools import CourtSearchTool, LawSearchTool

__all__ = [
    "BM25Index",
    "build_index",
    "load_jsonl_corpus",
    "search",
    "BM25SparseIndex",
    "build_sparse_index",
    "BM25FAISSIndex",
    "build_faiss_index",
    "DenseIndex",
    "rrf_fusion",
    "hybrid_search",
    "CitationExtractor",
    "extract_citations",
    "LawSearchTool",
    "CourtSearchTool",
]
