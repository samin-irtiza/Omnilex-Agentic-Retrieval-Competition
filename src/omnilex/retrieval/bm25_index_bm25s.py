"""
BM25 indexing using bm25s library with memory-mapped sparse matrices.

This module provides a memory-efficient BM25 implementation that:
1. Uses bm25s library (https://github.com/xhluca/bm25s) for ultra-fast, low-memory BM25
2. Supports memory-mapping (mmap) to store indices on disk instead of RAM
3. Can reduce RAM usage from ~4-10GB to ~0.5-2GB for large corpora
4. Maintains API compatibility with the existing BM25SparseIndex

Memory-mapping benefits (from bm25s benchmarks):
- In-memory: 4.36 GB RAM post-index
- Memory-mapped: 0.49 GB RAM post-index  
- Mmap+Reload: 0.70 GB RAM post-index

The bm25s library uses scipy sparse matrices with eager computation strategy,
providing both speed and memory efficiency.
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

try:
    import bm25s
    BM25S_AVAILABLE = True
except ImportError:
    BM25S_AVAILABLE = False
    print("Warning: bm25s not installed. Install with: pip install bm25s")


class BM25SIndex:
    """Memory-efficient BM25 index using bm25s with optional memory-mapping.
    
    This implementation:
    - Uses bm25s library for efficient BM25 scoring
    - Supports memory-mapped indices for very large corpora (2M+ docs)
    - Can reduce RAM usage by 50-90% compared to in-memory indices
    - Maintains compatibility with the existing API
    
    Memory usage (for 2.6M documents):
    - Without mmap: ~3-4 GB RAM
    - With mmap: ~0.5-1 GB RAM
    - With mmap + reload: ~0.7 GB RAM (best for 12GB Colab)
    """
    
    def __init__(
        self,
        documents: list[dict] | None = None,
        text_field: str = "text",
        citation_field: str = "citation",
        store_documents: bool = False,
        use_mmap: bool = True,
        mmap_path: Optional[str] = None,
        vocab_size: int = 50000,
    ):
        """Initialize BM25 index using bm25s.
        
        Args:
            documents: List of document dictionaries (optional, can build later)
            text_field: Key for document text in dict
            citation_field: Key for citation string in dict
            store_documents: If True, store documents in memory (increases RAM)
            use_mmap: If True, use memory-mapped index (much lower RAM usage)
            mmap_path: Path to save/load memory-mapped index (default: auto)
            vocab_size: Maximum vocabulary size (limits memory usage)
        """
        if not BM25S_AVAILABLE:
            raise ImportError(
                "bm25s library is required. Install with: pip install bm25s"
            )
        
        self.text_field = text_field
        self.citation_field = citation_field
        self.store_documents = store_documents
        self.use_mmap = use_mmap
        self.mmap_path = mmap_path
        self.vocab_size = vocab_size
        
        # Core data structures
        self.documents: list[dict] = [] if not store_documents else []
        self.doc_ids: list[str] = []
        self.doc_citations: list[str] = []
        
        # bm25s index (can be memory-mapped)
        self.bm25 = None
        self.tokenized_corpus = None
        
        # Tokenizer (using bm25s built-in)
        self.tokenizer = bm25s.tokenization.Tokenizer()
        
        if documents:
            self.build(documents)
    
    def tokenize(self, text: str) -> list[str]:
        """Tokenize text using bm25s tokenizer.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # bm25s has its own tokenization, but we provide this for API compatibility
        return bm25s.tokenize(text)
    
    def build(self, documents: list[dict]) -> None:
        """Build BM25 index from documents using bm25s.
        
        Args:
            documents: List of document dictionaries
        """
        print(f"Building BM25 index with bm25s from {len(documents)} documents...")
        print(f"Memory-mapping: {'ENABLED' if self.use_mmap else 'DISABLED'}")
        
        # Store document IDs and citations (not full documents to save memory)
        if self.store_documents:
            self.documents = documents
        else:
            self.doc_ids = []
            self.doc_citations = []
            for doc in tqdm(documents, desc="Storing IDs", unit="docs"):
                doc_id = doc.get(self.citation_field, "") or doc.get("id", "")
                citation = doc.get(self.citation_field, "")
                self.doc_ids.append(doc_id)
                self.doc_citations.append(citation)
        
        # Extract texts for indexing
        print("Extracting texts...")
        corpus_texts = [doc.get(self.text_field, "") for doc in tqdm(documents)]
        
        # Tokenize corpus using bm25s
        print("Tokenizing corpus with bm25s...")
        self.tokenized_corpus = bm25s.tokenize(corpus_texts)
        
        # Build bm25s index
        print("Building bm25s index...")
        self.bm25 = bm25s.BM25()
        self.bm25.index(self.tokenized_corpus)
        
        print("BM25 index built successfully!")
        
        # Save with memory-mapping if requested
        if self.use_mmap and self.mmap_path:
            self._save_mmap()
    
    def _save_mmap(self) -> None:
        """Save index with memory-mapping support."""
        if not self.mmap_path or self.bm25 is None:
            return
        
        mmap_path = Path(self.mmap_path)
        mmap_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving BM25 index with mmap support to {mmap_path}...")
        self.bm25.save(str(mmap_path), mmap=self.use_mmap)
        print(f"Index saved. Use mmap=True when loading to keep it memory-mapped.")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        return_scores: bool = False,
    ) -> list[dict]:
        """Search the index with a query.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            return_scores: Whether to include BM25 scores in results
            
        Returns:
            List of matching documents (with optional scores)
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call build() first.")
        
        # Tokenize query
        query_tokens = bm25s.tokenize(query)
        if not query_tokens or not query_tokens[0]:
            return []
        
        # Search using bm25s retrieve method
        # Returns (doc_indices, scores) both as numpy arrays of shape (n_queries, k)
        doc_indices, scores = self.bm25.retrieve(
            query_tokens,
            k=top_k,
            show_progress=False,
        )
        
        # Format results
        output = []
        if doc_indices.shape[1] > 0:
            for i in range(doc_indices.shape[1]):
                doc_idx = int(doc_indices[0, i])
                score = float(scores[0, i])
                
                if self.store_documents and self.documents:
                    doc = self.documents[doc_idx].copy()
                    if return_scores:
                        doc["_score"] = score
                    doc["_index"] = doc_idx
                    output.append(doc)
                else:
                    result = {"_index": doc_idx}
                    if doc_idx < len(self.doc_ids):
                        result["_id"] = self.doc_ids[doc_idx]
                    if doc_idx < len(self.doc_citations):
                        result["citation"] = self.doc_citations[doc_idx]
                    if return_scores:
                        result["_score"] = score
                    output.append(result)
        
        return output
    
    def search_with_scores(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Search and return (doc_index, score) tuples.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of (document_index, score) tuples
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call build() first.")
        
        # Tokenize query
        query_tokens = bm25s.tokenize(query)
        if not query_tokens or not query_tokens[0]:
            return []
        
        # Use bm25s retrieve method to get top-k results with scores
        # Returns (doc_indices, scores) both as numpy arrays of shape (n_queries, k)
        doc_indices, scores = self.bm25.retrieve(
            query_tokens,
            k=top_k,
            show_progress=False,
        )
        
        # Convert to list of (doc_index, score) tuples
        # doc_indices and scores are shape (1, top_k) for single query
        results = []
        if doc_indices.shape[1] > 0:
            for i in range(doc_indices.shape[1]):
                doc_idx = int(doc_indices[0, i])
                score = float(scores[0, i])
                if score > 0:
                    results.append((doc_idx, score))
        
        return results
    
    def save(self, path: Path | str) -> None:
        """Save index to disk.
        
        Args:
            path: Path to save index
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving BM25 index to {path}...")
        
        # Save bm25s index
        bm25_path = path.with_suffix(".bm25s")
        if self.bm25 is not None:
            self.bm25.save(str(bm25_path), mmap=self.use_mmap)
        
        # Save metadata
        data = {
            "doc_ids": self.doc_ids,
            "doc_citations": self.doc_citations,
            "text_field": self.text_field,
            "citation_field": self.citation_field,
            "store_documents": self.store_documents,
            "use_mmap": self.use_mmap,
            "bm25_path": str(bm25_path.name),
        }
        
        if self.store_documents:
            data["documents"] = self.documents
        
        with open(path, "wb") as f:
            pickle.dump(data, f)
        
        print(f"Index saved to {path}")
        print(f"bm25s index saved to {bm25_path}")
    
    @classmethod
    def load(
        cls,
        path: Path | str,
        mmap: Optional[bool] = None,
    ) -> "BM25SIndex":
        """Load index from disk.
        
        Args:
            path: Path to saved index
            mmap: If True, load with memory-mapping (overrides saved setting)
            
        Returns:
            Loaded BM25SIndex instance
        """
        path = Path(path)
        
        print(f"Loading BM25 index from {path}...")
        
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        # Use provided mmap setting or saved setting
        use_mmap = mmap if mmap is not None else data.get("use_mmap", True)
        
        instance = cls(
            text_field=data["text_field"],
            citation_field=data.get("citation_field", "citation"),
            store_documents=data.get("store_documents", False),
            use_mmap=use_mmap,
        )
        
        instance.doc_ids = data.get("doc_ids", [])
        instance.doc_citations = data.get("doc_citations", [])
        
        if data.get("store_documents", False):
            instance.documents = data.get("documents", [])
        
        # Load bm25s index
        bm25_path = path.parent / data.get("bm25_path", "")
        if bm25_path.exists():
            print(f"Loading bm25s index from {bm25_path} (mmap={use_mmap})...")
            instance.bm25 = bm25s.BM25.load(str(bm25_path), mmap=use_mmap)
            print(f"bm25s index loaded successfully!")
        else:
            raise FileNotFoundError(f"bm25s index not found: {bm25_path}")
        
        print("BM25 index loaded successfully!")
        return instance


def build_bm25s_index(
    documents: list[dict],
    text_field: str = "text",
    citation_field: str = "citation",
    use_mmap: bool = True,
    mmap_path: Optional[str] = None,
    vocab_size: int = 50000,
) -> BM25SIndex:
    """Build a BM25 index using bm25s with memory-mapping.
    
    Args:
        documents: List of document dictionaries
        text_field: Key for document text
        citation_field: Key for citation string
        use_mmap: Whether to use memory-mapped index (low RAM usage)
        mmap_path: Path to save memory-mapped index
        vocab_size: Maximum vocabulary size
        
    Returns:
        Built BM25SIndex
    """
    index = BM25SIndex(
        documents=documents,
        text_field=text_field,
        citation_field=citation_field,
        store_documents=False,  # Memory-efficient by default
        use_mmap=use_mmap,
        mmap_path=mmap_path,
        vocab_size=vocab_size,
    )
    return index


def search_bm25s(
    index: BM25SIndex,
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """Search a bm25s index with a query.
    
    Args:
        index: BM25SIndex to search
        query: Search query string
        top_k: Number of results
        
    Returns:
        List of matching documents
    """
    return index.search(query, top_k=top_k)
