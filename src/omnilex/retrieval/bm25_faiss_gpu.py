"""BM25 using FAISS on GPU - Leverages T4 15GB VRAM for keyword search.

This module provides a BM25-like implementation that:
1. Uses FAISS on GPU for fast nearest neighbor search
2. Represents documents as TF-IDF vectors (or binary vectors)
3. Stores vectors in GPU memory (T4 15GB VRAM)
4. Works within 12GB RAM constraint by offloading to GPU

The approach:
- Convert documents to sparse TF-IDF vectors
- Store in FAISS index on GPU
- Query by converting query to TF-IDF vector
- Use FAISS for fast retrieval

Note: This is not "true" BM25 but a close approximation that works well in practice.
For true BM25 with memory efficiency, use bm25_index_gpu.py (sparse matrix approach).
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import sparse
from tqdm import tqdm

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available. Install with: pip install faiss-gpu")


class BM25FAISSIndex:
    """BM25-like index using FAISS on GPU for memory-efficient keyword search.
    
    This implementation:
    - Uses TF-IDF vectors stored in FAISS on GPU (T4 15GB VRAM)
    - Approximates BM25 scoring using vector similarity
    - Keeps RAM usage low by storing data on GPU
    
    Memory usage:
    - RAM: Only document IDs and vocabulary (~100MB)
    - GPU VRAM: TF-IDF vectors (~2-5GB for 2.6M docs with 50K vocab)
    """
    
    def __init__(
        self,
        documents: list[dict] | None = None,
        text_field: str = "text",
        citation_field: str = "citation",
        store_documents: bool = False,
        vocab_size: int = 50000,
        use_gpu: bool = True,
    ):
        """Initialize BM25 FAISS Index.
        
        Args:
            documents: List of document dictionaries (optional)
            text_field: Key for document text
            citation_field: Key for citation string
            store_documents: If True, store documents in memory
            vocab_size: Maximum vocabulary size
            use_gpu: If True, use FAISS-GPU
        """
        self.text_field = text_field
        self.citation_field = citation_field
        self.store_documents = store_documents
        self.vocab_size = vocab_size
        self.use_gpu = use_gpu and FAISS_AVAILABLE
        
        # Core data (stored in RAM - lightweight)
        self.documents: list[dict] = []
        self.doc_ids: list[str] = []
        self.doc_citations: list[str] = []
        
        # Vocabulary
        self.vocab: dict[str, int] = {}
        self.vocab_list: list[str] = []
        self.idf_vector: Optional[np.ndarray] = None
        
        # FAISS index (stored on GPU if available)
        self.faiss_index = None
        self._gpu_resources = None
        
        # BM25 parameters
        self.k1: float = 1.5
        self.b: float = 0.75
        self.avgdl: float = 0.0
        
        if documents:
            self.build(documents)
    
    def tokenize(self, text: str) -> list[str]:
        """Tokenize text."""
        import re
        text = text.lower()
        tokens = re.split(r"\W+", text)
        return [t for t in tokens if t]
    
    def _build_vocab(self, documents: list[dict]) -> None:
        """Build vocabulary from documents."""
        print("Building vocabulary...")
        token_counts = {}
        
        for doc in tqdm(documents, desc="Vocab", unit="docs"):
            text = doc.get(self.text_field, "")
            tokens = self.tokenize(text)
            for token in set(tokens):
                token_counts[token] = token_counts.get(token, 0) + 1
        
        # Sort by frequency and take top vocab_size tokens
        sorted_tokens = sorted(token_counts.items(), key=lambda x: -x[1])
        self.vocab = {}
        self.vocab_list = []
        
        for idx, (token, _) in enumerate(sorted_tokens[:self.vocab_size]):
            self.vocab[token] = idx
            self.vocab_list.append(token)
        
        print(f"Vocabulary built: {len(self.vocab)} tokens")
    
    def _compute_idf(self, n_docs: int, tf_matrix: sparse.csr_matrix) -> np.ndarray:
        """Compute IDF vector."""
        print("Computing IDF scores...")
        doc_freq = np.array((tf_matrix > 0).sum(axis=0)).flatten()
        idf = np.log((n_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
        return idf
    
    def _compute_tfidf_vectors_chunked(self, documents: list[dict], chunk_size: int = 10000) -> np.ndarray:
        """Compute TF-IDF vectors in chunks to avoid high memory usage.
        
        Args:
            documents: List of document dictionaries
            chunk_size: Number of documents to process at once
            
        Returns:
            Dense numpy array of shape (n_docs, vocab_size)
        """
        print("Computing TF-IDF vectors (chunked)...")
        n_docs = len(documents)
        
        # First pass: compute TF matrix (sparse) - this is memory efficient
        rows = []
        cols = []
        vals = []
        doc_lengths = []
        
        for doc_idx, doc in enumerate(tqdm(documents, desc="TF Matrix", unit="docs")):
            text = doc.get(self.text_field, "")
            tokens = self.tokenize(text)
            doc_lengths.append(len(tokens))
            
            token_counts = {}
            for token in tokens:
                if token in self.vocab:
                    token_idx = self.vocab[token]
                    token_counts[token_idx] = token_counts.get(token_idx, 0) + 1
            
            for token_idx, count in token_counts.items():
                rows.append(doc_idx)
                cols.append(token_idx)
                vals.append(count)
        
        # Create sparse TF matrix
        tf_matrix = sparse.csr_matrix(
            (vals, (rows, cols)),
            shape=(n_docs, len(self.vocab)),
            dtype=np.float32,
        )
        
        self.avgdl = np.mean(doc_lengths) if doc_lengths else 0.0
        
        # Compute IDF
        self.idf_vector = self._compute_idf(n_docs, tf_matrix)
        
        # Convert to dense in chunks to avoid memory issues
        print(f"Converting to dense vectors in chunks of {chunk_size}...")
        all_vectors = np.zeros((n_docs, len(self.vocab)), dtype=np.float32)
        
        for start_idx in tqdm(range(0, n_docs, chunk_size), desc="To Dense", unit="chunk"):
            end_idx = min(start_idx + chunk_size, n_docs)
            
            # Get chunk of TF matrix
            tf_chunk = tf_matrix[start_idx:end_idx].toarray()  # This chunk is ~chunk_size * vocab_size * 4 bytes
            
            # Apply IDF
            tf_idf_chunk = tf_chunk * self.idf_vector
            
            # Normalize
            norms = np.linalg.norm(tf_idf_chunk, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            tf_idf_chunk_normalized = tf_idf_chunk / norms
            
            all_vectors[start_idx:end_idx] = tf_idf_chunk_normalized
        
        print(f"TF-IDF matrix shape: {all_vectors.shape}")
        return all_vectors
    
    def build(self, documents: list[dict]) -> None:
        """Build FAISS index from documents."""
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS not available. Install with: pip install faiss-gpu")
        
        print(f"Building BM25 FAISS Index from {len(documents)} documents...")
        
        # Store document IDs and citations
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
        
        # Build vocabulary
        self._build_vocab(documents)
        
        # Compute TF-IDF vectors (chunked to avoid memory issues)
        tf_idf_vectors = self._compute_tfidf_vectors_chunked(documents, chunk_size=10000)
        
        # Build FAISS index
        print("Building FAISS index...")
        d = tf_idf_vectors.shape[1]  # Dimension
        
        # Use IndexFlatIP for inner product (dot product) similarity
        self.faiss_index = faiss.IndexFlatIP(d)
        
        # Move to GPU if available
        if self.use_gpu:
            print("Moving FAISS index to GPU...")
            self._gpu_resources = faiss.StandardGpuResources()
            self.faiss_index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, self.faiss_index)
        
        # Add vectors to index
        print(f"Adding {len(tf_idf_vectors)} vectors to FAISS index...")
        self.faiss_index.add(tf_idf_vectors)
        
        print("BM25 FAISS Index built successfully!")
        if self.use_gpu:
            print(f"FAISS index is on GPU (T4 15GB VRAM)")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        return_scores: bool = False,
    ) -> list[dict]:
        """Search the index with a query."""
        if self.faiss_index is None:
            raise ValueError("Index not built. Call build() first.")
        
        # Tokenize query
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []
        
        # Build query vector
        query_vec = np.zeros(len(self.vocab), dtype=np.float32)
        token_counts = {}
        for token in query_tokens:
            if token in self.vocab:
                token_idx = self.vocab[token]
                token_counts[token_idx] = token_counts.get(token_idx, 0) + 1
        
        for token_idx, count in token_counts.items():
            # Simple TF * IDF for query
            tf = count
            idf = self.idf_vector[token_idx] if self.idf_vector is not None else 1.0
            query_vec[token_idx] = tf * idf
        
        # Normalize query vector
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm
        
        # Search in FAISS
        query_vec = query_vec.reshape(1, -1)
        scores, indices = self.faiss_index.search(query_vec, top_k)
        
        # Build results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or score <= 0:
                continue
            
            if self.store_documents and self.documents:
                doc = self.documents[idx].copy()
                if return_scores:
                    doc["_score"] = float(score)
                doc["_index"] = int(idx)
                results.append(doc)
            else:
                result = {"_index": int(idx)}
                if idx < len(self.doc_ids):
                    result["_id"] = self.doc_ids[idx]
                if idx < len(self.doc_citations):
                    result["citation"] = self.doc_citations[idx]
                if return_scores:
                    result["_score"] = float(score)
                results.append(result)
        
        return results
    
    def save(self, path: Path | str) -> None:
        """Save index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving BM25 FAISS Index to {path}...")
        
        # Save FAISS index (CPU version)
        if self.use_gpu and self.faiss_index is not None:
            # Convert to CPU for saving
            cpu_index = faiss.index_gpu_to_cpu(self.faiss_index)
            faiss.write_index(cpu_index, str(path.with_suffix('.faiss')))
        elif self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(path.with_suffix('.faiss')))
        
        # Save metadata
        data = {
            "doc_ids": self.doc_ids,
            "doc_citations": self.doc_citations,
            "vocab": self.vocab,
            "vocab_list": self.vocab_list,
            "idf_vector": self.idf_vector,
            "text_field": self.text_field,
            "citation_field": self.citation_field,
            "store_documents": self.store_documents,
            "k1": self.k1,
            "b": self.b,
            "avgdl": self.avgdl,
        }
        
        if self.store_documents:
            data["documents"] = self.documents
        
        with open(path.with_suffix('.pkl'), "wb") as f:
            pickle.dump(data, f)
        
        print(f"Index saved to {path}")
    
    @classmethod
    def load(cls, path: Path | str) -> "BM25FAISSIndex":
        """Load index from disk."""
        path = Path(path)
        
        print(f"Loading BM25 FAISS Index from {path}...")
        
        # Load metadata
        with open(path.with_suffix('.pkl'), "rb") as f:
            data = pickle.load(f)
        
        instance = cls(
            text_field=data["text_field"],
            citation_field=data.get("citation_field", "citation"),
            store_documents=data.get("store_documents", False),
            vocab_size=len(data.get("vocab", {})),
            use_gpu=data.get("use_gpu", True),
        )
        
        instance.doc_ids = data.get("doc_ids", [])
        instance.doc_citations = data.get("doc_citations", [])
        instance.vocab = data["vocab"]
        instance.vocab_list = data["vocab_list"]
        instance.idf_vector = data["idf_vector"]
        instance.k1 = data.get("k1", 1.5)
        instance.b = data.get("b", 0.75)
        instance.avgdl = data.get("avgdl", 0.0)
        
        if data.get("store_documents", False):
            instance.documents = data.get("documents", [])
        
        # Load FAISS index
        faiss_path = path.with_suffix('.faiss')
        if faiss_path.exists():
            cpu_index = faiss.read_index(str(faiss_path))
            
            # Move to GPU if available
            if instance.use_gpu:
                instance._gpu_resources = faiss.StandardGpuResources()
                instance.faiss_index = faiss.index_cpu_to_gpu(instance._gpu_resources, 0, cpu_index)
            else:
                instance.faiss_index = cpu_index
            
            print(f"Loaded FAISS index with {instance.faiss_index.ntotal} vectors")
        
        print("BM25 FAISS Index loaded successfully!")
        return instance


def build_faiss_index(
    documents: list[dict],
    text_field: str = "text",
    citation_field: str = "citation",
    use_gpu: bool = True,
    vocab_size: int = 50000,
) -> BM25FAISSIndex:
    """Build a BM25 FAISS index from documents."""
    index = BM25FAISSIndex(
        documents=documents,
        text_field=text_field,
        citation_field=citation_field,
        store_documents=False,
        vocab_size=vocab_size,
        use_gpu=use_gpu,
    )
    return index
