"""Dense retrieval using FAISS and sentence embeddings."""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

# Optional imports - will be loaded at runtime
faiss = None
SentenceTransformer = None


def load_faiss():
    """Lazy load faiss."""
    global faiss
    if faiss is None:
        import faiss as _faiss
        faiss = _faiss
    return faiss


def load_embedding_model():
    """Lazy load sentence transformer."""
    global SentenceTransformer
    if SentenceTransformer is None:
        from sentence_transformers import SentenceTransformer
    return SentenceTransformer


class DenseIndex:
    """Dense retrieval index using FAISS.
    
    Uses sentence embeddings for semantic search.
    """

    def __init__(
        self,
        documents: Optional[List[Dict]] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: Optional[Path] = None,
    ):
        """Initialize dense index.
        
        Args:
            documents: List of document dictionaries (if building)
            model_name: Name of sentence transformer model
            index_path: Path to load/save FAISS index
        """
        self.model_name = model_name
        self.index_path = index_path
        self.documents: List[Dict] = []
        self.index = None
        self.model = None
        self.dimension = 0
        
        if documents:
            self.build(documents, model_name)
        elif index_path and Path(index_path).exists():
            self.load(index_path)

    def build(
        self,
        documents: List[Dict],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        """Build FAISS index from documents.
        
        Args:
            documents: List of document dictionaries
            model_name: Sentence transformer model name
        """
        self.documents = documents
        self.model_name = model_name
        
        # Load model
        print(f"Loading embedding model: {model_name}")
        model = load_embedding_model()
        self.model = model(model_name)
        
        # Generate embeddings
        print(f"Generating embeddings for {len(documents)} documents...")
        texts = [doc.get('text', '')[:512] for doc in documents]  # Truncate
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True,
        )
        
        # Build FAISS index
        faiss = load_faiss()
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product
        self.index.add(embeddings.astype('float32'))
        
        print(f"FAISS index built. Dimension: {self.dimension}")

    def search(
        self,
        query: str,
        top_k: int = 50,
    ) -> List[Tuple[int, float]]:
        """Search the index.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of (document_index, score) tuples
        """
        if self.index is None or self.model is None:
            raise ValueError("Index not built. Call build() first.")
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Search
        scores, indices = self.index.search(
            query_embedding.astype('float32'),
            top_k
        )
        
        # Convert to list of tuples
        results = [
            (int(indices[0][i]), float(scores[0][i]))
            for i in range(len(indices[0]))
            if indices[0][i] >= 0
        ]
        
        return results

    def get_document(self, idx: int) -> Optional[Dict]:
        """Get document by index."""
        if 0 <= idx < len(self.documents):
            return self.documents[idx]
        return None

    def save(self, path: Path | str) -> None:
        """Save index to disk.
        
        Args:
            path: Path to save index
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss = load_faiss()
        faiss.write_index(str(path.with_suffix('.index')), self.index)
        
        # Save metadata
        metadata = {
            "documents": self.documents,
            "model_name": self.model_name,
            "dimension": self.dimension,
        }
        
        with open(path.with_suffix('.metadata'), 'wb') as f:
            pickle.dump(metadata, f)

    @classmethod
    def load(cls, path: Path | str) -> "DenseIndex":
        """Load index from disk.
        
        Args:
            path: Path to saved index
            
        Returns:
            Loaded DenseIndex
        """
        path = Path(path)
        
        # Load metadata
        with open(path.with_suffix('.metadata'), 'rb') as f:
            metadata = pickle.load(f)
        
        # Create instance
        instance = cls(
            documents=metadata["documents"],
            model_name=metadata["model_name"],
        )
        instance.dimension = metadata["dimension"]
        
        # Load FAISS index
        faiss = load_faiss()
        instance.index = faiss.read_index(str(path.with_suffix('.index')))
        
        return instance


def load_jsonl_corpus(path: Path | str) -> List[Dict]:
    """Load corpus from JSONL file.
    
    Args:
        path: Path to JSONL file
        
    Returns:
        List of document dictionaries
    """
    path = Path(path)
    documents = []
    
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))
    
    return documents


def build_index(
    documents: List[Dict],
    text_field: str = "text",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> DenseIndex:
    """Build dense index from documents.
    
    Args:
        documents: List of document dictionaries
        text_field: Key for document text
        model_name: Sentence transformer model name
        
    Returns:
        Built DenseIndex
    """
    return DenseIndex(documents=documents, model_name=model_name)


def search(
    index: DenseIndex,
    query: str,
    top_k: int = 50,
) -> List[Tuple[int, float]]:
    """Search dense index.
    
    Args:
        index: DenseIndex to search
        query: Search query
        top_k: Number of results
        
    Returns:
        List of (document_index, score) tuples
    """
    return index.search(query, top_k=top_k)