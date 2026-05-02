"""Dense retrieval module using BGE-M3 embeddings and FAISS indexing.

Provides:
- BGE-M3 embedding generation (FP16 and GGUF quantized)
- FAISS index construction with multiple presets (FlatIP, HNSW, IVFPQ)
- Dense similarity search with configurable top_k
- Index persistence (save/load to disk)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Default embedding dimension for BGE-M3
BGE_M3_DIM = 1024

# FAISS index presets
INDEX_PRESETS = {
    "quality": "IndexFlatIP",  # Exact search, 100% recall
    "balanced": "IndexHNSWFlat",  # HNSW graph, ~95% recall
    "minimal": "IndexIVFPQ",  # Quantized, ~90% recall, lowest memory
}


class DenseIndex:
    """Dense retrieval using BGE-M3 embeddings and FAISS."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        index_preset: str = "balanced",
        use_quantized: bool = False,
        device: str = "cpu",
    ):
        """Initialize dense index.

        Args:
            model_name: HuggingFace model name for BGE-M3
            index_preset: One of "quality", "balanced", "minimal"
            use_quantized: Whether to use GGUF quantized model
            device: Device to run embeddings on ("cpu" or "cuda")
        """
        self.model_name = model_name
        self.index_preset = index_preset
        self.use_quantized = use_quantized
        self.device = device

        self.model = None
        self.index = None
        self.doc_ids: list[str] = []
        self.doc_metadata: list[dict] = []

    def load_model(self) -> None:
        """Load BGE-M3 model for embedding generation.

        Tries flag_embedding first, falls back to sentence-transformers.
        Supports GGUF quantized models via llama-cpp-python.
        """
        if self.use_quantized:
            self._load_quantized_model()
        else:
            self._load_full_model()

    def _load_full_model(self) -> None:
        """Load full BGE-M3 model using flag_embedding or sentence-transformers."""
        try:
            from flag_embedding import FlagModel

            self.model = FlagModel(
                self.model_name,
                use_fp16=True if self.device == "cpu" else False,
                device=self.device,
            )
            logger.info(f"Loaded BGE-M3 model via flag_embedding: {self.model_name}")
        except ImportError:
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(self.model_name, device=self.device)
                logger.info(f"Loaded BGE-M3 model via sentence-transformers: {self.model_name}")
            except ImportError as e:
                raise ImportError(
                    "Neither flag_embedding nor sentence_transformers available. "
                    "Install with: pip install flag-embedding sentence-transformers"
                ) from e

    def _load_quantized_model(self) -> None:
        """Load quantized GGUF BGE-M3 model using llama-cpp-python."""
        try:
            from llama_cpp import Llama

            # GGUF model path - user should provide full path to .gguf file
            gguf_path = self.model_name
            if not gguf_path.endswith(".gguf"):
                raise ValueError(
                    f"For quantized models, model_name must be a path to .gguf file, "
                    f"got: {gguf_path}"
                )

            self.model = Llama(
                model_path=gguf_path,
                embedding=True,
                n_ctx=8192,
                n_threads=8,
                verbose=False,
            )
            logger.info(f"Loaded quantized GGUF model: {gguf_path}")
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python not available. Install with: pip install llama-cpp-python"
            ) from e

    def _encode_text(self, texts: list[str]) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            texts: List of text strings to encode

        Returns:
            Numpy array of embeddings with shape (len(texts), BGE_M3_DIM)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self.use_quantized:
            # llama-cpp-python embedding
            embeddings = []
            for text in texts:
                emb = self.model.embed(text)
                embeddings.append(emb)
            embeddings = np.array(embeddings, dtype=np.float32)
        else:
            # flag_embedding or sentence-transformers
            if hasattr(self.model, "encode"):
                embeddings = self.model.encode(texts, normalize_embeddings=True)
            else:
                # FlagModel
                embeddings = self.model.encode(texts, batch_size=32)
                # Normalize to unit length
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.clip(norms, a_min=1e-12, a_max=None)

            embeddings = np.array(embeddings, dtype=np.float32)

        return embeddings

    def build_index(
        self,
        documents: list[dict],
        force_rebuild: bool = False,
    ) -> None:
        """Build FAISS index from documents.

        Args:
            documents: List of dicts with 'id' and 'text' keys
            force_rebuild: Whether to rebuild index even if cached
        """
        import faiss

        if not documents:
            logger.warning("No documents provided for index building")
            return

        # Extract document ids, texts, and metadata
        self.doc_ids = [doc.get("id", f"doc_{i}") for i, doc in enumerate(documents)]
        texts = [doc.get("text", "") for doc in documents]
        self.doc_metadata = [
            {k: v for k, v in doc.items() if k not in ("id", "text")} for doc in documents
        ]

        # Generate embeddings
        logger.info(f"Encoding {len(texts)} documents...")
        embeddings = self._encode_text(texts)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Build FAISS index based on preset
        dim = embeddings.shape[1]

        if self.index_preset == "quality":
            self.index = faiss.IndexFlatIP(dim)
        elif self.index_preset == "balanced":
            self.index = faiss.IndexHNSWFlat(dim, 16)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 50
        elif self.index_preset == "minimal":
            nlist = min(100, len(embeddings) // 4)
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFPQ(quantizer, dim, nlist, 32, 8)
            # Train the index
            logger.info("Training IndexIVFPQ...")
            self.index.train(embeddings)
        else:
            raise ValueError(f"Unknown index preset: {self.index_preset}")

        # Add embeddings to index
        logger.info(f"Adding {len(embeddings)} vectors to index...")
        self.index.add(embeddings)
        logger.info(f"Index built successfully with {self.index.ntotal} vectors")

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict]:
        """Search for similar documents.

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of dicts with 'id', 'score', 'metadata' keys
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Encode query
        query_embedding = self._encode_text([query])
        import faiss

        faiss.normalize_L2(query_embedding)

        # Search
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, top_k)

        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue
            results.append(
                {
                    "id": self.doc_ids[idx],
                    "score": float(score),
                    "metadata": self.doc_metadata[idx] if idx < len(self.doc_metadata) else {},
                }
            )

        return results

    def save(self, path: str | Path) -> None:
        """Save index and metadata to disk.

        Args:
            path: Directory path to save index
        """
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.index is None:
            raise RuntimeError("No index to save. Build index first.")

        # Save FAISS index
        index_path = path / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))

        # Save metadata
        metadata = {
            "doc_ids": self.doc_ids,
            "doc_metadata": self.doc_metadata,
            "index_preset": self.index_preset,
            "model_name": self.model_name,
            "use_quantized": self.use_quantized,
        }
        metadata_path = path / "metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Saved index and metadata to {path}")

    def load(self, path: str | Path) -> bool:
        """Load index and metadata from disk.

        Args:
            path: Directory path to load index from

        Returns:
            True if loaded successfully, False otherwise
        """
        import faiss

        path = Path(path)

        index_path = path / "faiss_index.bin"
        metadata_path = path / "metadata.pkl"

        if not index_path.exists() or not metadata_path.exists():
            logger.warning(f"Index or metadata file not found at {path}")
            return False

        try:
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))

            # Load metadata
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)

            self.doc_ids = metadata["doc_ids"]
            self.doc_metadata = metadata["doc_metadata"]
            self.index_preset = metadata.get("index_preset", "balanced")
            self.model_name = metadata.get("model_name", "BAAI/bge-m3")
            self.use_quantized = metadata.get("use_quantized", False)

            logger.info(f"Loaded index and metadata from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load index from {path}: {e}")
            return False
