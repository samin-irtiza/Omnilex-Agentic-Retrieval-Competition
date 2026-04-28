"""Updated BM25 section for the notebook - Memory-efficient BM25 with Sparse Matrices.

Replace the BM25 cell in the notebook (around cell with FORCE_BM25) with this code.
This uses the new BM25SparseIndex that works within 12GB RAM and can use GPU.
"""

# === BM25 Index (Memory-Efficient with Sparse Matrices) ===
# This implementation uses scipy sparse matrices to reduce RAM usage
# and can leverage GPU (T4 15GB VRAM) for computations

# Import the memory-efficient BM25 implementation
from omnilex.retrieval.bm25_index_gpu import BM25SparseIndex, build_sparse_index

bm25_index = None  # Default to None

# Configuration for memory-efficient BM25
BM25_USE_GPU = GPU_AVAILABLE  # Use GPU if available (T4 15GB VRAM)
BM25_VOCAB_SIZE = 50000  # Limit vocabulary to control memory usage
BM25_CACHE_PATH = PROCESSED_DATA_DIR / "bm25_sparse_index"

print("=" * 60)
print("BM25 Sparse Index Configuration")
print("=" * 60)
print(f"GPU Available: {GPU_AVAILABLE}")
print(f"BM25 Use GPU: {BM25_USE_GPU}")
print(f"Vocab Size: {BM25_VOCAB_SIZE}")
print(f"Cache Path: {BM25_CACHE_PATH}")
print("=" * 60)

# Try to load cached index or build new one
if CACHE_INDICES and BM25_CACHE_PATH.with_suffix('.pkl').exists():
    print(f"\nLoading cached BM25 Sparse Index from {BM25_CACHE_PATH}...")
    try:
        bm25_index = BM25SparseIndex.load(BM25_CACHE_PATH.with_suffix('.pkl'))
        print(f"Loaded BM25 index with {len(bm25_index.doc_ids)} documents")
        print(f"Vocabulary size: {len(bm25_index.vocab)} tokens")
    except Exception as e:
        print(f"Error loading cached index: {e}")
        print("Will build new index...")
        bm25_index = None

if bm25_index is None:
    print("\nBuilding BM25 Sparse Index (memory-efficient)...")
    print("This uses sparse matrices to reduce RAM usage.")
    print(f"Processing {len(all_documents)} documents...")
    
    try:
        # Check available memory
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        print(f"Available RAM: {available_gb:.1f} GB")
        
        if available_gb < 4.0:
            print("WARNING: Less than 4GB RAM available. Consider closing other processes.")
        
        # Build sparse index
        # Note: store_documents=False to save memory (only store IDs)
        bm25_index = build_sparse_index(
            documents=all_documents,
            text_field="text",
            citation_field="citation",
            use_gpu=BM25_USE_GPU,
            vocab_size=BM25_VOCAB_SIZE,
        )
        
        print(f"\nBM25 Sparse Index built successfully!")
        print(f"TF Matrix shape: {bm25_index.tf_matrix.shape}")
        print(f"TF Matrix nnz: {bm25_index.tf_matrix.nnz}")
        print(f"Memory usage (TF matrix): ~{bm25_index.tf_matrix.data.nbytes / 1e6:.1f} MB")
        
        # Cache the index
        if CACHE_INDICES:
            bm25_index.save(BM25_CACHE_PATH.with_suffix('.pkl'))
            print(f"Index cached to {BM25_CACHE_PATH}")
        
        # Clear all_documents from memory after building index
        # (if not needed for other purposes)
        print("\nClearing all_documents from memory to save RAM...")
        all_documents = None
        import gc
        gc.collect()
        if GPU_AVAILABLE:
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error building BM25 index: {e}")
        print("Falling back to Dense-only retrieval.")
        bm25_index = None

print(f"\nBM25 index: {'ENABLED' if bm25_index else 'DISABLED'}")

# Update the retrieve_with_bm25 function to use the sparse index
def retrieve_with_bm25(query: str, top_k: int = BM25_TOP_K) -> List[Tuple[int, float]]:
    """Retrieve documents using BM25 keyword search (memory-efficient).
    
    Args:
        query: Search query string
        top_k: Number of results to return
        
    Returns:
        List of (document_index, score) tuples
    """
    if bm25_index is None:
        print("WARNING: BM25 index not available. Returning empty results.")
        return []
    
    try:
        results = bm25_index.search(query, top_k=top_k, return_scores=True)
        
        # Convert to (doc_index, score) tuples
        output = []
        for doc in results:
            doc_index = doc.get("_index")
            if doc_index is not None:
                output.append((doc_index, doc.get("_score", 0.0)))
        return output
    except Exception as e:
        print(f"Error in BM25 search: {e}")
        return []

print("BM25 retrieve function updated for sparse index.")
