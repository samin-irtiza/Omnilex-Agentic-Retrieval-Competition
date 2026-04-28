"""Test script to verify the BM25 memory-efficient implementations.

Run this to test both implementations and check memory usage.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_sparse_bm25():
    """Test BM25SparseIndex with sample documents."""
    print("=" * 60)
    print("Testing BM25SparseIndex (CPU, memory-efficient)")
    print("=" * 60)
    
    from omnilex.retrieval.bm25_index_gpu import BM25SparseIndex
    
    # Create sample documents
    documents = [
        {"text": "The court decided that the law applies to all citizens.", "citation": "Art. 1", "id": "doc1"},
        {"text": "According to the constitution, every person has rights.", "citation": "Art. 2", "id": "doc2"},
        {"text": "The federal court ruled on the matter of taxation.", "citation": "Art. 3", "id": "doc3"},
        {"text": "Citizens must pay taxes according to federal law.", "citation": "Art. 4", "id": "doc4"},
        {"text": "The law court has jurisdiction over tax matters.", "citation": "Art. 5", "id": "doc5"},
    ]
    
    print(f"\nBuilding index with {len(documents)} sample documents...")
    
    try:
        index = BM25SparseIndex(
            documents=documents,
            text_field="text",
            citation_field="citation",
            store_documents=False,
            use_gpu=False,
            vocab_size=1000,
        )
        
        print(f"Index built successfully!")
        print(f"Vocabulary size: {len(index.vocab)}")
        print(f"TF matrix shape: {index.tf_matrix.shape}")
        print(f"TF matrix nnz: {index.tf_matrix.nnz}")
        
        # Test search
        print("\nTesting search...")
        query = "court law tax"
        results = index.search(query, top_k=3, return_scores=True)
        
        print(f"\nQuery: '{query}'")
        print(f"Results:")
        for r in results:
            print(f"  - {r.get('citation', 'N/A')}: score={r.get('_score', 0):.4f}")
        
        print("\n✓ BM25SparseIndex test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n✗ BM25SparseIndex test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_faiss_bm25():
    """Test BM25FAISSIndex with sample documents."""
    print("\n" + "=" * 60)
    print("Testing BM25FAISSIndex (GPU-accelerated, if available)")
    print("=" * 60)
    
    try:
        from omnilex.retrieval.bm25_faiss_gpu import BM25FAISSIndex
        
        # Create sample documents
        documents = [
            {"text": "The court decided that the law applies to all citizens.", "citation": "Art. 1", "id": "doc1"},
            {"text": "According to the constitution, every person has rights.", "citation": "Art. 2", "id": "doc2"},
            {"text": "The federal court ruled on the matter of taxation.", "citation": "Art. 3", "id": "doc3"},
            {"text": "Citizens must pay taxes according to federal law.", "citation": "Art. 4", "id": "doc4"},
            {"text": "The law court has jurisdiction over tax matters.", "citation": "Art. 5", "id": "doc5"},
        ]
        
        print(f"\nBuilding index with {len(documents)} sample documents...")
        
        index = BM25FAISSIndex(
            documents=documents,
            text_field="text",
            citation_field="citation",
            store_documents=False,
            vocab_size=1000,
            use_gpu=False,  # Set to True if GPU available
        )
        
        print(f"Index built successfully!")
        print(f"FAISS index size: {index.faiss_index.ntotal if index.faiss_index else 0}")
        
        # Test search
        print("\nTesting search...")
        query = "court law tax"
        results = index.search(query, top_k=3, return_scores=True)
        
        print(f"\nQuery: '{query}'")
        print(f"Results:")
        for r in results:
            print(f"  - {r.get('citation', 'N/A')}: score={r.get('_score', 0):.4f}")
        
        print("\n✓ BM25FAISSIndex test PASSED!")
        return True
        
    except ImportError as e:
        print(f"\n⚠ BM25FAISSIndex test SKIPPED: {e}")
        print("Install FAISS with: pip install faiss-gpu (or faiss-cpu)")
        return False
    except Exception as e:
        print(f"\n✗ BM25FAISSIndex test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_memory_usage():
    """Check current memory usage."""
    print("\n" + "=" * 60)
    print("Memory Usage Check")
    print("=" * 60)
    
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"Total RAM: {mem.total / 1e9:.1f} GB")
        print(f"Available RAM: {mem.available / 1e9:.1f} GB")
        print(f"Used RAM: {mem.used / 1e9:.1f} GB ({mem.percent}%)")
    except ImportError:
        print("psutil not available. Install with: pip install psutil")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BM25 Memory-Efficient Implementations - Test Suite")
    print("=" * 60)
    
    # Check memory
    check_memory_usage()
    
    # Run tests
    results = []
    results.append(("BM25SparseIndex", test_sparse_bm25()))
    
    # Only test FAISS if available
    try:
        import faiss
        results.append(("BM25FAISSIndex", test_faiss_bm25()))
    except ImportError:
        print("\n⚠ FAISS not available, skipping FAISS test")
        print("Install with: pip install faiss-gpu")
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
    
    print("\n" + "=" * 60)
    print("Recommendation for 12GB RAM + T4 GPU:")
    print("=" * 60)
    print("Use BM25FAISSIndex (if FAISS-GPU available)")
    print("- Uses T4's 15GB VRAM for vector storage")
    print("- Very low RAM usage (~500MB)")
    print("- Fast GPU-accelerated search")
    print("\nFallback: Use BM25SparseIndex")
    print("- CPU-only, uses sparse matrices")
    print("- Low RAM usage (~3GB for 2.6M docs)")
    print("=" * 60)
