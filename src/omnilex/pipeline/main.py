"""Main retrieval pipeline for Swiss law citation retrieval."""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional

from ..retrieval.bm25_index import BM25Index, load_jsonl_corpus
from ..retrieval.dense_index import DenseIndex
from ..retrieval.fusion import rrf_fusion
from ..retrieval.citation_extractor import extract_from_results
from ..pipeline.verifier import CitationVerifier, build_citation_set


class RetrievalPipeline:
    """End-to-end retrieval pipeline.
    
    Combines BM25 + dense retrieval with RRF fusion,
    citation extraction, and verification.
    """

    def __init__(
        self,
        bm25_index: Optional[BM25Index] = None,
        dense_index: Optional[DenseIndex] = None,
        verifier: Optional[CitationVerifier] = None,
        corpus: Optional[List[Dict]] = None,
    ):
        """Initialize pipeline.
        
        Args:
            bm25_index: BM25 sparse index
            dense_index: Dense FAISS index
            verifier: Citation verifier
            corpus: Original document list
        """
        self.bm25_index = bm25_index
        self.dense_index = dense_index
        self.verifier = verifier
        self.corpus = corpus or []

    @classmethod
    def from_files(
        cls,
        data_dir: str = "data/raw",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> "RetrievalPipeline":
        """Build pipeline from data files.
        
        Args:
            data_dir: Directory containing data files
            model_name: Sentence transformer model name
            
        Returns:
            Configured RetrievalPipeline
        """
        data_dir = Path(data_dir)
        
        # Load corpus
        print("Loading documents...")
        federal_laws = load_jsonl_corpus(data_dir / "federal_laws.jsonl")
        court_decisions = load_jsonl_corpus(data_dir / "court_decisions.jsonl")
        corpus = federal_laws + court_decisions
        print(f"Total documents: {len(corpus)}")
        
        # Build BM25 index
        print("Building BM25 index...")
        bm25_index = BM25Index(documents=corpus)
        
        # Build dense index
        print("Building dense index...")
        dense_index = DenseIndex(documents=corpus, model_name=model_name)
        
        # Build verifier
        print("Building citation verifier...")
        verifier = CitationVerifier()
        verifier.build_from_corpus(corpus)
        
        return cls(
            bm25_index=bm25_index,
            dense_index=dense_index,
            verifier=verifier,
            corpus=corpus,
        )

    def search(
        self,
        query: str,
        top_k_results: int = 50,
        top_k_citations: int = 10,
    ) -> Tuple[List[str], float]:
        """Search for citations.
        
        Args:
            query: Search query
            top_k_results: Number of documents to retrieve
            top_k_citations: Number of citations to extract
            
        Returns:
            Tuple of (verified_citations, hallucination_rate)
        """
        # BM25 search
        bm25_results = []
        if self.bm25_index:
            bm25_results = self.bm25_index.search(
                query, top_k=top_k_results, return_scores=True
            )
            bm25_results = [(r['_index'], r.get('_score', 0)) for r in bm25_results]
        
        # Dense search
        dense_results = []
        if self.dense_index:
            dense_results = self.dense_index.search(query, top_k=top_k_results)
        
        # RRF fusion
        combined_results = rrf_fusion([bm25_results, dense_results])
        
        # Extract citations from top results
        citation_list = extract_from_results(
            combined_results,
            self.corpus,
            top_k=top_k_citations,
        )
        
        # Verify citations
        verified_citations = []
        hall_rate = 0.0
        
        if self.verifier:
            verified_citations = self.verifier.verify(citation_list)
            hall_rate = self.verifier.hallucination_rate(citation_list)
        else:
            verified_citations = citation_list
        
        return verified_citations, hall_rate

    def process_queries(
        self,
        queries: List[str],
        query_ids: Optional[List[str]] = None,
        top_k_results: int = 50,
        top_k_citations: int = 10,
    ) -> pd.DataFrame:
        """Process multiple queries.
        
        Args:
            queries: List of query strings
            query_ids: Optional list of query IDs
            top_k_results: Number of documents to retrieve
            top_k_citations: Number of citations to extract
            
        Returns:
            DataFrame with query_id, predicted_citations, hallucination_rate
        """
        results = []
        
        for i, query in enumerate(queries):
            citations, hall_rate = self.search(
                query,
                top_k_results=top_k_results,
                top_k_citations=top_k_citations,
            )
            
            row = {
                'predicted_citations': ';'.join(citations),
                'hallucination_rate': hall_rate,
            }
            if query_ids:
                row['query_id'] = query_ids[i]
            
            results.append(row)
        
        df = pd.DataFrame(results)
        if query_ids:
            df = df[['query_id', 'predicted_citations', 'hallucination_rate']]
        
        return df

    def predict_test(
        self,
        test_csv: str,
        output_csv: str = "submission.csv",
        top_k_results: int = 50,
        top_k_citations: int = 10,
    ) -> None:
        """Run pipeline on test set.
        
        Args:
            test_csv: Path to test CSV
            output_csv: Path to save submission
            top_k_results: Number of documents to retrieve
            top_k_citations: Number of citations to extract
        """
        test_df = pd.read_csv(test_csv)
        
        print(f"Processing {len(test_df)} test queries...")
        
        predictions = self.process_queries(
            queries=test_df['query'].tolist(),
            query_ids=test_df['query_id'].tolist(),
            top_k_results=top_k_results,
            top_k_citations=top_k_citations,
        )
        
        # Save submission
        submission = predictions[['query_id', 'predicted_citations']]
        submission.to_csv(output_csv, index=False)
        print(f"Saved to {output_csv}")


def run_pipeline(
    data_dir: str = "data/raw",
    test_csv: str = "data/raw/test.csv",
    output_csv: str = "submission.csv",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k_results: int = 50,
    top_k_citations: int = 10,
) -> None:
    """Run full pipeline.
    
    Args:
        data_dir: Directory containing data files
        test_csv: Path to test CSV
        output_csv: Path to save submission
        model_name: Sentence transformer model name
        top_k_results: Number of documents to retrieve
        top_k_citations: Number of citations to extract
    """
    # Build pipeline
    pipeline = RetrievalPipeline.from_files(data_dir, model_name)
    
    # Run on test set
    pipeline.predict_test(
        test_csv=test_csv,
        output_csv=output_csv,
        top_k_results=top_k_results,
        top_k_citations=top_k_citations,
    )


__all__ = [
    "RetrievalPipeline",
    "run_pipeline",
]