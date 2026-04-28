"""Citation verification to filter hallucinated citations."""

from typing import List, Set, Dict, Tuple
import json
import re

from .citation_extractor import extract_from_documents


class CitationVerifier:
    """Verify citations against corpus to remove hallucinated ones.
    
    Builds a set of valid citations from the corpus and filters
    predicted citations to only include valid ones.
    """

    def __init__(self):
        """Initialize verifier."""
        self.valid_citations: Set[str] = set()
        self._built = False

    def build_from_corpus(
        self,
        documents: List[Dict],
        text_field: str = "text",
        citation_field: str = "citation",
    ) -> None:
        """Build valid citation set from corpus.
        
        Args:
            documents: List of document dictionaries
            text_field: Key for document text
            citation_field: Key for existing citation in doc
        """
        print(f"Building citation set from {len(documents)} documents...")
        
        for doc in documents:
            # Try citation field first
            if citation_field in doc:
                cit = doc.get(citation_field, '')
                if cit and isinstance(cit, str):
                    # Normalize citation
                    normalized = self._normalize(cit)
                    if normalized:
                        self.valid_citations.add(normalized)
            
            # Then try text field
            text = doc.get(text_field, '')
            if text:
                # Use extractor
                from .citation_extractor import extract_citations
                extracted = extract_citations(text)
                for cit in extracted:
                    normalized = self._normalize(cit)
                    if normalized:
                        self.valid_citations.add(normalized)
        
        self._built = True
        print(f"Valid citations: {len(self.valid_citations)}")

    def build_from_jsonl(self, path: str) -> None:
        """Build valid citation set from JSONL file.
        
        Args:
            path: Path to JSONL file
        """
        print(f"Loading citations from {path}...")
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    doc = json.loads(line)
                    
                    # Citation field
                    if 'citation' in doc:
                        cit = doc.get('citation', '')
                        if cit:
                            normalized = self._normalize(cit)
                            if normalized:
                                self.valid_citations.add(normalized)
                    
                    # Text field
                    text = doc.get('text', '')
                    if text:
                        from .citation_extractor import extract_citations
                        extracted = extract_citations(text)
                        for cit in extracted:
                            normalized = self._normalize(cit)
                            if normalized:
                                self.valid_citations.add(normalized)
        
        self._built = True
        print(f"Valid citations: {len(self.valid_citations)}")

    def _normalize(self, citation: str) -> str:
        """Normalize citation to canonical form.
        
        Args:
            citation: Raw citation string
            
        Returns:
            Normalized citation
        """
        if not citation:
            return ""
        
        citation = citation.strip()
        
        # Common normalizations
        # Remove "Art." variations
        citation = re.sub(r'^Art\.?\s*', 'Art. ', citation, flags=re.IGNORECASE)
        
        # Standardize spaces
        citation = re.sub(r'\s+', ' ', citation)
        
        return citation

    def is_valid(self, citation: str) -> bool:
        """Check if citation is valid.
        
        Args:
            citation: Citation to check
            
        Returns:
            True if citation exists in corpus
        """
        if not self._built:
            raise ValueError("Citation set not built. Call build_from_corpus() first.")
        
        normalized = self._normalize(citation)
        return normalized in self.valid_citations

    def verify(self, citations: List[str]) -> List[str]:
        """Filter to only valid citations.
        
        Args:
            citations: List of predicted citations
            
        Returns:
            List of valid citations
        """
        if not self._built:
            raise ValueError("Citation set not built. Call build_from_corpus() first.")
        
        valid = []
        for cit in citations:
            normalized = self._normalize(cit)
            if normalized and normalized in self.valid_citations:
                valid.append(normalized)
        
        return valid

    def filter_with_metadata(
        self,
        citations: List[str]
    ) -> Tuple[List[str], List[Dict]]:
        """Filter and return metadata about filtered citations.
        
        Args:
            citations: List of predicted citations
            
        Returns:
            Tuple of (valid_citations, metadata_list)
        """
        if not self._built:
            raise ValueError("Citation set not built. Call build_from_corpus() first.")
        
        valid = []
        metadata = []
        
        for cit in citations:
            normalized = self._normalize(cit)
            is_valid = normalized in self.valid_citations if normalized else False
            
            metadata.append({
                'original': cit,
                'normalized': normalized,
                'valid': is_valid,
            })
            
            if is_valid:
                valid.append(normalized)
        
        return valid, metadata

    def hallucination_rate(self, citations: List[str]) -> float:
        """Compute hallucination rate.
        
        Args:
            citations: List of predicted citations
            
        Returns:
            Rate of invalid citations (0.0 to 1.0)
        """
        if not citations:
            return 0.0
        
        if not self._built:
            raise ValueError("Citation set not built. Call build_from_corpus() first.")
        
        invalid = sum(1 for cit in citations if not self.is_valid(cit))
        return invalid / len(citations)

    def get_stats(self) -> Dict:
        """Get statistics about the citation set.
        
        Returns:
            Dictionary with stats
        """
        if not self._built:
            return {"built": False}
        
        return {
            "built": True,
            "total_citations": len(self.valid_citations),
            "articles": len([c for c in self.valid_citations if c.startswith("Art.")]),
            "court_decisions": len([c for c in self.valid_citations if c.startswith("BGE")]),
        }


def verify_citations(
    citations: List[str],
    valid_set: Set[str],
) -> List[str]:
    """Verify citations against valid set.
    
    Args:
        citations: List of predicted citations
        valid_set: Set of valid citations
        
    Returns:
        List of valid citations
    """
    valid = []
    for cit in citations:
        cit = cit.strip()
        if cit and cit in valid_set:
            valid.append(cit)
    return valid


def build_citation_set(
    documents: List[Dict],
    text_field: str = "text",
    citation_field: str = "citation",
) -> Set[str]:
    """Build set of all valid citations.
    
    Args:
        documents: List of document dictionaries
        text_field: Key for document text
        citation_field: Key for citation in doc
        
    Returns:
        Set of valid citation strings
    """
    from .citation_extractor import extract_from_documents
    return set(extract_from_documents(documents, text_field, citation_field))


def compute_hallucination_rate(
    predicted: List[str],
    valid_set: Set[str],
) -> float:
    """Compute hallucination rate.
    
    Args:
        predicted: List of predicted citations
        valid_set: Set of valid citations
        
    Returns:
        Hallucination rate (0.0 to 1.0)
    """
    if not predicted:
        return 0.0
    
    invalid = sum(1 for c in predicted if c not in valid_set)
    return invalid / len(predicted)


__all__ = [
    "CitationVerifier",
    "verify_citations",
    "build_citation_set",
    "compute_hallucination_rate",
]