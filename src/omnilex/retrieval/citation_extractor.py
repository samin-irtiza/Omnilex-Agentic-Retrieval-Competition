"""Citation extraction from legal documents."""

import re
from typing import List, Set, Dict, Tuple, Optional

# Citation patterns for Swiss law
ARTICLE_PATTERN = re.compile(
    r'Art\.?\s*(\d+[a-z]?)\s*'
    r'(?:Abs\.?\s*\d+)?\s*'
    r'([A-Z]{2,5})',
    re.IGNORECASE
)

BGE_PATTERN = re.compile(
    r'BGE\s+(\d+)\s+([IVX]+[a-z]?)\s+(\d+)',
    re.IGNORECASE
)

# Court consideration pattern (used in court_considerations.csv)
CONSIDERATION_PATTERN = re.compile(
    r'BGE\s+(\d+)\s+([IVX]+)\s+([E]?)\s*(\d+)',
    re.IGNORECASE
)


class CitationExtractor:
    """Extract legal citations from documents.
    
    Supports Swiss law citation formats:
    - Federal laws: Art. XXX [LAW] (e.g., Art. 10a Abs. 1 USG)
    - Court decisions: BGE XXX I XXX (e.g., BGE 139 I 2)
    """

    def __init__(self):
        """Initialize extractor."""
        self._article_pattern = ARTICLE_PATTERN
        self._bge_pattern = BGE_PATTERN
        self._consideration_pattern = CONSIDERATION_PATTERN

    def extract_citations(self, text: str) -> List[str]:
        """Extract all citations from text.
        
        Args:
            text: Document text
            
        Returns:
            List of unique citation strings
        """
        citations = []
        
        # Extract Article citations
        for match in self._article_pattern.finditer(text):
            article, law = match.groups()
            citation = f"Art. {article} {law}"
            if citation not in citations:
                citations.append(citation)
        
        # Extract BGE citations
        for match in self._bge_pattern.finditer(text):
            volume, section, page = match.groups()
            citation = f"BGE {volume} {section} {page}"
            if citation not in citations:
                citations.append(citation)
        
        return citations

    def extract_articles(self, text: str) -> List[str]:
        """Extract only federal law citations.
        
        Args:
            text: Document text
            
        Returns:
            List of article citations
        """
        citations = []
        
        for match in self._article_pattern.finditer(text):
            article, law = match.groups()
            citation = f"Art. {article} {law}"
            if citation not in citations:
                citations.append(citation)
        
        return citations

    def extract_bge(self, text: str) -> List[str]:
        """Extract only court decision citations.
        
        Args:
            text: Document text
            
        Returns:
            List of BGE citations
        """
        citations = []
        
        for match in self._bge_pattern.finditer(text):
            volume, section, page = match.groups()
            citation = f"BGE {volume} {section} {page}"
            if citation not in citations:
                citations.append(citation)
        
        return citations

    def extract_from_documents(
        self,
        documents: List[Dict],
        text_field: str = "text",
        citation_field: str = "citation",
    ) -> List[str]:
        """Extract all unique citations from documents.
        
        Args:
            documents: List of document dictionaries
            text_field: Key for document text
            citation_field: Key for existing citation in doc
            
        Returns:
            List of unique citation strings
        """
        all_citations = set()
        
        for doc in documents:
            # Try citation field first
            if citation_field in doc:
                cit = doc.get(citation_field, '')
                if cit:
                    all_citations.add(cit)
            
            # Then try text field
            text = doc.get(text_field, '')
            if text:
                extracted = self.extract_citations(text)
                all_citations.update(extracted)
        
        return sorted(list(all_citations))

    def extract_from_results(
        self,
        hybrid_results: List[Tuple[int, float]],
        documents: List[Dict],
        top_k: int = 10,
        text_field: str = "text",
    ) -> List[str]:
        """Extract citations from top-ranked search results.
        
        Args:
            hybrid_results: List of (doc_id, score) tuples
            documents: Full document list
            top_k: Number of documents to examine
            text_field: Key for document text
            
        Returns:
            List of unique citations (deduplicated)
        """
        all_citations = []
        
        for idx, score in hybrid_results[:top_k]:
            if idx < len(documents):
                doc = documents[idx]
                text = doc.get(text_field, '')
                citations = self.extract_citations(text)
                all_citations.extend(citations)
        
        # Deduplicate while preserving order
        seen = set()
        unique_citations = []
        for c in all_citations:
            if c not in seen:
                seen.add(c)
                unique_citations.append(c)
        
        return unique_citations


def extract_citations(text: str) -> List[str]:
    """Extract citations from text.
    
    Args:
        text: Document text
        
    Returns:
        List of unique citation strings
    """
    extractor = CitationExtractor()
    return extractor.extract_citations(text)


def extract_from_documents(
    documents: List[Dict],
    text_field: str = "text",
    citation_field: str = "citation",
) -> List[str]:
    """Extract all unique citations from documents.
    
    Args:
        documents: List of document dictionaries
        text_field: Key for document text
        citation_field: Key for existing citation
        
    Returns:
        List of unique citations
    """
    extractor = CitationExtractor()
    return extractor.extract_from_documents(
        documents, text_field, citation_field
    )


def extract_from_results(
    hybrid_results: List[Tuple[int, float]],
    documents: List[Dict],
    top_k: int = 10,
    text_field: str = "text",
) -> List[str]:
    """Extract citations from top search results.
    
    Args:
        hybrid_results: List of (doc_id, score) tuples
        documents: Full document list
        top_k: Number of documents to examine
        text_field: Key for document text
        
    Returns:
        List of unique citations
    """
    extractor = CitationExtractor()
    return extractor.extract_from_results(
        hybrid_results, documents, top_k, text_field
    )


__all__ = [
    "CitationExtractor",
    "extract_citations",
    "extract_from_documents",
    "extract_from_results",
]