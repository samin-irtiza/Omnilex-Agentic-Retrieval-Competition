"""BM25 indexing and search for legal document corpora."""

import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

# 86 German stopwords from Hybrid GraphRAG paper
GERMAN_STOPWORDS = {
    "ab",
    "aber",
    "alle",
    "allem",
    "allen",
    "aller",
    "alles",
    "als",
    "also",
    "am",
    "an",
    "ander",
    "andere",
    "anderem",
    "anderen",
    "anderer",
    "anderes",
    "auch",
    "auf",
    "aus",
    "bei",
    "bin",
    "bis",
    "bist",
    "da",
    "damit",
    "dann",
    "der",
    "den",
    "des",
    "dem",
    "die",
    "das",
    "daß",
    "dazu",
    "dein",
    "deine",
    "deinem",
    "deinen",
    "deiner",
    "deines",
    "dem",
    "denn",
    "derer",
    "dessen",
    "dich",
    "dies",
    "diese",
    "diesem",
    "diesen",
    "dieser",
    "dieses",
    "dir",
    "doch",
    "dort",
    "du",
    "durch",
    "ein",
    "eine",
    "einem",
    "einen",
    "einer",
    "eines",
    "einmal",
    "er",
    "es",
    "etwas",
    "für",
    "gegen",
    "gewesen",
    "hab",
    "habe",
    "haben",
    "hat",
    "hatte",
    "hatten",
    "hier",
    "hin",
    "hinter",
    "ich",
    "mich",
    "mir",
    "mit",
    "nach",
    "nicht",
    "noch",
    "oder",
    "seid",
    "sein",
    "seine",
    "seinem",
    "seinen",
    "seiner",
    "seines",
    "selbst",
    "sich",
    "sie",
    "sind",
    "so",
    "solche",
    "solchem",
    "solchen",
    "solcher",
    "solches",
    "soll",
    "sollen",
    "sollte",
    "sondern",
    "sonst",
    "um",
    "und",
    "uns",
    "unser",
    "unsere",
    "unserem",
    "unseren",
    "unserer",
    "unseres",
    "unter",
    "viel",
    "vom",
    "von",
    "vor",
    "während",
    "wieder",
    "will",
    "wir",
    "wird",
    "wirst",
    "wo",
    "wollen",
    "wollte",
    "würde",
    "würden",
    "zu",
    "zum",
    "zur",
    "zwar",
    "zwischen",
}


class BM25Index:
    """BM25 index for keyword search over legal documents.

    Supports Swiss federal laws (SR) and court decisions (BGE).
    """

    def __init__(
        self,
        documents: list[dict] | None = None,
        text_field: str = "text",
        citation_field: str = "citation",
        use_german_stemming: bool = False,
    ):
        """Initialize BM25 index.

        Args:
            documents: List of document dictionaries
            text_field: Key for document text in dict
            citation_field: Key for citation string in dict
            use_german_stemming: Whether to apply German Snowball stemming
        """
        self.text_field = text_field
        self.citation_field = citation_field
        self.use_german_stemming = use_german_stemming

        self.documents: list[dict] = []
        self.index: BM25Okapi | None = None
        self._tokenized_corpus: list[list[str]] = []

        # Initialize German stemmer if needed
        self._stemmer = None
        if use_german_stemming:
            from snowballstemmer import GermanStemmer

            self._stemmer = GermanStemmer()

        if documents:
            self.build(documents)

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text for BM25 indexing.

        Implements 5-step pipeline matching Hybrid GraphRAG paper:
        1. Lowercase conversion
        2. Split on non-alphanumeric characters
        3. Remove German stopwords (if stemming enabled)
        4. Discard single-character tokens
        5. Apply Snowball stemming (if enabled)

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Step 1: Lowercase
        text = text.lower()

        # Step 2: Split on non-alphanumeric characters
        tokens = re.split(r"\W+", text)

        # Filter empty tokens
        tokens = [t for t in tokens if t]

        # Step 3: Remove German stopwords (only when stemming is enabled)
        if self.use_german_stemming and self._stemmer:
            tokens = [t for t in tokens if t not in GERMAN_STOPWORDS]

        # Step 4: Discard single-character tokens
        tokens = [t for t in tokens if len(t) > 1]

        # Step 5: Apply Snowball stemming (if enabled)
        if self.use_german_stemming and self._stemmer:
            tokens = [self._stemmer.stemWord(t) for t in tokens]

        return tokens

    def build(self, documents: list[dict]) -> None:
        """Build BM25 index from documents.

        Args:
            documents: List of document dictionaries
        """
        self.documents = documents

        # Tokenize all documents
        self._tokenized_corpus = []
        for doc in documents:
            text = doc.get(self.text_field, "")
            tokens = self.tokenize(text)
            self._tokenized_corpus.append(tokens)

        # Build BM25 index
        self.index = BM25Okapi(self._tokenized_corpus)

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
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")

        # Tokenize query
        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        # Get BM25 scores
        scores = self.index.get_scores(query_tokens)

        # Get top-k indices (highest scores first)
        if len(scores) <= top_k:
            top_indices = scores.argsort()[::-1]
        else:
            top_k = min(top_k, len(scores))
            top_indices = scores.argsort()[-top_k:][::-1]

        # Build results
        results = []
        for idx in top_indices:
            doc = self.documents[idx].copy()
            if return_scores:
                doc["_score"] = float(scores[idx])
            results.append(doc)

        return results

    def save(self, path: Path | str) -> None:
        """Save index to disk.

        Args:
            path: Path to save index (creates .pkl file)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "documents": self.documents,
            "tokenized_corpus": self._tokenized_corpus,
            "text_field": self.text_field,
            "citation_field": self.citation_field,
            "use_german_stemming": self.use_german_stemming,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path | str) -> "BM25Index":
        """Load index from disk.

        Args:
            path: Path to saved index

        Returns:
            Loaded BM25Index instance
        """
        path = Path(path)

        with open(path, "rb") as f:
            data = pickle.load(f)

        use_german_stemming = data.get("use_german_stemming", False)
        instance = cls(
            text_field=data["text_field"],
            citation_field=data.get("citation_field", "citation"),
            use_german_stemming=use_german_stemming,
        )
        instance.documents = data["documents"]
        instance._tokenized_corpus = data["tokenized_corpus"]
        instance.index = BM25Okapi(instance._tokenized_corpus)

        return instance


def build_index(
    documents: list[dict],
    text_field: str = "text",
    citation_field: str = "citation",
) -> BM25Index:
    """Build a BM25 index from documents.

    Convenience function for quick index creation.

    Args:
        documents: List of document dictionaries
        text_field: Key for document text
        citation_field: Key for citation string

    Returns:
        Built BM25Index
    """
    return BM25Index(
        documents=documents,
        text_field=text_field,
        citation_field=citation_field,
    )


def search(
    index: BM25Index,
    query: str,
    top_k: int = 10,
) -> list[dict]:
    """Search an index with a query.

    Convenience function for quick search.

    Args:
        index: BM25Index to search
        query: Search query string
        top_k: Number of results

    Returns:
        List of matching documents
    """
    return index.search(query, top_k=top_k)


def load_jsonl_corpus(path: Path | str) -> list[dict]:
    """Load a corpus from a JSONL file.

    Args:
        path: Path to JSONL file

    Returns:
        List of document dictionaries
    """
    path = Path(path)
    documents = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))

    return documents


def save_jsonl_corpus(documents: list[dict], path: Path | str) -> None:
    """Save a corpus to a JSONL file.

    Args:
        documents: List of document dictionaries
        path: Path to save JSONL file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def load_corpus_from_csv(
    path: Path | str,
    citation_col: str = "citation",
    text_col: str = "text",
    max_rows: int | None = None,
) -> list[dict]:
    """Load a corpus from a CSV file.

    Args:
        path: Path to CSV file
        citation_col: Column name for citation field
        text_col: Column name for text field
        max_rows: Maximum number of rows to load (for testing)

    Returns:
        List of document dictionaries with 'citation' and 'text' keys
    """
    import csv

    path = Path(path)
    documents = []

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            documents.append(
                {
                    "citation": row.get(citation_col, ""),
                    "text": row.get(text_col, ""),
                }
            )

    return documents
