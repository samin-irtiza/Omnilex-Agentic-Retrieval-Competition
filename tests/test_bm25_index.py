"""Unit tests for BM25Index tokenization with/without German stemming."""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilex.retrieval.bm25_index import GERMAN_STOPWORDS, BM25Index


class TestTokenization:
    """Test tokenization with and without German stemming."""

    def test_tokenize_without_stemming(self):
        """Test basic tokenization without stemming."""
        index = BM25Index(use_german_stemming=False)
        text = "Das ist ein Test für die Tokenisierung."
        tokens = index.tokenize(text)
        assert "das" in tokens
        assert "ist" in tokens
        assert "ein" in tokens
        assert "test" in tokens
        assert "für" in tokens
        assert "die" in tokens
        assert "tokenisierung" in tokens

    def test_tokenize_with_stemming(self):
        """Test tokenization with German stemming enabled."""
        index = BM25Index(use_german_stemming=True)
        text = "Gesetze und Verordnungen"
        tokens = index.tokenize(text)
        # "Gesetze" should be stemmed to "gesetz"
        assert "gesetz" in tokens
        # "Verordnungen" should be stemmed to "verordn" (German Snowball stemming)
        assert "verordn" in tokens
        # "und" is a stopword, should be removed
        assert "und" not in tokens

    def test_tokenize_removes_german_stopwords(self):
        """Test that German stopwords are removed when stemming is enabled."""
        index = BM25Index(use_german_stemming=True)
        stopwords_sample = ["und", "oder", "aber", "die", "der", "das"]
        text = " ".join(stopwords_sample)
        tokens = index.tokenize(text)
        # All these should be removed
        for word in stopwords_sample:
            assert word not in tokens, f"Stopword '{word}' was not removed"

    def test_tokenize_discards_single_char_tokens(self):
        """Test that single-character tokens are discarded."""
        index = BM25Index(use_german_stemming=True)
        text = "a b c groß d e f"
        tokens = index.tokenize(text)
        # Single char tokens should be removed
        assert "a" not in tokens
        assert "b" not in tokens
        assert "c" not in tokens
        assert "d" not in tokens
        assert "e" not in tokens
        assert "f" not in tokens
        # "groß" should be present (and stemmed)
        assert "groß" in tokens or "gross" in tokens

    def test_tokenize_lowercase(self):
        """Test that all tokens are lowercased."""
        index = BM25Index(use_german_stemming=False)
        text = "GROSS geschrieben"
        tokens = index.tokenize(text)
        assert "gross" in tokens
        assert "geschrieben" in tokens
        # No uppercase tokens should remain
        for token in tokens:
            assert token == token.lower(), f"Token '{token}' is not lowercase"

    def test_tokenize_split_on_non_alphanumeric(self):
        """Test that text is split on non-alphanumeric characters."""
        index = BM25Index(use_german_stemming=False)
        text = "Hallo, Welt! Wie geht's?"
        tokens = index.tokenize(text)
        assert "hallo" in tokens
        assert "welt" in tokens
        assert "wie" in tokens
        assert "geht" in tokens
        assert "s" not in tokens  # "geht's" -> "geht" and "s" (removed as single char)

    def test_tokenize_empty_string(self):
        """Test tokenization of empty string."""
        index = BM25Index(use_german_stemming=True)
        tokens = index.tokenize("")
        assert tokens == []

    def test_tokenize_only_stopwords(self):
        """Test tokenization when text contains only stopwords."""
        index = BM25Index(use_german_stemming=True)
        text = "und oder aber"
        tokens = index.tokenize(text)
        assert tokens == []

    def test_stemming_accuracy(self):
        """Test stemming produces expected German tokens per spec."""
        index = BM25Index(use_german_stemming=True)
        # Test cases: "Gesetz und Verordnungen" -> ["gesetz", "verordn"]
        # Note: German Snowball stemmer stems "Verordnungen" to "verordn"
        text = "Gesetz und Verordnungen"
        tokens = index.tokenize(text)
        assert "gesetz" in tokens
        assert "verordn" in tokens
        assert len(tokens) == 2  # "und" is stopword, removed

    def test_german_stopwords_count(self):
        """Test that we have the expected number of German stopwords."""
        # Paper specifies 86, but we use inflected forms (138) for pre-stemming removal
        # After stemming, these collapse to ~93 unique forms
        assert len(GERMAN_STOPWORDS) == 138

    def test_tokenize_with_stemming_disabled_keeps_stopwords(self):
        """Test that stopwords are NOT removed when stemming is disabled."""
        index = BM25Index(use_german_stemming=False)
        text = "und oder aber die der das"
        tokens = index.tokenize(text)
        # Without stemming, stopwords should NOT be removed
        assert "und" in tokens
        assert "oder" in tokens
        assert "aber" in tokens


class TestBM25IndexBuild:
    """Test BM25Index build with stemming."""

    def test_build_with_stemming(self):
        """Test that index can be built with stemming enabled."""
        documents = [
            {"text": "Gesetz über die Verordnung", "citation": "SR 123.1"},
            {"text": "Verordnung zum Gesetz", "citation": "SR 123.2"},
        ]
        index = BM25Index(documents=documents, use_german_stemming=True)
        assert index.index is not None
        # Search should work with stemmed tokens
        results = index.search("gesetz", top_k=5)
        assert len(results) > 0

    def test_build_without_stemming(self):
        """Test that index can be built without stemming."""
        documents = [
            {"text": "Gesetz über die Verordnung", "citation": "SR 123.1"},
            {"text": "Verordnung zum Gesetz", "citation": "SR 123.2"},
        ]
        index = BM25Index(documents=documents, use_german_stemming=False)
        assert index.index is not None
        results = index.search("Gesetz", top_k=5)
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
