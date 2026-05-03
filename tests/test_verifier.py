"""Tests for LLM verifier module."""

from __future__ import annotations

import re
from unittest.mock import MagicMock

from src.omnilex.retrieval.verifier import SCORE_PATTERN, LLMVerifier


class TestLLMVerifierInit:
    """Tests for LLMVerifier initialization."""

    def test_init_default(self):
        """Test default initialization."""
        verifier = LLMVerifier()
        assert verifier.model_path == "qwen2.5-7b-q4_k_m.gguf"
        assert verifier.n_ctx == 8192
        assert verifier.n_threads == 8
        assert verifier.llm is None

    def test_init_custom(self):
        """Test custom initialization."""
        verifier = LLMVerifier(
            model_path="custom.gguf",
            n_ctx=4096,
            n_threads=4,
        )
        assert verifier.model_path == "custom.gguf"
        assert verifier.n_ctx == 4096
        assert verifier.n_threads == 4


class TestLoadModel:
    """Tests for model loading."""

    def test_load_model_fallback_on_missing_file(self):
        """Test model loading falls back when file not found."""
        verifier = LLMVerifier(model_path="/nonexistent/path.gguf")
        verifier.load_model()
        assert verifier.llm is None

    def test_load_model_fallback_on_import_error(self):
        """Test model loading falls back on import error."""
        # Since llama_cpp may not be installed, just verify the method
        # handles errors gracefully and sets llm to None
        verifier = LLMVerifier()
        verifier.load_model()
        # llm should be None if llama_cpp is not available
        # (which is the case in test environment)
        assert verifier.llm is None

    def test_load_model_method_exists(self):
        """Test that load_model method exists and is callable."""
        verifier = LLMVerifier()
        assert hasattr(verifier, "load_model")
        assert callable(verifier.load_model)


class TestVerificationPrompt:
    """Tests for verification prompt formatting."""

    def test_prompt_contains_query(self):
        """Test prompt includes query."""
        verifier = LLMVerifier()
        prompt = verifier.VERIFICATION_PROMPT.format(
            query="test query",
            candidates="1. BGE 127 III 248",
        )
        assert "test query" in prompt
        assert "BGE 127 III 248" in prompt

    def test_prompt_no_generation_instruction(self):
        """Test prompt explicitly forbids citation generation."""
        verifier = LLMVerifier()
        prompt = verifier.VERIFICATION_PROMPT.format(
            query="test",
            candidates="test",
        )
        assert "MUST NOT generate" in prompt


class TestFormatCandidates:
    """Tests for candidate formatting."""

    def test_format_single_candidate(self):
        """Test formatting a single candidate."""
        verifier = LLMVerifier()
        candidates = [{"citation": "BGE 127 III 248"}]
        result = verifier._format_candidates(candidates)
        assert "1. BGE 127 III 248" in result

    def test_format_multiple_candidates(self):
        """Test formatting multiple candidates."""
        verifier = LLMVerifier()
        candidates = [
            {"citation": "BGE 127 III 248"},
            {"citation": "SR 210 Art. 1"},
        ]
        result = verifier._format_candidates(candidates)
        assert "1. BGE 127 III 248" in result
        assert "2. SR 210 Art. 1" in result


class TestParseScores:
    """Tests for score parsing from LLM output."""

    def test_parse_valid_output(self):
        """Test parsing valid LLM output."""
        verifier = LLMVerifier()
        output = """BGE 127 III 248 | 0.95 | Directly relevant
SR 210 Art. 1 | 0.30 | Not relevant"""
        candidates = [
            {"citation": "BGE 127 III 248"},
            {"citation": "SR 210 Art. 1"},
        ]
        result = verifier._parse_scores(output, candidates)
        assert result[0]["verifier_score"] == 0.95
        assert result[1]["verifier_score"] == 0.30

    def test_parse_with_whitespace(self):
        """Test parsing output with extra whitespace."""
        verifier = LLMVerifier()
        output = "BGE 127 III 248   |   0.85   |   Good match   "
        candidates = [{"citation": "BGE 127 III 248"}]
        result = verifier._parse_scores(output, candidates)
        assert result[0]["verifier_score"] == 0.85

    def test_parse_no_match(self):
        """Test parsing when no match found."""
        verifier = LLMVerifier()
        output = "Some unrelated text"
        candidates = [{"citation": "BGE 127 III 248"}]
        result = verifier._parse_scores(output, candidates)
        assert result[0]["verifier_score"] == 0.0


class TestFilterByThreshold:
    """Tests for threshold filtering."""

    def test_filter_above_threshold(self):
        """Test filtering keeps candidates above threshold."""
        verifier = LLMVerifier()
        candidates = [
            {"citation": "A", "verifier_score": 0.8},
            {"citation": "B", "verifier_score": 0.3},
            {"citation": "C", "verifier_score": 0.6},
        ]
        result = verifier.filter_by_threshold(candidates, threshold=0.5)
        assert len(result) == 2
        assert result[0]["citation"] == "A"
        assert result[1]["citation"] == "C"

    def test_filter_default_threshold(self):
        """Test filtering with default threshold."""
        verifier = LLMVerifier()
        candidates = [
            {"citation": "A", "verifier_score": 0.5},
            {"citation": "B", "verifier_score": 0.49},
        ]
        result = verifier.filter_by_threshold(candidates)
        assert len(result) == 1
        assert result[0]["citation"] == "A"


class TestEstimateCitationCount:
    """Tests for citation count estimation."""

    def test_estimate_with_scores(self):
        """Test estimation with score list."""
        verifier = LLMVerifier()
        scores = [0.95, 0.85, 0.70, 0.30, 0.20]
        count = verifier.estimate_citation_count("test query", scores)
        assert 1 <= count <= 5

    def test_estimate_empty_scores(self):
        """Test estimation with empty scores."""
        verifier = LLMVerifier()
        count = verifier.estimate_citation_count("test query", [])
        assert count == 0

    def test_estimate_single_score(self):
        """Test estimation with single score."""
        verifier = LLMVerifier()
        count = verifier.estimate_citation_count("test query", [0.9])
        assert count == 1


class TestHeuristicCitationCount:
    """Tests for heuristic citation count."""

    def test_compare_query(self):
        """Test query with compare keywords."""
        verifier = LLMVerifier()
        count = verifier._heuristic_citation_count("compare these two laws")
        assert count == 4

    def test_analyze_query(self):
        """Test query with analysis keywords."""
        verifier = LLMVerifier()
        count = verifier._heuristic_citation_count("analyze this legal concept")
        assert count == 3

    def test_define_query(self):
        """Test query with definition keywords."""
        verifier = LLMVerifier()
        count = verifier._heuristic_citation_count("what is the definition of")
        assert count == 1


class TestDetectScoreElbow:
    """Tests for elbow detection."""

    def test_elbow_clear_drop(self):
        """Test elbow detection with clear drop."""
        verifier = LLMVerifier()
        scores = [0.95, 0.90, 0.85, 0.40, 0.35, 0.30]
        count = verifier._detect_score_elbow(scores)
        assert count >= 3  # Should detect elbow around index 3

    def test_elbow_no_clear_drop(self):
        """Test elbow detection with gradual drop."""
        verifier = LLMVerifier()
        scores = [0.90, 0.85, 0.80, 0.75, 0.70]
        count = verifier._detect_score_elbow(scores)
        assert 1 <= count <= 5


class TestHeuristicVerify:
    """Tests for heuristic verification fallback."""

    def test_heuristic_with_overlap(self):
        """Test heuristic with keyword overlap."""
        verifier = LLMVerifier()
        candidates = [
            {"citation": "BGE 127 III 248", "text": "contract law damages"},
        ]
        result = verifier.heuristic_verify("contract damages", candidates)
        assert result[0]["verifier_score"] > 0.5

    def test_heuristic_no_overlap(self):
        """Test heuristic with no keyword overlap."""
        verifier = LLMVerifier()
        candidates = [
            {"citation": "BGE 127 III 248", "text": "contract law"},
        ]
        result = verifier.heuristic_verify("tax regulations", candidates)
        assert result[0]["verifier_score"] < 0.5

    def test_heuristic_empty_query_words(self):
        """Test heuristic with empty query."""
        verifier = LLMVerifier()
        candidates = [{"citation": "BGE 127 III 248"}]
        result = verifier.heuristic_verify("", candidates)
        assert result[0]["verifier_score"] == 0.5


class TestVerify:
    """Tests for full verify method."""

    def test_verify_with_llm_mock(self):
        """Test verify with mocked LLM."""
        verifier = LLMVerifier()
        mock_llm = MagicMock()
        mock_llm.return_value = {
            "choices": [{"text": "BGE 127 III 248 | 0.95 | Relevant\n"}],
        }
        verifier.llm = mock_llm

        candidates = [{"citation": "BGE 127 III 248"}]
        result = verifier.verify("test query", candidates)
        assert result[0]["verifier_score"] == 0.95
        mock_llm.assert_called_once()

    def test_verify_fallback_to_heuristic(self):
        """Test verify falls back to heuristic when LLM fails."""
        verifier = LLMVerifier()
        verifier.llm = None  # Force fallback

        candidates = [{"citation": "BGE 127 III 248", "text": "contract"}]
        result = verifier.verify("contract", candidates)
        assert "verifier_score" in result[0]
        assert result[0]["verifier_reason"] == "Heuristic keyword matching"

    def test_verify_empty_candidates(self):
        """Test verify with empty candidates."""
        verifier = LLMVerifier()
        result = verifier.verify("test query", [])
        assert result == []


class TestScorePattern:
    """Tests for score pattern regex."""

    def test_pattern_match(self):
        """Test regex pattern matches expected format."""
        line = "BGE 127 III 248 | 0.95 | Relevant citation"
        match = re.match(SCORE_PATTERN, line)
        assert match is not None
        assert match.group(1).strip() == "BGE 127 III 248"
        assert match.group(2).strip() == "0.95"
        assert match.group(3).strip() == "Relevant citation"

    def test_pattern_no_match(self):
        """Test regex pattern doesn't match invalid format."""
        line = "This is not a valid score line"
        match = re.match(SCORE_PATTERN, line)
        assert match is None
