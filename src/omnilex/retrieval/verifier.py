"""LLM verification module using Qwen2.5-7B (GGUF quantized).

Provides:
- Qwen2.5-7B GGUF loading (using llama-cpp-python)
- Verification-only prompt (never generates citations)
- Verification scoring (parse LLM output for relevance scores)
- Score threshold filtering
- Adaptive citation count estimation
- Fallback to heuristic verification
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Pattern to parse LLM output: <citation> | <score> | <reason>
SCORE_PATTERN = r"(.+?)\s*\|\s*([0-9.]+)\s*\|\s*(.+)"

# Prompt to predict citation count
COUNT_PROMPT = """Based on the query type, predict how many citations would be appropriate.

Query: {query}

Query types:
- Factual/definition queries: 1-2 citations
- Legal analysis questions: 2-4 citations
- Comparative questions: 3-5 citations
- Comprehensive questions: 4-6 citations

Output only a single number (the estimated citation count)."""


class LLMVerifier:
    """LLM-based verifier that scores citations without generating new ones."""

    VERIFICATION_PROMPT = """You are a legal citation verifier. Your task is to
verify whether each candidate citation is relevant to the query.

IMPORTANT: You MUST NOT generate any new citations. Only verify the
provided candidates.

Query: {query}

Candidate citations:
{candidates}

For each candidate, output a single line in this format:
<citation> | <score 0-1> | <brief reason>

Example:
BGE 127 III 248 | 0.95 | Directly addresses contract law question
SR 210 Art. 1 | 0.30 | Not relevant to this query

Scores should be between 0.0 (completely irrelevant) and 1.0 (highly relevant).
"""

    def __init__(
        self,
        model_path: str = "qwen2.5-7b-q4_k_m.gguf",
        n_ctx: int = 8192,
        n_threads: int = 8,
    ):
        """Initialize LLM verifier.

        Args:
            model_path: Path to GGUF model file
            n_ctx: Context window size
            n_threads: Number of CPU threads
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.llm = None

    def load_model(self) -> None:
        """Load Qwen2.5-7B GGUF model using llama-cpp-python.

        Task 6.1: Implement Qwen2.5-7B GGUF loading
        """
        try:
            from llama_cpp import Llama

            logger.info(f"Loading GGUF model from {self.model_path}")
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
            )
            logger.info("Model loaded successfully")
        except ImportError:
            logger.warning("llama-cpp-python not installed. Using heuristic fallback.")
            self.llm = None
        except FileNotFoundError:
            logger.warning(f"Model file not found: {self.model_path}. Using heuristic fallback.")
            self.llm = None
        except Exception as e:
            logger.warning(f"Failed to load model: {e}. Using heuristic fallback.")
            self.llm = None

    def verify(
        self,
        query: str,
        candidates: list[dict],
    ) -> list[dict]:
        """Verify candidate citations using LLM.

        Args:
            query: Query text
            candidates: List of candidate dicts with 'citation' key

        Returns:
            List of candidates with 'verifier_score' added

        Task 6.2-6.3: Implement verification and scoring
        """
        if not candidates:
            return candidates

        # If model not available, use heuristic fallback
        if self.llm is None:
            logger.info("Model not available, using heuristic verification")
            return self.heuristic_verify(query, candidates)

        # Format candidates for prompt
        formatted_candidates = self._format_candidates(candidates)

        # Create prompt
        prompt = self.VERIFICATION_PROMPT.format(
            query=query,
            candidates=formatted_candidates,
        )

        # Call LLM
        try:
            response = self.llm(
                prompt,
                max_tokens=2048,
                temperature=0.1,
                stop=["</s>", "Human:", "Query:"],
            )
            output = response["choices"][0]["text"].strip()
            logger.debug(f"LLM verification output: {output}")
        except Exception as e:
            logger.warning(f"LLM verification failed: {e}. Using heuristic fallback.")
            return self.heuristic_verify(query, candidates)

        # Parse scores from output
        scored_candidates = self._parse_scores(output, candidates)

        return scored_candidates

    def _format_candidates(self, candidates: list[dict]) -> str:
        """Format candidates for LLM prompt."""
        lines = []
        for i, candidate in enumerate(candidates, 1):
            citation = candidate.get("citation", "Unknown")
            lines.append(f"{i}. {citation}")
        return "\n".join(lines)

    def _parse_scores(self, output: str, candidates: list[dict]) -> list[dict]:
        """Parse scores from LLM output.

        Args:
            output: Raw LLM output text
            candidates: Original candidate list

        Returns:
            Candidates with verifier_score added
        """
        pattern = re.compile(SCORE_PATTERN)
        scored_candidates = []

        # Build a mapping from citation text to score
        citation_scores: dict[str, tuple[float, str]] = {}
        for line in output.split("\n"):
            line = line.strip()
            if not line:
                continue
            match = pattern.match(line)
            if match:
                citation_text = match.group(1).strip()
                score = float(match.group(2).strip())
                reason = match.group(3).strip()
                citation_scores[citation_text] = (score, reason)

        # Assign scores to candidates
        for candidate in candidates:
            citation = candidate.get("citation", "")
            scored_candidate = candidate.copy()

            # Try exact match first
            if citation in citation_scores:
                score, reason = citation_scores[citation]
            else:
                # Try partial match
                score = 0.0
                reason = "No match in LLM output"
                for cite_key, (s, r) in citation_scores.items():
                    if citation in cite_key or cite_key in citation:
                        score = s
                        reason = r
                        break

            scored_candidate["verifier_score"] = score
            scored_candidate["verifier_reason"] = reason
            scored_candidates.append(scored_candidate)

        return scored_candidates

    def filter_by_threshold(
        self,
        candidates: list[dict],
        threshold: float = 0.5,
    ) -> list[dict]:
        """Filter candidates by score threshold.

        Args:
            candidates: List of candidates with 'verifier_score'
            threshold: Minimum score to keep

        Returns:
            Filtered list of candidates

        Task 6.4: Implement score threshold filtering
        """
        return [c for c in candidates if c.get("verifier_score", 0) >= threshold]

    def estimate_citation_count(
        self,
        query: str,
        scores: list[float],
    ) -> int:
        """Estimate optimal number of citations using LLM prediction and elbow detection.

        Args:
            query: Query text
            scores: Verification scores sorted by relevance

        Returns:
            Estimated number of citations to include

        Task 6.5: Implement adaptive citation count estimation
        """
        if not scores:
            return 0

        # Method 1: LLM prediction
        llm_count = self._predict_citation_count(query)

        # Method 2: Score elbow detection
        elbow_count = self._detect_score_elbow(scores)

        # Return minimum of the two, capped at len(scores)
        estimated = min(llm_count, elbow_count, len(scores))
        return max(1, estimated)  # At least 1 citation

    def _predict_citation_count(self, query: str) -> int:
        """Use LLM to predict appropriate citation count."""
        if self.llm is None:
            return self._heuristic_citation_count(query)

        prompt = COUNT_PROMPT.format(query=query)
        try:
            response = self.llm(
                prompt,
                max_tokens=10,
                temperature=0.1,
                stop=["\n", "</s>"],
            )
            output = response["choices"][0]["text"].strip()
            # Extract number from output
            match = re.search(r"\d+", output)
            if match:
                count = int(match.group())
                return max(1, min(count, 10))  # Clamp between 1 and 10
        except Exception as e:
            logger.warning(f"Citation count prediction failed: {e}")

        return self._heuristic_citation_count(query)

    def _heuristic_citation_count(self, query: str) -> int:
        """Heuristic fallback for citation count estimation."""
        query_lower = query.lower()
        if any(word in query_lower for word in ["compare", "versus", "against", "difference"]):
            return 4
        elif any(word in query_lower for word in ["analyze", "analysis", "explain", "discuss"]):
            return 3
        elif any(word in query_lower for word in ["define", "what is", "meaning"]):
            return 1
        else:
            return 2

    def _detect_score_elbow(self, scores: list[float]) -> int:
        """Detect elbow point in sorted scores using second derivative.

        Args:
            scores: List of scores (should be sorted descending)

        Returns:
            Index of elbow point (number of citations to include)
        """
        if len(scores) < 3:
            return len(scores)

        # Calculate second differences
        diffs = []
        for i in range(1, len(scores) - 1):
            second_diff = scores[i - 1] - 2 * scores[i] + scores[i + 1]
            diffs.append((i + 1, abs(second_diff)))

        if not diffs:
            return len(scores)

        # Find point with maximum second derivative (elbow)
        elbow_idx = max(diffs, key=lambda x: x[1])[0]
        return elbow_idx

    def heuristic_verify(
        self,
        query: str,
        candidates: list[dict],
    ) -> list[dict]:
        """Fallback heuristic verification when model unavailable.

        Args:
            query: Query text
            candidates: List of candidates

        Returns:
            Candidates with heuristic scores

        Task 6.6: Implement fallback verification
        """
        query_words = set(re.findall(r"\w+", query.lower()))

        scored_candidates = []
        for candidate in candidates:
            scored_candidate = candidate.copy()
            citation = candidate.get("citation", "")
            citation_text = candidate.get("text", "")

            # Combine citation and any available text
            full_text = f"{citation} {citation_text}".lower()
            text_words = set(re.findall(r"\w+", full_text))

            # Calculate keyword overlap score
            if query_words and text_words:
                overlap = len(query_words & text_words) / len(query_words)
                score = min(overlap * 1.5, 1.0)  # Scale and cap at 1.0
            else:
                score = 0.5  # Default neutral score

            scored_candidate["verifier_score"] = score
            scored_candidate["verifier_reason"] = "Heuristic keyword matching"
            scored_candidates.append(scored_candidate)

        return scored_candidates
