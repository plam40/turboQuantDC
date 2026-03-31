"""Tests for the benchmark module.

Tests perplexity computation, generation quality scoring, and score
normalization without requiring GPU or model loading.
"""

from __future__ import annotations

import math

import pytest
import torch


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

from benchmark import (
    GENERATION_PROMPTS,
    compute_perplexity,
    needle_in_haystack_score,
    normalize_ppl_score,
    combined_score,
    score_response_similarity,
)


# ---------------------------------------------------------------------------
# Perplexity normalization
# ---------------------------------------------------------------------------


class TestNormalizePplScore:
    """Test the perplexity-to-score mapping."""

    def test_zero_increase_is_one(self):
        """No perplexity increase = perfect score of 1.0."""
        assert normalize_ppl_score(10.0, 10.0) == pytest.approx(1.0)

    def test_eight_percent_increase(self):
        """8% perplexity increase should give 0.92."""
        baseline = 10.0
        compressed = 10.8  # 8% increase
        score = normalize_ppl_score(baseline, compressed)
        assert score == pytest.approx(0.92, abs=0.001)

    def test_fifty_percent_increase(self):
        """50% perplexity increase = 0.5."""
        score = normalize_ppl_score(10.0, 15.0)
        assert score == pytest.approx(0.5, abs=0.001)

    def test_hundred_percent_increase(self):
        """100% perplexity increase should clamp to 0.0."""
        score = normalize_ppl_score(10.0, 20.0)
        assert score == pytest.approx(0.0, abs=0.001)

    def test_decrease_clamps_to_one(self):
        """If compressed is somehow lower, clamp to 1.0."""
        score = normalize_ppl_score(10.0, 9.5)
        assert score == pytest.approx(1.0)

    def test_baseline_zero_returns_zero(self):
        """Edge case: baseline of 0 should not crash."""
        score = normalize_ppl_score(0.0, 5.0)
        assert 0.0 <= score <= 1.0


class TestComputePerplexity:
    """Test the perplexity computation from log-likelihoods."""

    def test_uniform_distribution(self):
        """Uniform distribution over V=100 tokens -> ppl = 100."""
        # log(1/100) = -log(100) for each token
        V = 100
        seq_len = 50
        neg_log_likelihoods = torch.full((seq_len,), math.log(V))
        ppl = compute_perplexity(neg_log_likelihoods)
        assert ppl == pytest.approx(V, rel=0.01)

    def test_perfect_prediction(self):
        """Perfect prediction -> ppl = 1."""
        neg_log_likelihoods = torch.zeros(50)
        ppl = compute_perplexity(neg_log_likelihoods)
        assert ppl == pytest.approx(1.0, abs=0.01)

    def test_higher_loss_gives_higher_ppl(self):
        """Higher average NLL -> higher perplexity."""
        low_loss = torch.full((50,), 1.0)
        high_loss = torch.full((50,), 3.0)
        ppl_low = compute_perplexity(low_loss)
        ppl_high = compute_perplexity(high_loss)
        assert ppl_high > ppl_low


# ---------------------------------------------------------------------------
# Generation quality scoring
# ---------------------------------------------------------------------------


class TestScoreResponseSimilarity:
    """Test response similarity scoring."""

    def test_identical_responses_score_one(self):
        """Identical baseline and compressed responses should score 1.0."""
        text = "The answer is 42."
        score = score_response_similarity(text, text)
        assert score == pytest.approx(1.0)

    def test_empty_responses(self):
        """Both empty should not crash."""
        score = score_response_similarity("", "")
        assert 0.0 <= score <= 1.0

    def test_completely_different(self):
        """Completely different responses should score low."""
        score = score_response_similarity(
            "The capital of France is Paris.",
            "xkcd random noise gibberish qwerty uiop",
        )
        assert score < 0.5

    def test_similar_responses_score_higher_than_garbage(self):
        """Responses with same key facts but different wording should beat garbage."""
        similar_score = score_response_similarity(
            "Jupiter is the largest planet in our solar system.",
            "The largest planet in the solar system is Jupiter.",
        )
        garbage_score = score_response_similarity(
            "Jupiter is the largest planet in our solar system.",
            "xkcd random noise gibberish qwerty uiop",
        )
        assert similar_score > garbage_score


class TestGenerationPrompts:
    """Verify the prompt list is well-formed."""

    def test_minimum_prompt_count(self):
        """Should have at least 12 prompts."""
        assert len(GENERATION_PROMPTS) >= 12

    def test_prompts_have_required_fields(self):
        """Each prompt config must have prompt, type, and expected fields."""
        for p in GENERATION_PROMPTS:
            assert "prompt" in p, f"Missing 'prompt' key in {p}"
            assert "type" in p, f"Missing 'type' key in {p}"
            assert "expected" in p, f"Missing 'expected' key in {p}"

    def test_no_broken_1984_prompt(self):
        """The broken '1984' prompt should not be present."""
        for p in GENERATION_PROMPTS:
            assert "1984" not in p["prompt"], "The broken 1984 prompt should be removed"

    def test_diverse_types(self):
        """Should cover multiple prompt types."""
        types = {p["type"] for p in GENERATION_PROMPTS}
        assert "factual" in types
        assert "math" in types
        assert "code" in types
        assert "reasoning" in types


# ---------------------------------------------------------------------------
# Combined score
# ---------------------------------------------------------------------------


class TestCombinedScore:
    """Test the combined scoring formula."""

    def test_perfect_scores(self):
        """Both perfect -> combined = 1.0."""
        score = combined_score(ppl_score=1.0, gen_score=1.0)
        assert score == pytest.approx(1.0)

    def test_weights_sum_to_one(self):
        """60% ppl + 40% gen should sum correctly."""
        score = combined_score(ppl_score=0.8, gen_score=0.6)
        expected = 0.6 * 0.8 + 0.4 * 0.6
        assert score == pytest.approx(expected, abs=0.001)

    def test_zero_gen_score(self):
        """Only perplexity contributing."""
        score = combined_score(ppl_score=1.0, gen_score=0.0)
        assert score == pytest.approx(0.6)

    def test_zero_ppl_score(self):
        """Only generation contributing."""
        score = combined_score(ppl_score=0.0, gen_score=1.0)
        assert score == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Needle-in-haystack score helper
# ---------------------------------------------------------------------------


class TestNeedleInHaystack:
    """Test the needle-in-haystack scoring helper."""

    def test_returns_float(self):
        """Should return a float between 0 and 1."""
        score = needle_in_haystack_score(
            response="The secret word is banana.",
            needle="banana",
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_found_needle(self):
        """Response containing the needle should score 1.0."""
        score = needle_in_haystack_score(
            response="The hidden fact is that the capital is Paris.",
            needle="Paris",
        )
        assert score == pytest.approx(1.0)

    def test_missing_needle(self):
        """Response without the needle should score 0.0."""
        score = needle_in_haystack_score(
            response="I don't know the answer to that.",
            needle="Paris",
        )
        assert score == pytest.approx(0.0)
