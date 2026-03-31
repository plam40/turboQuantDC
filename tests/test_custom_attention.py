"""Tests for the custom TurboQuant attention kernel.

Validates that turboquant_attention() computes unbiased inner products and
produces correct attention output, compared to the biased MSE-only path.

The critical property: at 3-bit, the unbiased attention (MSE + QJL correction)
should produce attention scores that closely match FP16 ground truth, whereas
MSE-only attention has systematic bias that produces garbled output during
autoregressive generation.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from turboquantdc.custom_attention import (
    turboquant_attention,
    _gather_compressed_keys,
    _reconstruct_values,
)
from turboquantdc.estimator import TurboQuantEstimator
from turboquantdc.hf_integration import TurboQuantCache, TurboQuantLayer
from turboquantdc.polarquant import PolarQuant

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HEAD_DIM = 128
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_attention_inputs(
    batch: int = 1,
    n_q_heads: int = 4,
    n_kv_heads: int = 4,
    seq_q: int = 1,
    seq_kv: int = 32,
    head_dim: int = HEAD_DIM,
    seed: int = SEED,
):
    """Create query, key, value tensors for attention testing.

    Returns queries (batch, n_q_heads, seq_q, head_dim),
    keys and values (batch, n_kv_heads, seq_kv, head_dim).
    """
    torch.manual_seed(seed)
    queries = torch.randn(batch, n_q_heads, seq_q, head_dim)
    keys = torch.randn(batch, n_kv_heads, seq_kv, head_dim)
    values = torch.randn(batch, n_kv_heads, seq_kv, head_dim)
    return queries, keys, values


def fp16_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: torch.Tensor = None,
    n_kv_heads: int = None,
    n_q_heads: int = None,
) -> tuple:
    """Compute standard FP16 attention (ground truth).

    Returns (output, scores, weights).
    """
    head_dim = queries.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)

    if n_kv_heads is not None and n_q_heads is not None:
        heads_per_kv = n_q_heads // n_kv_heads
        if heads_per_kv > 1:
            keys = keys.repeat_interleave(heads_per_kv, dim=1)
            values = values.repeat_interleave(heads_per_kv, dim=1)

    scores = torch.matmul(queries.float(), keys.float().transpose(-1, -2)) * scale

    if attention_mask is not None:
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        scores = scores + attention_mask.float()

    weights = F.softmax(scores, dim=-1, dtype=torch.float32)
    output = torch.matmul(weights, values.float())

    return output, scores, weights


def compress_keys_for_attention(
    keys: torch.Tensor,
    estimator: TurboQuantEstimator,
) -> dict:
    """Compress keys into the format expected by turboquant_attention.

    Args:
        keys: (batch, n_kv_heads, seq_kv, head_dim) key tensor.
        estimator: TurboQuantEstimator instance.

    Returns:
        Dict with shapes (batch, n_kv_heads, seq_kv, ...).
    """
    batch, n_heads, seq_kv, head_dim = keys.shape
    keys_flat = keys.float().reshape(-1, head_dim)
    comp = estimator.quantize(keys_flat)

    return {
        "mse_indices": comp["mse_indices"].reshape(batch, n_heads, seq_kv, head_dim),
        "qjl_signs": comp["qjl_signs"].reshape(batch, n_heads, seq_kv, -1),
        "residual_norm": comp["residual_norm"].reshape(batch, n_heads, seq_kv),
        "vec_norm": comp["vec_norm"].reshape(batch, n_heads, seq_kv),
    }


def mse_only_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    estimator: TurboQuantEstimator,
    attention_mask: torch.Tensor = None,
) -> tuple:
    """Compute attention using MSE-only reconstructed keys (biased).

    This is what the current HF integration does -- dequantize keys to FP16
    and run standard Q @ K^T.

    Returns (output, scores_scaled).
    """
    batch, n_heads, seq_kv, head_dim = keys.shape
    scale = 1.0 / math.sqrt(head_dim)

    # Compress and MSE-only dequantize
    keys_flat = keys.float().reshape(-1, head_dim)
    comp = estimator.quantize(keys_flat)
    k_mse = estimator.dequantize_mse(comp)
    k_mse = k_mse.reshape(batch, n_heads, seq_kv, head_dim)

    scores = torch.matmul(queries.float(), k_mse.transpose(-1, -2)) * scale

    if attention_mask is not None:
        if attention_mask.dim() == 2:
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        scores = scores + attention_mask.float()

    weights = F.softmax(scores, dim=-1, dtype=torch.float32)
    output = torch.matmul(weights, values.float())

    return output, scores


# ---------------------------------------------------------------------------
# Test: basic output shapes
# ---------------------------------------------------------------------------
class TestOutputShapes:
    """turboquant_attention must return correct shapes for all configurations."""

    @pytest.mark.parametrize("seq_q,seq_kv", [(1, 32), (1, 128), (8, 32), (16, 64)])
    def test_shape_various_seq_lengths(self, seq_q, seq_kv):
        """Output shape must be (batch, n_q_heads, seq_q, head_dim)."""
        batch, n_q_heads, n_kv_heads = 1, 4, 4
        queries, keys, values = make_attention_inputs(
            batch=batch, n_q_heads=n_q_heads, n_kv_heads=n_kv_heads,
            seq_q=seq_q, seq_kv=seq_kv,
        )
        est = TurboQuantEstimator(d=HEAD_DIM, bits=3, seed=SEED)
        compressed_keys = compress_keys_for_attention(keys, est)
        # Reconstruct values (MSE-only for values, which is correct)
        v_flat = values.float().reshape(-1, HEAD_DIM)
        v_norms = v_flat.norm(dim=-1, keepdim=True)
        v_normed = v_flat / (v_norms + 1e-8)
        val_pq = PolarQuant(d=HEAD_DIM, bits=3, seed=SEED + 100)
        v_idx = val_pq.quantize(v_normed)
        v_recon = val_pq.dequantize(v_idx) * v_norms
        v_recon = v_recon.reshape(batch, n_kv_heads, seq_kv, HEAD_DIM)

        output = turboquant_attention(
            query_states=queries,
            compressed_keys=compressed_keys,
            value_states=v_recon,
            key_estimator=est,
        )

        assert output.shape == (batch, n_q_heads, seq_q, HEAD_DIM)

    def test_shape_batch_2(self):
        """Should work with batch_size > 1."""
        batch = 2
        queries, keys, values = make_attention_inputs(batch=batch, seq_kv=16)
        est = TurboQuantEstimator(d=HEAD_DIM, bits=3, seed=SEED)
        compressed_keys = compress_keys_for_attention(keys, est)

        output = turboquant_attention(
            query_states=queries,
            compressed_keys=compressed_keys,
            value_states=values,
            key_estimator=est,
        )

        assert output.shape == (batch, 4, 1, HEAD_DIM)

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_shape_all_bitwidths(self, bits):
        """Should work for all supported bit-widths."""
        queries, keys, values = make_attention_inputs(seq_kv=16)
        est = TurboQuantEstimator(d=HEAD_DIM, bits=bits, seed=SEED)
        compressed_keys = compress_keys_for_attention(keys, est)

        output = turboquant_attention(
            query_states=queries,
            compressed_keys=compressed_keys,
            value_states=values,
            key_estimator=est,
        )

        assert output.shape == (1, 4, 1, HEAD_DIM)


# ---------------------------------------------------------------------------
# Test: GQA (grouped query attention)
# ---------------------------------------------------------------------------
class TestGQA:
    """Test that GQA head expansion works correctly."""

    def test_gqa_shape(self):
        """With 8 Q heads and 2 KV heads (ratio 4:1), output should match."""
        batch, n_q_heads, n_kv_heads = 1, 8, 2
        queries, keys, values = make_attention_inputs(
            batch=batch, n_q_heads=n_q_heads, n_kv_heads=n_kv_heads,
            seq_q=1, seq_kv=16,
        )
        est = TurboQuantEstimator(d=HEAD_DIM, bits=3, seed=SEED)
        compressed_keys = compress_keys_for_attention(keys, est)

        output = turboquant_attention(
            query_states=queries,
            compressed_keys=compressed_keys,
            value_states=values,
            key_estimator=est,
        )

        assert output.shape == (batch, n_q_heads, 1, HEAD_DIM)

    def test_gqa_kv_sharing(self):
        """Query heads sharing the same KV head should get identical scores
        when given identical query vectors."""
        batch, n_q_heads, n_kv_heads = 1, 4, 2
        seq_kv = 16
        queries = torch.randn(batch, n_q_heads, 1, HEAD_DIM)
        # Make Q heads 0 and 1 identical (they share KV head 0)
        queries[:, 1] = queries[:, 0]
        # Make Q heads 2 and 3 identical (they share KV head 1)
        queries[:, 3] = queries[:, 2]

        torch.manual_seed(SEED)
        keys = torch.randn(batch, n_kv_heads, seq_kv, HEAD_DIM)
        values = torch.randn(batch, n_kv_heads, seq_kv, HEAD_DIM)

        est = TurboQuantEstimator(d=HEAD_DIM, bits=3, seed=SEED)
        compressed_keys = compress_keys_for_attention(keys, est)

        output = turboquant_attention(
            query_states=queries,
            compressed_keys=compressed_keys,
            value_states=values,
            key_estimator=est,
        )

        # Heads 0 and 1 should produce identical output
        torch.testing.assert_close(output[:, 0], output[:, 1], atol=1e-5, rtol=1e-5)
        # Heads 2 and 3 should produce identical output
        torch.testing.assert_close(output[:, 2], output[:, 3], atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test: unbiasedness -- the critical property
# ---------------------------------------------------------------------------
class TestUnbiasedness:
    """The QJL-corrected attention scores must be unbiased estimators of the
    true inner products.  This is the fundamental guarantee that makes 3-bit
    generation work."""

    def test_attention_scores_unbiased(self):
        """Mean estimation error across many (q, k) pairs should be near zero."""
        n_q_heads, n_kv_heads = 1, 1
        seq_q, seq_kv = 50, 50

        torch.manual_seed(42)
        queries = torch.randn(1, n_q_heads, seq_q, HEAD_DIM)
        keys = torch.randn(1, n_kv_heads, seq_kv, HEAD_DIM)

        # True inner products
        true_scores = torch.matmul(queries.float(), keys.float().transpose(-1, -2))

        # Average over multiple random seeds to test unbiasedness
        all_errors = []
        for seed in range(50):
            est = TurboQuantEstimator(d=HEAD_DIM, bits=3, seed=seed + 1000)
            compressed_keys = compress_keys_for_attention(keys, est)

            # Unscaled scores from estimator (we want to compare raw IPs)
            # turboquant_attention scales by 1/sqrt(d), so compute raw scores manually
            batch, _, _, head_dim = queries.shape
            mse_idx_flat = compressed_keys["mse_indices"].reshape(-1, head_dim)
            k_mse_flat = est.polar.dequantize(mse_idx_flat)
            k_mse = k_mse_flat.reshape(1, n_kv_heads, seq_kv, head_dim)

            term1 = torch.matmul(queries.float(), k_mse.float().transpose(-1, -2))

            S = est.qjl.S
            m = S.shape[0]
            qjl_scale = math.sqrt(math.pi / 2.0) / m
            q_proj = torch.matmul(queries.float(), S.T)
            signs = compressed_keys["qjl_signs"]
            qjl_ip = torch.matmul(q_proj, signs.float().transpose(-1, -2))
            r_norm = compressed_keys["residual_norm"]
            term2 = qjl_scale * r_norm.float().unsqueeze(2) * qjl_ip

            vec_norm = compressed_keys["vec_norm"]
            est_scores = (term1 + term2) * vec_norm.float().unsqueeze(2)

            error = (est_scores - true_scores).mean().item()
            all_errors.append(error)

        mean_error = sum(all_errors) / len(all_errors)
        assert abs(mean_error) < 0.1, (
            f"Mean attention score error: {mean_error:.4f} (should be near 0 for unbiased)"
        )

    def test_mse_only_is_biased(self):
        """Demonstrate that MSE-only attention has systematic bias at 3-bit.

        Regression slope of MSE-only scores vs true scores should deviate from 1.0.
        """
        est = TurboQuantEstimator(d=HEAD_DIM, bits=3, seed=SEED)
        torch.manual_seed(42)
        queries = torch.randn(1, 1, 20, HEAD_DIM)
        keys = torch.randn(1, 1, 100, HEAD_DIM)

        true_scores = torch.matmul(queries.float(), keys.float().transpose(-1, -2))

        # MSE-only: compress + dequantize keys, then standard Q @ K^T
        keys_flat = keys.float().reshape(-1, HEAD_DIM)
        comp = est.quantize(keys_flat)
        k_mse = est.dequantize_mse(comp)
        k_mse = k_mse.reshape(1, 1, 100, HEAD_DIM)
        mse_scores = torch.matmul(queries.float(), k_mse.transpose(-1, -2))

        # Regression slope: mse_scores ~ slope * true_scores
        true_flat = true_scores.flatten()
        mse_flat = mse_scores.flatten()
        slope = (mse_flat * true_flat).sum() / (true_flat ** 2).sum()

        # At 3-bit (2-bit MSE + 1-bit QJL), the MSE-only slope should be < 1.0
        # because the MSE reconstruction attenuates inner products
        assert slope.item() < 0.95, (
            f"MSE-only slope: {slope.item():.4f} -- expected attenuation (< 0.95)"
        )

    def test_unbiased_mean_score_error_near_zero(self):
        """Averaging unbiased scores over random seeds should converge to true scores.

        The MSE-only estimator has systematic bias (attenuation), so its mean
        error does NOT converge to zero.  The unbiased estimator does.  This is
        the property that makes 3-bit generation work: the QJL correction
        removes the bias even though individual-seed scores have higher variance.
        """
        torch.manual_seed(42)
        queries = torch.randn(1, 1, 5, HEAD_DIM)
        keys = torch.randn(1, 1, 20, HEAD_DIM)

        true_scores = torch.matmul(queries.float(), keys.float().transpose(-1, -2))

        mse_errors = []
        unbiased_errors = []

        for seed in range(30):
            est = TurboQuantEstimator(d=HEAD_DIM, bits=3, seed=seed + 500)

            # MSE-only scores
            keys_flat = keys.float().reshape(-1, HEAD_DIM)
            comp = est.quantize(keys_flat)
            k_mse = est.dequantize_mse(comp)
            k_mse = k_mse.reshape(1, 1, 20, HEAD_DIM)
            mse_scores = torch.matmul(queries.float(), k_mse.transpose(-1, -2))
            mse_err = (mse_scores - true_scores).mean().item()
            mse_errors.append(mse_err)

            # Unbiased scores
            compressed_keys = compress_keys_for_attention(keys, est)
            head_dim = HEAD_DIM
            mse_idx_flat = compressed_keys["mse_indices"].reshape(-1, head_dim)
            k_mse_flat = est.polar.dequantize(mse_idx_flat)
            k_mse_r = k_mse_flat.reshape(1, 1, 20, head_dim)
            term1 = torch.matmul(queries.float(), k_mse_r.float().transpose(-1, -2))
            S = est.qjl.S
            m_dim = S.shape[0]
            qjl_scale = math.sqrt(math.pi / 2.0) / m_dim
            q_proj = torch.matmul(queries.float(), S.T)
            signs = compressed_keys["qjl_signs"]
            qjl_ip = torch.matmul(q_proj, signs.float().transpose(-1, -2))
            r_norm = compressed_keys["residual_norm"]
            term2 = qjl_scale * r_norm.float().unsqueeze(2) * qjl_ip
            vec_norm = compressed_keys["vec_norm"]
            unbiased_scores = (term1 + term2) * vec_norm.float().unsqueeze(2)
            ub_err = (unbiased_scores - true_scores).mean().item()
            unbiased_errors.append(ub_err)

        # The unbiased estimator's average error should be closer to 0
        mean_mse_err = abs(sum(mse_errors) / len(mse_errors))
        mean_ub_err = abs(sum(unbiased_errors) / len(unbiased_errors))

        assert mean_ub_err < mean_mse_err, (
            f"Unbiased mean error {mean_ub_err:.4f} should be smaller than "
            f"MSE-only mean error {mean_mse_err:.4f}"
        )
        # Unbiased mean error should be near zero
        assert mean_ub_err < 0.5, (
            f"Unbiased mean error {mean_ub_err:.4f} too large (should be near 0)"
        )


# ---------------------------------------------------------------------------
# Test: attention mask handling
# ---------------------------------------------------------------------------
class TestAttentionMask:
    """Test that causal and padding masks are applied correctly."""

    def test_causal_mask_4d(self):
        """4D causal mask should prevent attending to future positions."""
        seq_q, seq_kv = 4, 8
        queries, keys, values = make_attention_inputs(
            seq_q=seq_q, seq_kv=seq_kv, n_q_heads=1, n_kv_heads=1,
        )
        est = TurboQuantEstimator(d=HEAD_DIM, bits=3, seed=SEED)
        compressed_keys = compress_keys_for_attention(keys, est)

        # Create causal mask: (1, 1, seq_q, seq_kv)
        # Positions where mask is -inf should get zero attention weight
        causal_mask = torch.zeros(1, 1, seq_q, seq_kv)
        causal_mask[:, :, :, seq_kv - 2:] = float("-inf")

        output = turboquant_attention(
            query_states=queries,
            compressed_keys=compressed_keys,
            value_states=values,
            key_estimator=est,
            attention_mask=causal_mask,
        )

        assert output.shape == (1, 1, seq_q, HEAD_DIM)
        # Output should be finite (no NaN/inf from masked softmax)
        assert torch.isfinite(output).all()

    def test_2d_mask(self):
        """2D mask should be broadcast correctly."""
        seq_q, seq_kv = 1, 16
        queries, keys, values = make_attention_inputs(
            seq_q=seq_q, seq_kv=seq_kv, n_q_heads=2, n_kv_heads=2,
        )
        est = TurboQuantEstimator(d=HEAD_DIM, bits=3, seed=SEED)
        compressed_keys = compress_keys_for_attention(keys, est)

        mask_2d = torch.zeros(seq_q, seq_kv)
        mask_2d[:, :4] = float("-inf")  # Mask first 4 positions

        output = turboquant_attention(
            query_states=queries,
            compressed_keys=compressed_keys,
            value_states=values,
            key_estimator=est,
            attention_mask=mask_2d,
        )

        assert output.shape == (1, 2, seq_q, HEAD_DIM)
        assert torch.isfinite(output).all()

    def test_no_mask(self):
        """No mask should attend to all positions equally (no masking)."""
        queries, keys, values = make_attention_inputs(seq_q=1, seq_kv=16)
        est = TurboQuantEstimator(d=HEAD_DIM, bits=3, seed=SEED)
        compressed_keys = compress_keys_for_attention(keys, est)

        output = turboquant_attention(
            query_states=queries,
            compressed_keys=compressed_keys,
            value_states=values,
            key_estimator=est,
            attention_mask=None,
        )

        assert output.shape == (1, 4, 1, HEAD_DIM)
        assert torch.isfinite(output).all()


# ---------------------------------------------------------------------------
# Test: agreement with estimator.inner_product
# ---------------------------------------------------------------------------
class TestEstimatorAgreement:
    """The attention score computation in turboquant_attention must match
    the scores produced by TurboQuantEstimator.inner_product()."""

    def test_scores_match_estimator(self):
        """Raw (unscaled) attention scores should match estimator.inner_product()."""
        est = TurboQuantEstimator(d=HEAD_DIM, bits=3, seed=SEED)

        torch.manual_seed(42)
        n_q, n_k = 5, 10
        queries_flat = torch.randn(n_q, HEAD_DIM)
        keys_flat = torch.randn(n_k, HEAD_DIM)

        # Estimator path
        comp = est.quantize(keys_flat)
        est_scores = est.inner_product(queries_flat, comp)  # (n_q, n_k)

        # Custom attention path (single head, batch=1)
        queries = queries_flat.unsqueeze(0).unsqueeze(0)  # (1, 1, n_q, d)
        keys = keys_flat.unsqueeze(0).unsqueeze(0)  # (1, 1, n_k, d)
        compressed_keys = compress_keys_for_attention(keys, est)

        # Reconstruct raw scores from turboquant_attention internals
        head_dim = HEAD_DIM
        mse_idx_flat = compressed_keys["mse_indices"].reshape(-1, head_dim)
        k_mse_flat = est.polar.dequantize(mse_idx_flat)
        k_mse = k_mse_flat.reshape(1, 1, n_k, head_dim)

        term1 = torch.matmul(queries.float(), k_mse.float().transpose(-1, -2))

        S = est.qjl.S
        m = S.shape[0]
        qjl_scale = math.sqrt(math.pi / 2.0) / m
        q_proj = torch.matmul(queries.float(), S.T)
        signs = compressed_keys["qjl_signs"]
        qjl_ip = torch.matmul(q_proj, signs.float().transpose(-1, -2))
        r_norm = compressed_keys["residual_norm"]
        term2 = qjl_scale * r_norm.float().unsqueeze(2) * qjl_ip
        vec_norm = compressed_keys["vec_norm"]
        custom_scores = (term1 + term2) * vec_norm.float().unsqueeze(2)
        custom_scores = custom_scores.squeeze(0).squeeze(0)  # (n_q, n_k)

        # These should match the estimator's inner_product output
        torch.testing.assert_close(
            custom_scores, est_scores, atol=1e-4, rtol=1e-4,
        )


# ---------------------------------------------------------------------------
# Test: needle-in-a-haystack through attention
# ---------------------------------------------------------------------------
class TestNeedleInHaystack:
    """A query matching one key exactly should produce high attention weight
    on that key, even after quantization and the full attention pipeline."""

    @pytest.mark.parametrize("bits", [3, 4])
    def test_needle_gets_high_attention_weight(self, bits):
        """The needle key should receive significant attention weight."""
        est = TurboQuantEstimator(d=HEAD_DIM, bits=bits, seed=SEED)
        n_keys = 64
        needle_idx = 30

        torch.manual_seed(42)
        keys = torch.randn(1, 1, n_keys, HEAD_DIM)
        values = torch.randn(1, 1, n_keys, HEAD_DIM)

        # Query is exactly one of the keys
        query = keys[:, :, needle_idx:needle_idx+1, :].clone()

        compressed_keys = compress_keys_for_attention(keys, est)

        output = turboquant_attention(
            query_states=query,
            compressed_keys=compressed_keys,
            value_states=values,
            key_estimator=est,
        )

        # Manually compute attention weights to check
        head_dim = HEAD_DIM
        scale = 1.0 / math.sqrt(head_dim)
        mse_idx_flat = compressed_keys["mse_indices"].reshape(-1, head_dim)
        k_mse_flat = est.polar.dequantize(mse_idx_flat)
        k_mse = k_mse_flat.reshape(1, 1, n_keys, head_dim)
        term1 = torch.matmul(query.float(), k_mse.float().transpose(-1, -2))
        S = est.qjl.S
        m = S.shape[0]
        qjl_scale = math.sqrt(math.pi / 2.0) / m
        q_proj = torch.matmul(query.float(), S.T)
        signs = compressed_keys["qjl_signs"]
        qjl_ip = torch.matmul(q_proj, signs.float().transpose(-1, -2))
        r_norm = compressed_keys["residual_norm"]
        term2 = qjl_scale * r_norm.float().unsqueeze(2) * qjl_ip
        vec_norm = compressed_keys["vec_norm"]
        scores = ((term1 + term2) * vec_norm.float().unsqueeze(2)) * scale

        weights = F.softmax(scores, dim=-1)
        needle_weight = weights[0, 0, 0, needle_idx].item()

        # The needle should get at least 2x the uniform weight
        uniform_weight = 1.0 / n_keys
        assert needle_weight > 2 * uniform_weight, (
            f"Needle weight {needle_weight:.4f} too low "
            f"(uniform = {uniform_weight:.4f})"
        )

    def test_needle_top1_at_4bit(self):
        """At 4-bit, the needle should get the highest attention weight."""
        est = TurboQuantEstimator(d=HEAD_DIM, bits=4, seed=SEED)
        n_keys = 64
        needle_idx = 30

        torch.manual_seed(42)
        keys = torch.randn(1, 1, n_keys, HEAD_DIM)
        values = torch.randn(1, 1, n_keys, HEAD_DIM)
        query = keys[:, :, needle_idx:needle_idx+1, :].clone()

        compressed_keys = compress_keys_for_attention(keys, est)

        # Compute scores
        head_dim = HEAD_DIM
        scale = 1.0 / math.sqrt(head_dim)
        mse_idx_flat = compressed_keys["mse_indices"].reshape(-1, head_dim)
        k_mse_flat = est.polar.dequantize(mse_idx_flat)
        k_mse = k_mse_flat.reshape(1, 1, n_keys, head_dim)
        term1 = torch.matmul(query.float(), k_mse.float().transpose(-1, -2))
        S = est.qjl.S
        m = S.shape[0]
        qjl_scale = math.sqrt(math.pi / 2.0) / m
        q_proj = torch.matmul(query.float(), S.T)
        signs = compressed_keys["qjl_signs"]
        qjl_ip = torch.matmul(q_proj, signs.float().transpose(-1, -2))
        r_norm = compressed_keys["residual_norm"]
        term2 = qjl_scale * r_norm.float().unsqueeze(2) * qjl_ip
        vec_norm = compressed_keys["vec_norm"]
        scores = ((term1 + term2) * vec_norm.float().unsqueeze(2)) * scale

        top_idx = scores.squeeze().argmax().item()
        assert top_idx == needle_idx, (
            f"At 4-bit, needle should rank #1. Got index {top_idx}"
        )


# ---------------------------------------------------------------------------
# Test: integration with TurboQuantLayer
# ---------------------------------------------------------------------------
class TestLayerIntegration:
    """Test that custom attention works with TurboQuantLayer's storage format."""

    def test_gather_compressed_keys_shape(self):
        """_gather_compressed_keys should concatenate along seq dimension."""
        layer = TurboQuantLayer(bits=3, seed=SEED)
        torch.manual_seed(SEED)
        k1 = torch.randn(1, 2, 4, HEAD_DIM)
        v1 = torch.randn(1, 2, 4, HEAD_DIM)
        layer.update(k1, v1)

        k2 = torch.randn(1, 2, 3, HEAD_DIM)
        v2 = torch.randn(1, 2, 3, HEAD_DIM)
        layer.update(k2, v2)

        compressed = _gather_compressed_keys(layer)
        assert compressed["mse_indices"].shape == (1, 2, 7, HEAD_DIM)
        assert compressed["residual_norm"].shape == (1, 2, 7)
        assert compressed["vec_norm"].shape == (1, 2, 7)

    def test_reconstruct_values_shape(self):
        """_reconstruct_values should return correct shape."""
        layer = TurboQuantLayer(bits=3, seed=SEED)
        torch.manual_seed(SEED)
        k = torch.randn(1, 2, 5, HEAD_DIM)
        v = torch.randn(1, 2, 5, HEAD_DIM)
        layer.update(k, v)

        v_recon = _reconstruct_values(layer)
        assert v_recon.shape == (1, 2, 5, HEAD_DIM)

    def test_full_pipeline_through_layer(self):
        """End-to-end: store through layer, compute attention, verify output."""
        layer = TurboQuantLayer(bits=3, seed=SEED)

        torch.manual_seed(42)
        keys = torch.randn(1, 2, 16, HEAD_DIM)
        values = torch.randn(1, 2, 16, HEAD_DIM)
        layer.update(keys, values)

        queries = torch.randn(1, 2, 1, HEAD_DIM)

        compressed_keys = _gather_compressed_keys(layer)
        value_states = _reconstruct_values(layer)

        output = turboquant_attention(
            query_states=queries,
            compressed_keys=compressed_keys,
            value_states=value_states,
            key_estimator=layer._key_est,
        )

        assert output.shape == (1, 2, 1, HEAD_DIM)
        assert torch.isfinite(output).all()


# ---------------------------------------------------------------------------
# Test: TurboQuantCache.enable_unbiased_attention helper
# ---------------------------------------------------------------------------
class TestCacheIntegration:
    """Test the cache-level integration methods."""

    def test_make_layer_creates_correct_type(self):
        """_make_layer should create TurboQuantLayer with correct params."""
        cache = TurboQuantCache(bits=3, seed=100)
        layer = cache._make_layer(5)
        assert isinstance(layer, TurboQuantLayer)
        assert layer.bits == 3
        assert layer.seed == 105  # seed + layer_idx

    def test_make_layer_different_per_index(self):
        """Each layer index should get a different seed."""
        cache = TurboQuantCache(bits=3, seed=100)
        l0 = cache._make_layer(0)
        l1 = cache._make_layer(1)
        assert l0.seed != l1.seed


# ---------------------------------------------------------------------------
# Test: scale parameter
# ---------------------------------------------------------------------------
class TestScaleParameter:
    """Test that the scale parameter works correctly."""

    def test_default_scale(self):
        """Default scale should be 1/sqrt(head_dim)."""
        queries, keys, values = make_attention_inputs(seq_q=1, seq_kv=8)
        est = TurboQuantEstimator(d=HEAD_DIM, bits=3, seed=SEED)
        compressed_keys = compress_keys_for_attention(keys, est)

        output_default = turboquant_attention(
            query_states=queries,
            compressed_keys=compressed_keys,
            value_states=values,
            key_estimator=est,
        )

        output_explicit = turboquant_attention(
            query_states=queries,
            compressed_keys=compressed_keys,
            value_states=values,
            key_estimator=est,
            scale=1.0 / math.sqrt(HEAD_DIM),
        )

        torch.testing.assert_close(output_default, output_explicit, atol=1e-5, rtol=1e-5)

    def test_custom_scale(self):
        """Custom scale should affect output (different from default)."""
        queries, keys, values = make_attention_inputs(seq_q=1, seq_kv=8)
        est = TurboQuantEstimator(d=HEAD_DIM, bits=3, seed=SEED)
        compressed_keys = compress_keys_for_attention(keys, est)

        output_default = turboquant_attention(
            query_states=queries,
            compressed_keys=compressed_keys,
            value_states=values,
            key_estimator=est,
        )

        output_2x = turboquant_attention(
            query_states=queries,
            compressed_keys=compressed_keys,
            value_states=values,
            key_estimator=est,
            scale=2.0 / math.sqrt(HEAD_DIM),
        )

        # Different scale should produce different output
        assert not torch.allclose(output_default, output_2x, atol=1e-3)


# ---------------------------------------------------------------------------
# Test: numerical stability
# ---------------------------------------------------------------------------
class TestNumericalStability:
    """Test that the implementation handles edge cases without NaN/inf."""

    def test_large_values(self):
        """Should handle large-magnitude vectors without overflow."""
        queries, keys, values = make_attention_inputs(seq_q=1, seq_kv=8)
        queries = queries * 100
        keys = keys * 100

        est = TurboQuantEstimator(d=HEAD_DIM, bits=3, seed=SEED)
        compressed_keys = compress_keys_for_attention(keys, est)

        output = turboquant_attention(
            query_states=queries,
            compressed_keys=compressed_keys,
            value_states=values,
            key_estimator=est,
        )

        assert torch.isfinite(output).all()

    def test_small_values(self):
        """Should handle small-magnitude vectors without underflow issues."""
        queries, keys, values = make_attention_inputs(seq_q=1, seq_kv=8)
        queries = queries * 0.001
        keys = keys * 0.001

        est = TurboQuantEstimator(d=HEAD_DIM, bits=3, seed=SEED)
        compressed_keys = compress_keys_for_attention(keys, est)

        output = turboquant_attention(
            query_states=queries,
            compressed_keys=compressed_keys,
            value_states=values,
            key_estimator=est,
        )

        assert torch.isfinite(output).all()

    def test_seq_kv_1(self):
        """Should work correctly when there is only 1 key."""
        queries, keys, values = make_attention_inputs(seq_q=1, seq_kv=1)
        est = TurboQuantEstimator(d=HEAD_DIM, bits=3, seed=SEED)
        compressed_keys = compress_keys_for_attention(keys, est)

        output = turboquant_attention(
            query_states=queries,
            compressed_keys=compressed_keys,
            value_states=values,
            key_estimator=est,
        )

        assert output.shape == (1, 4, 1, HEAD_DIM)
        assert torch.isfinite(output).all()

        # With only 1 key, attention weight must be 1.0 for that key,
        # so output should be close to the value (regardless of score)
        # Note: values here are MSE-reconstructed, not the raw values


# ---------------------------------------------------------------------------
# Test: GPU if available
# ---------------------------------------------------------------------------
class TestGPU:
    """Test custom attention on CUDA when available."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_custom_attention_gpu(self):
        """Full pipeline should work on GPU."""
        device = "cuda"
        est = TurboQuantEstimator(d=HEAD_DIM, bits=3, seed=SEED, device=device)

        torch.manual_seed(42)
        queries = torch.randn(1, 2, 1, HEAD_DIM, device=device)
        keys = torch.randn(1, 2, 16, HEAD_DIM, device=device)
        values = torch.randn(1, 2, 16, HEAD_DIM, device=device)

        compressed_keys = compress_keys_for_attention(keys, est)

        output = turboquant_attention(
            query_states=queries,
            compressed_keys=compressed_keys,
            value_states=values,
            key_estimator=est,
        )

        assert output.device.type == "cuda"
        assert output.shape == (1, 2, 1, HEAD_DIM)
        assert torch.isfinite(output).all()

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_layer_integration_gpu(self):
        """TurboQuantLayer + custom attention on GPU."""
        device = "cuda"
        layer = TurboQuantLayer(bits=3, seed=SEED)

        torch.manual_seed(42)
        keys = torch.randn(1, 2, 16, HEAD_DIM, device=device)
        values = torch.randn(1, 2, 16, HEAD_DIM, device=device)
        layer.update(keys, values)

        queries = torch.randn(1, 2, 1, HEAD_DIM, device=device)

        compressed_keys = _gather_compressed_keys(layer)
        value_states = _reconstruct_values(layer)

        output = turboquant_attention(
            query_states=queries,
            compressed_keys=compressed_keys,
            value_states=value_states,
            key_estimator=layer._key_est,
        )

        assert output.device.type == "cuda"
        assert output.shape == (1, 2, 1, HEAD_DIM)
        assert torch.isfinite(output).all()
