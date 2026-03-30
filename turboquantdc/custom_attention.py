"""Custom attention using TurboQuant's unbiased inner product estimator.

Standard HF attention computes Q @ K^T using dequantized (MSE-only) keys.
At 3-bit, this produces biased attention scores because the QJL correction
term is discarded during dequantization. The result: garbled autoregressive
generation output.

This module replaces the attention score computation with:

    score_ij = ||k_j|| * (<q_i, k_mse_j> + ||r_j|| * sqrt(pi/2)/m * <S@q_i, signs_j>)

This preserves TurboQuant's mathematical unbiasedness guarantee: E[score] = <q, k>.

Two integration paths are provided:

1. ``turboquant_attention()`` -- standalone function that computes attention
   output from queries and compressed KV data.  Can be called directly.

2. ``patch_model_attention()`` -- monkey-patches every attention layer in a
   HuggingFace model so that ``model.generate()`` uses the unbiased estimator
   transparently.  Works with Qwen2, Llama, Mistral, and other models that
   follow the standard ``(query, key, value) -> output`` attention pattern.

Reference: TurboQuant paper (arxiv 2504.19874), Algorithm 2 / Theorem 2.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def turboquant_attention(
    query_states: torch.Tensor,
    compressed_keys: Dict[str, torch.Tensor],
    value_states: torch.Tensor,
    key_estimator,
    attention_mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute attention using TurboQuant's unbiased inner product estimator.

    Instead of ``scores = Q @ K^T`` (biased at low bit-widths), this computes:

        score_ij = ||k_j|| * (<q_i, k_mse_j> + ||r_j|| * sqrt(pi/2)/m * <S@q_i, signs_j>)

    The value path is unchanged: values are MSE-reconstructed FP16 tensors and
    the standard ``weights @ V`` weighted sum is applied.

    Args:
        query_states: (batch, n_heads, seq_q, head_dim) query tensor.
        compressed_keys: Dict containing compressed key data with shapes
            (batch, n_kv_heads, seq_kv, ...).  Required keys:

            - ``mse_indices``: (batch, n_kv_heads, seq_kv, head_dim) codebook indices
            - ``qjl_signs``:   (batch, n_kv_heads, seq_kv, m) sign bits in {-1, +1}
            - ``residual_norm``: (batch, n_kv_heads, seq_kv) residual norms
            - ``vec_norm``:      (batch, n_kv_heads, seq_kv) original key norms

        value_states: (batch, n_kv_heads, seq_kv, head_dim) MSE-reconstructed values.
        key_estimator: ``TurboQuantEstimator`` for the key quantizer (provides
            ``polar.dequantize()``, ``qjl.inner_product_correction()``, and the
            QJL projection matrix ``qjl.S``).
        attention_mask: Optional mask broadcastable to (batch, n_heads, seq_q, seq_kv).
            Additive mask -- positions to ignore should be ``-inf`` (or a large
            negative number).
        scale: Attention score scaling factor.  Defaults to ``1 / sqrt(head_dim)``.

    Returns:
        (batch, n_heads, seq_q, head_dim) attention output tensor.
    """
    batch, n_q_heads, seq_q, head_dim = query_states.shape
    n_kv_heads = compressed_keys["mse_indices"].shape[1]
    seq_kv = compressed_keys["mse_indices"].shape[2]
    heads_per_kv = n_q_heads // n_kv_heads

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    dtype = query_states.dtype
    device = query_states.device

    # --- Reconstruct MSE keys: dequantize per (batch, kv_head, seq_kv) ------
    # Flatten to (batch * n_kv_heads * seq_kv, head_dim) for the codebook lookup.
    mse_idx_flat = compressed_keys["mse_indices"].reshape(-1, head_dim)
    k_mse_flat = key_estimator.polar.dequantize(mse_idx_flat)  # float32
    k_mse = k_mse_flat.reshape(batch, n_kv_heads, seq_kv, head_dim)

    # --- Compute Term 1: <q, k_mse> per KV head --------------------------------
    # Expand KV heads via repeat_interleave if GQA (heads_per_kv > 1).
    if heads_per_kv > 1:
        k_mse_expanded = k_mse.repeat_interleave(heads_per_kv, dim=1)
    else:
        k_mse_expanded = k_mse

    # (batch, n_q_heads, seq_q, head_dim) @ (batch, n_q_heads, head_dim, seq_kv)
    # -> (batch, n_q_heads, seq_q, seq_kv)
    term1 = torch.matmul(
        query_states.float(), k_mse_expanded.float().transpose(-1, -2)
    )

    # --- Compute Term 2: QJL correction -----------------------------------------
    # Project all queries through S: q_proj = q @ S^T  shape (batch, n_q_heads, seq_q, m)
    S = key_estimator.qjl.S  # (m, d)
    m = S.shape[0]
    qjl_scale = math.sqrt(math.pi / 2.0) / m

    q_proj = torch.matmul(query_states.float(), S.T)  # (..., m)

    # signs shape: (batch, n_kv_heads, seq_kv, m) -> expand for GQA
    signs = compressed_keys["qjl_signs"]
    if heads_per_kv > 1:
        signs_expanded = signs.repeat_interleave(heads_per_kv, dim=1)
    else:
        signs_expanded = signs

    # <S@q, signs> = q_proj @ signs^T
    # (batch, n_q_heads, seq_q, m) @ (batch, n_q_heads, m, seq_kv)
    # -> (batch, n_q_heads, seq_q, seq_kv)
    qjl_ip = torch.matmul(q_proj, signs_expanded.float().transpose(-1, -2))

    # residual_norm: (batch, n_kv_heads, seq_kv) -> expand for GQA
    residual_norm = compressed_keys["residual_norm"]
    if heads_per_kv > 1:
        residual_norm_expanded = residual_norm.repeat_interleave(heads_per_kv, dim=1)
    else:
        residual_norm_expanded = residual_norm

    # term2 = ||r|| * sqrt(pi/2)/m * <S@q, signs>
    # residual_norm_expanded: (batch, n_q_heads, seq_kv) -> unsqueeze to (batch, n_q_heads, 1, seq_kv)
    term2 = qjl_scale * residual_norm_expanded.float().unsqueeze(2) * qjl_ip

    # --- Combine: score = vec_norm * (term1 + term2) ----------------------------
    vec_norm = compressed_keys["vec_norm"]
    if heads_per_kv > 1:
        vec_norm_expanded = vec_norm.repeat_interleave(heads_per_kv, dim=1)
    else:
        vec_norm_expanded = vec_norm

    # vec_norm_expanded: (batch, n_q_heads, seq_kv) -> (batch, n_q_heads, 1, seq_kv)
    scores = (term1 + term2) * vec_norm_expanded.float().unsqueeze(2)

    # --- Scale and mask ---------------------------------------------------------
    scores = scores * scale

    if attention_mask is not None:
        # HF masks come in various shapes.  Try to broadcast.
        if attention_mask.dim() == 2:
            # (seq_q, seq_kv) -> (1, 1, seq_q, seq_kv)
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        elif attention_mask.dim() == 3:
            # (batch, seq_q, seq_kv) -> (batch, 1, seq_q, seq_kv)
            attention_mask = attention_mask.unsqueeze(1)
        # Slice to match actual seq dims (handles prefill + decode size mismatches)
        mask_seq_q = attention_mask.shape[-2]
        mask_seq_kv = attention_mask.shape[-1]
        if mask_seq_q > seq_q:
            attention_mask = attention_mask[..., -seq_q:, :]
        if mask_seq_kv > seq_kv:
            attention_mask = attention_mask[..., :, -seq_kv:]
        elif mask_seq_kv < seq_kv:
            # Pad mask to match seq_kv (shouldn't happen, but be safe)
            pad = torch.zeros(
                *attention_mask.shape[:-1], seq_kv - mask_seq_kv,
                dtype=attention_mask.dtype, device=device,
            )
            attention_mask = torch.cat([pad, attention_mask], dim=-1)
        scores = scores + attention_mask.float()

    # --- Softmax and weighted value sum -----------------------------------------
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)

    # Values: (batch, n_kv_heads, seq_kv, head_dim) -> expand for GQA
    if heads_per_kv > 1:
        v_expanded = value_states.repeat_interleave(heads_per_kv, dim=1)
    else:
        v_expanded = value_states

    # (batch, n_q_heads, seq_q, seq_kv) @ (batch, n_q_heads, seq_kv, head_dim)
    # -> (batch, n_q_heads, seq_q, head_dim)
    output = torch.matmul(weights.to(v_expanded.dtype), v_expanded)

    return output


def patch_model_attention(
    model: torch.nn.Module,
    cache: "TurboQuantCache",
) -> torch.nn.Module:
    """Monkey-patch a HuggingFace model to use TurboQuant unbiased attention.

    After patching, ``model.generate()`` will use the full two-stage estimator
    for key inner products instead of standard Q @ K^T on MSE-reconstructed keys.

    The patching intercepts each attention layer's forward method:

    1. Q/K/V linear projections run normally.
    2. New K/V tokens are compressed and stored in the TurboQuantCache.
    3. Attention scores are computed via ``turboquant_attention()`` using the
       estimator's unbiased inner product.
    4. The output is returned to the model as usual.

    Args:
        model: A HuggingFace CausalLM model (Qwen2, Llama, Mistral, etc.).
        cache: A ``TurboQuantCache`` instance that will be used as the KV cache.

    Returns:
        The same model object, with attention layers patched in-place.
    """
    # Detect model architecture
    inner_model = _get_inner_model(model)
    layers = _get_layers(inner_model)
    config = model.config

    n_q_heads = config.num_attention_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_q_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // n_q_heads)

    for layer_idx, layer in enumerate(layers):
        attn_module = _get_attention_module(layer)
        if attn_module is None:
            continue

        original_forward = attn_module.forward

        def _make_patched_forward(lidx, orig_fwd, attn_mod):
            def patched_forward(
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                past_key_values=None,
                output_attentions: bool = False,
                use_cache: bool = False,
                cache_position: Optional[torch.Tensor] = None,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                **kwargs,
            ):
                # Run Q/K/V projections (works for Qwen2, Llama, Mistral)
                bsz, q_len, hidden_size = hidden_states.shape

                q_proj, k_proj, v_proj = _run_qkv_projections(
                    attn_mod, hidden_states, n_q_heads, n_kv_heads, head_dim,
                )

                # Apply rotary embeddings if position_embeddings are provided
                if position_embeddings is not None:
                    cos, sin = position_embeddings
                    q_proj, k_proj = _apply_rotary_pos_emb(
                        q_proj, k_proj, cos, sin, attn_mod
                    )
                elif position_ids is not None and hasattr(attn_mod, "rotary_emb"):
                    cos, sin = attn_mod.rotary_emb(v_proj, position_ids)
                    q_proj, k_proj = _apply_rotary_pos_emb(
                        q_proj, k_proj, cos, sin, attn_mod
                    )

                # Compress new K/V into the TurboQuantCache
                # update() stores compressed data and returns dequantized FP16 tensors.
                # We need to store compressed AND get reconstructed values.
                #
                # Ensure the layer exists in cache:
                while len(cache._layers) <= lidx:
                    cache._layers.append(
                        cache._make_layer(len(cache._layers))
                    )

                tq_layer = cache._layers[lidx]
                if tq_layer._key_est is None:
                    tq_layer._lazy_init(k_proj, v_proj)

                # Compress the new tokens
                _compress_and_store(tq_layer, k_proj, v_proj)

                # Gather full compressed keys and reconstructed values
                compressed_keys = _gather_compressed_keys(tq_layer)
                value_states = _reconstruct_values(tq_layer)

                # Compute attention using the unbiased estimator
                attn_output = turboquant_attention(
                    query_states=q_proj,
                    compressed_keys=compressed_keys,
                    value_states=value_states,
                    key_estimator=tq_layer._key_est,
                    attention_mask=attention_mask,
                )

                # Reshape back to (bsz, q_len, hidden_size) and project
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, q_len, -1)
                attn_output = attn_output.to(hidden_states.dtype)

                if hasattr(attn_mod, "o_proj"):
                    attn_output = attn_mod.o_proj(attn_output)

                outputs = (attn_output,)
                if output_attentions:
                    outputs += (None,)
                if use_cache:
                    outputs += (cache,)
                return outputs

            return patched_forward

        attn_module.forward = _make_patched_forward(
            layer_idx, original_forward, attn_module
        )

    return model


# ---------------------------------------------------------------------------
# Internal helpers for patch_model_attention
# ---------------------------------------------------------------------------

def _get_inner_model(model: torch.nn.Module) -> torch.nn.Module:
    """Get the inner transformer model (handles wrapper classes)."""
    if hasattr(model, "model"):
        return model.model
    return model


def _get_layers(inner_model: torch.nn.Module) -> list:
    """Get the list of transformer layers."""
    if hasattr(inner_model, "layers"):
        return list(inner_model.layers)
    if hasattr(inner_model, "h"):
        return list(inner_model.h)
    raise AttributeError(
        "Cannot find transformer layers. Expected .model.layers or .model.h"
    )


def _get_attention_module(layer: torch.nn.Module):
    """Get the attention sub-module from a transformer layer."""
    if hasattr(layer, "self_attn"):
        return layer.self_attn
    if hasattr(layer, "attn"):
        return layer.attn
    return None


def _run_qkv_projections(
    attn_mod: torch.nn.Module,
    hidden_states: torch.Tensor,
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run Q, K, V linear projections and reshape to multi-head format.

    Returns:
        (query, key, value) each of shape (batch, n_heads, seq_len, head_dim).
    """
    bsz, seq_len, _ = hidden_states.shape

    if hasattr(attn_mod, "q_proj"):
        q = attn_mod.q_proj(hidden_states)
        k = attn_mod.k_proj(hidden_states)
        v = attn_mod.v_proj(hidden_states)
    elif hasattr(attn_mod, "qkv_proj"):
        # Fused QKV projection (some models)
        qkv = attn_mod.qkv_proj(hidden_states)
        q_size = n_q_heads * head_dim
        kv_size = n_kv_heads * head_dim
        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
    else:
        raise AttributeError(
            "Cannot find Q/K/V projections. Expected q_proj/k_proj/v_proj or qkv_proj"
        )

    q = q.reshape(bsz, seq_len, n_q_heads, head_dim).transpose(1, 2)
    k = k.reshape(bsz, seq_len, n_kv_heads, head_dim).transpose(1, 2)
    v = v.reshape(bsz, seq_len, n_kv_heads, head_dim).transpose(1, 2)

    return q, k, v


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attn_mod: torch.nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to Q and K.

    Tries the model's own ``_apply_rotary_pos_emb`` if available, otherwise
    uses the standard rotate-half implementation.
    """
    # Try model-level function first
    try:
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        return apply_rotary_pos_emb(q, k, cos, sin)
    except (ImportError, AttributeError):
        pass

    try:
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        return apply_rotary_pos_emb(q, k, cos, sin)
    except (ImportError, AttributeError):
        pass

    # Fallback: standard rotate_half implementation
    return _rotate_half_apply(q, cos, sin), _rotate_half_apply(k, cos, sin)


def _rotate_half_apply(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary embedding using the rotate-half convention."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    # cos and sin may need unsqueezing for broadcast
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    return x * cos + rotated * sin


def _compress_and_store(
    tq_layer: "TurboQuantLayer",
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
) -> None:
    """Compress new K/V tokens and store in the TurboQuantLayer.

    This mirrors TurboQuantLayer.update() but avoids the dequantize step
    since we compute attention from compressed data directly.

    Args:
        tq_layer: The TurboQuantLayer to store into.
        k_proj: (batch, n_kv_heads, new_seq, head_dim) new key projections.
        v_proj: (batch, n_kv_heads, new_seq, head_dim) new value projections.
    """
    batch, num_heads, new_seq, head_dim = k_proj.shape

    # Flatten for quantization: (batch * num_heads * new_seq, head_dim)
    keys_flat = k_proj.float().reshape(-1, head_dim)
    vals_flat = v_proj.float().reshape(-1, head_dim)

    # Compress keys with full estimator (MSE + QJL)
    key_comp = tq_layer._key_est.quantize(keys_flat)

    # Compress values with MSE-only PolarQuant
    val_norms = vals_flat.norm(dim=-1, keepdim=True)
    vals_normalized = vals_flat / (val_norms + 1e-8)
    val_indices = tq_layer._val_pq.quantize(vals_normalized)

    # Reshape back to (batch, num_heads, new_seq, ...)
    key_entry = {
        "mse_indices": key_comp["mse_indices"].reshape(batch, num_heads, new_seq, head_dim),
        "qjl_signs": key_comp["qjl_signs"].reshape(batch, num_heads, new_seq, -1),
        "residual_norm": key_comp["residual_norm"].reshape(batch, num_heads, new_seq),
        "vec_norm": key_comp["vec_norm"].reshape(batch, num_heads, new_seq),
    }
    val_entry = {
        "indices": val_indices.reshape(batch, num_heads, new_seq, head_dim),
        "norms": val_norms.squeeze(-1).reshape(batch, num_heads, new_seq),
    }

    tq_layer._key_compressed.append(key_entry)
    tq_layer._value_compressed.append(val_entry)
    tq_layer._seq_len += new_seq


def _gather_compressed_keys(
    tq_layer: "TurboQuantLayer",
) -> Dict[str, torch.Tensor]:
    """Concatenate all compressed key data along the sequence dimension.

    Returns:
        Dict with shapes (batch, n_kv_heads, total_seq, ...).
    """
    return {
        "mse_indices": torch.cat(
            [e["mse_indices"] for e in tq_layer._key_compressed], dim=2
        ),
        "qjl_signs": torch.cat(
            [e["qjl_signs"] for e in tq_layer._key_compressed], dim=2
        ),
        "residual_norm": torch.cat(
            [e["residual_norm"] for e in tq_layer._key_compressed], dim=2
        ),
        "vec_norm": torch.cat(
            [e["vec_norm"] for e in tq_layer._key_compressed], dim=2
        ),
    }


def _reconstruct_values(
    tq_layer: "TurboQuantLayer",
) -> torch.Tensor:
    """Reconstruct all cached values via MSE dequantization.

    Returns:
        (batch, n_kv_heads, total_seq, head_dim) tensor.
    """
    all_indices = torch.cat(
        [e["indices"] for e in tq_layer._value_compressed], dim=2
    )
    all_norms = torch.cat(
        [e["norms"] for e in tq_layer._value_compressed], dim=2
    )

    batch, num_heads, total_seq, head_dim = all_indices.shape

    # Flatten for dequantization
    idx_flat = all_indices.reshape(-1, head_dim)
    norms_flat = all_norms.reshape(-1)

    val_recon = tq_layer._val_pq.dequantize(idx_flat)
    val_recon = val_recon * norms_flat.unsqueeze(-1)

    return val_recon.reshape(batch, num_heads, total_seq, head_dim)
