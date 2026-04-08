"""Fused attention kernel — computes attention directly from compressed data.

The breakthrough: instead of dequantizing keys to FP16 and computing Q @ K^T
(which introduces FP16 rounding that destroys quality at long context), this
kernel computes inner products DIRECTLY from compressed indices in float32.

The algebraic trick (from DEJAN / hackimov):
    <q, R^T @ centroids[idx]> = <R @ q, centroids[idx]>

Pre-rotate queries ONCE, then gather centroids and dot-product. No intermediate
FP16 materialization. No rounding of the reconstructed key vector. The inner
product stays in the compressed domain, computed exactly in float32.

Three variants:
    1. fused_turboquant_attention() — Full MSE + QJL unbiased estimator
    2. fused_mse_attention()        — MSE-only (no QJL), lower variance
    3. patch_model_fused_attention() — Monkey-patch HF model to use fused path

Reference: TurboQuant paper (arxiv 2504.19874), Algorithm 2.
Community: hackimov/turboquant-kv, DEJAN blog, llama.cpp #20969.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch


def fused_turboquant_attention(
    query_states: torch.Tensor,      # (batch, n_q_heads, seq_q, d)
    mse_indices: torch.Tensor,       # (batch, n_kv_heads, seq_kv, d) — int indices
    qjl_signs: torch.Tensor,         # (batch, n_kv_heads, seq_kv, m) — {-1, +1}
    residual_norms: torch.Tensor,    # (batch, n_kv_heads, seq_kv)
    vec_norms: torch.Tensor,         # (batch, n_kv_heads, seq_kv)
    value_states: torch.Tensor,      # (batch, n_kv_heads, seq_kv, d) — FP16 values
    rotation_matrix: torch.Tensor,   # (d, d) — PolarQuant rotation R
    centroids: torch.Tensor,         # (2^b,) — Lloyd-Max centroid values
    qjl_matrix: torch.Tensor,        # (m, d) — QJL projection matrix S
    attention_mask: torch.Tensor = None,
    scale: float = None,
) -> torch.Tensor:
    """Compute attention without ever materializing FP16 keys.

    Instead of dequantizing keys to FP16 and computing Q @ K^T,
    this function:
    1. Pre-rotates queries: q_rot = Q @ R (once, in float32)
    2. Gathers centroids: c[j,coord] = centroids[indices[j,coord]]
    3. Computes MSE IP: score_mse = q_rot @ c^T (in float32)
    4. Pre-projects queries: q_proj = Q @ S^T (once, in float32)
    5. Computes QJL correction: correction = r_norm * sqrt(pi/2)/m * q_proj @ signs^T
    6. Combines: score = vec_norm * (score_mse + correction)

    Step 2-3 is the key difference from the broken approach. Instead of:
        k_recon = R^T @ centroids[idx]  (FP16 materialization)
        score = Q @ k_recon^T            (FP16 matmul)
    We do:
        q_rot = Q @ R                    (float32)
        score = q_rot @ centroids[idx]^T (float32, via gather)

    Mathematically identical, but avoids FP16 rounding of the reconstructed key.

    Args:
        query_states: (batch, n_q_heads, seq_q, d) query tensor.
        mse_indices: (batch, n_kv_heads, seq_kv, d) codebook indices.
        qjl_signs: (batch, n_kv_heads, seq_kv, m) sign bits in {-1, +1}.
        residual_norms: (batch, n_kv_heads, seq_kv) residual norms.
        vec_norms: (batch, n_kv_heads, seq_kv) original key norms (or corrected).
        value_states: (batch, n_kv_heads, seq_kv, d) reconstructed values.
        rotation_matrix: (d, d) PolarQuant rotation matrix R (Pi in the paper).
        centroids: (2^b,) Lloyd-Max centroid values, float32.
        qjl_matrix: (m, d) QJL projection matrix S.
        attention_mask: Optional additive mask, -inf for masked positions.
        scale: Attention score scaling factor. Defaults to 1/sqrt(d).

    Returns:
        (batch, n_q_heads, seq_q, d) attention output tensor.
    """
    batch, n_q_heads, seq_q, d = query_states.shape
    n_kv_heads = mse_indices.shape[1]
    seq_kv = mse_indices.shape[2]
    heads_per_kv = n_q_heads // n_kv_heads
    m = qjl_matrix.shape[0]

    if scale is None:
        scale = 1.0 / (d ** 0.5)

    # -- Step 1: Pre-rotate ALL queries through R (once) --
    # R is (d, d). Q is (batch, n_q_heads, seq_q, d).
    # q_rot = Q @ R^T (rotate into codebook space, same as PolarQuant.rotate)
    # Note: PolarQuant.rotate does x @ Pi.T, so q_rot = Q @ Pi.T
    q_rot = torch.matmul(query_states.float(), rotation_matrix.float().T)

    # -- Step 2-3: MSE inner product via centroid gather --
    # indices: (batch, n_kv_heads, seq_kv, d), each value indexes into centroids
    # Gather: c[b,h,s,coord] = centroids[indices[b,h,s,coord]]
    c_gathered = centroids.float()[mse_indices.long()]  # (batch, n_kv_heads, seq_kv, d) float32

    # GQA expansion for centroid-gathered keys
    if heads_per_kv > 1:
        c_gathered = c_gathered.repeat_interleave(heads_per_kv, dim=1)

    # MSE inner product: q_rot @ c_gathered^T
    # (batch, n_q_heads, seq_q, d) @ (batch, n_q_heads, d, seq_kv)
    # -> (batch, n_q_heads, seq_q, seq_kv)
    score_mse = torch.matmul(q_rot, c_gathered.transpose(-1, -2))

    # -- Step 4-5: QJL correction --
    # Pre-project queries: q_proj = Q @ S^T -> (batch, n_q_heads, seq_q, m)
    q_proj = torch.matmul(query_states.float(), qjl_matrix.float().T)

    # GQA expansion for signs
    signs_expanded = qjl_signs.float()
    if heads_per_kv > 1:
        signs_expanded = signs_expanded.repeat_interleave(heads_per_kv, dim=1)

    # qjl_ip = q_proj @ signs^T -> (batch, n_q_heads, seq_q, seq_kv)
    qjl_ip = torch.matmul(q_proj, signs_expanded.transpose(-1, -2))

    qjl_scale = math.sqrt(math.pi / 2.0) / m

    # r_norm expansion for GQA
    r_norm = residual_norms.float()
    if heads_per_kv > 1:
        r_norm = r_norm.repeat_interleave(heads_per_kv, dim=1)

    # correction = r_norm * sqrt(pi/2)/m * qjl_ip
    correction = qjl_scale * r_norm.unsqueeze(2) * qjl_ip

    # -- Step 6: Combine and scale --
    v_norm = vec_norms.float()
    if heads_per_kv > 1:
        v_norm = v_norm.repeat_interleave(heads_per_kv, dim=1)

    scores = v_norm.unsqueeze(2) * (score_mse + correction)
    scores = scores * scale

    # -- Mask + softmax + value weighted sum --
    if attention_mask is not None:
        scores = _apply_attention_mask(scores, attention_mask, seq_q, seq_kv)

    weights = torch.softmax(scores, dim=-1, dtype=torch.float32)

    # Values: expand for GQA
    if heads_per_kv > 1:
        v_expanded = value_states.repeat_interleave(heads_per_kv, dim=1)
    else:
        v_expanded = value_states

    output = torch.matmul(weights.to(v_expanded.dtype), v_expanded)
    return output


def fused_mse_attention(
    query_states: torch.Tensor,      # (batch, n_q_heads, seq_q, d)
    mse_indices: torch.Tensor,       # (batch, n_kv_heads, seq_kv, d) — int indices
    vec_norms: torch.Tensor,         # (batch, n_kv_heads, seq_kv)
    value_states: torch.Tensor,      # (batch, n_kv_heads, seq_kv, d) — FP16 values
    rotation_matrix: torch.Tensor,   # (d, d) — PolarQuant rotation R
    centroids: torch.Tensor,         # (2^b,) — Lloyd-Max centroid values
    attention_mask: torch.Tensor = None,
    scale: float = None,
) -> torch.Tensor:
    """Compute attention from compressed indices, MSE-only (no QJL).

    Same as fused_turboquant_attention but without the QJL correction term.
    Lower variance scores at the cost of bias. Often better for generation
    at higher bit-widths (4-bit) where MSE error is already small.

    Args:
        query_states: (batch, n_q_heads, seq_q, d) query tensor.
        mse_indices: (batch, n_kv_heads, seq_kv, d) codebook indices.
        vec_norms: (batch, n_kv_heads, seq_kv) key norms (or corrected norms).
        value_states: (batch, n_kv_heads, seq_kv, d) reconstructed values.
        rotation_matrix: (d, d) PolarQuant rotation matrix R.
        centroids: (2^b,) Lloyd-Max centroid values, float32.
        attention_mask: Optional additive mask.
        scale: Attention score scaling factor. Defaults to 1/sqrt(d).

    Returns:
        (batch, n_q_heads, seq_q, d) attention output tensor.
    """
    batch, n_q_heads, seq_q, d = query_states.shape
    n_kv_heads = mse_indices.shape[1]
    seq_kv = mse_indices.shape[2]
    heads_per_kv = n_q_heads // n_kv_heads

    if scale is None:
        scale = 1.0 / (d ** 0.5)

    # Step 1: Pre-rotate queries
    q_rot = torch.matmul(query_states.float(), rotation_matrix.float().T)

    # Step 2: Gather centroids
    c_gathered = centroids.float()[mse_indices.long()]

    # GQA expansion
    if heads_per_kv > 1:
        c_gathered = c_gathered.repeat_interleave(heads_per_kv, dim=1)

    # Step 3: MSE inner product
    score_mse = torch.matmul(q_rot, c_gathered.transpose(-1, -2))

    # Scale by key norms
    v_norm = vec_norms.float()
    if heads_per_kv > 1:
        v_norm = v_norm.repeat_interleave(heads_per_kv, dim=1)

    scores = v_norm.unsqueeze(2) * score_mse
    scores = scores * scale

    # Mask + softmax + value weighted sum
    if attention_mask is not None:
        scores = _apply_attention_mask(scores, attention_mask, seq_q, seq_kv)

    weights = torch.softmax(scores, dim=-1, dtype=torch.float32)

    if heads_per_kv > 1:
        v_expanded = value_states.repeat_interleave(heads_per_kv, dim=1)
    else:
        v_expanded = value_states

    output = torch.matmul(weights.to(v_expanded.dtype), v_expanded)
    return output


# ---------------------------------------------------------------------------
# Norm correction helper
# ---------------------------------------------------------------------------

def compute_norm_correction(
    keys_flat: torch.Tensor,
    polar,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute corrected norms: original_norm / reconstruction_norm.

    This corrects the magnitude distortion introduced by quantization.
    Instead of storing ||k||, we store ||k|| / ||k_hat|| so that
    k_recon = k_hat * corrected_norm has the correct magnitude.

    Per spiritbuun's llama.cpp finding, this gives -1.17% perplexity improvement.

    Args:
        keys_flat: (N, d) key vectors.
        polar: PolarQuant instance.

    Returns:
        (indices, corrected_norms) where corrected_norms = ||k|| / ||k_hat||.
    """
    # Original norms
    key_norms = keys_flat.norm(dim=-1, keepdim=True)  # (N, 1)
    keys_normalized = keys_flat / (key_norms + 1e-8)

    # Quantize
    indices = polar.quantize(keys_normalized)

    # Compute reconstruction norm
    k_recon = polar.dequantize(indices)
    recon_norms = k_recon.norm(dim=-1, keepdim=True)  # (N, 1)

    # Corrected norm: original / reconstruction
    corrected_norms = key_norms / (recon_norms + 1e-8)  # (N, 1)

    return indices, corrected_norms.squeeze(-1)


# ---------------------------------------------------------------------------
# Internal mask helper
# ---------------------------------------------------------------------------

def _apply_attention_mask(
    scores: torch.Tensor,
    attention_mask: torch.Tensor,
    seq_q: int,
    seq_kv: int,
) -> torch.Tensor:
    """Apply attention mask with shape handling for HF compatibility."""
    if attention_mask.dim() == 2:
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
    elif attention_mask.dim() == 3:
        attention_mask = attention_mask.unsqueeze(1)

    mq = attention_mask.shape[-2]
    mkv = attention_mask.shape[-1]

    if mq > seq_q:
        attention_mask = attention_mask[..., -seq_q:, :]
    if mkv > seq_kv:
        attention_mask = attention_mask[..., -seq_kv:]
    elif mkv < seq_kv:
        pad = torch.zeros(
            *attention_mask.shape[:-1], seq_kv - mkv,
            dtype=attention_mask.dtype, device=attention_mask.device,
        )
        attention_mask = torch.cat([pad, attention_mask], dim=-1)

    return scores + attention_mask.float()


# ---------------------------------------------------------------------------
# Compression helpers for the fused path
# ---------------------------------------------------------------------------

def _fused_compress_and_store(
    tq_layer,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    use_norm_correction: bool = True,
) -> None:
    """Compress new K/V tokens with optional norm correction.

    Similar to custom_attention._compress_and_store but adds the norm
    correction optimization (store ||k|| / ||k_hat|| instead of ||k||).

    Args:
        tq_layer: TurboQuantLayer to store into.
        k_proj: (batch, n_kv_heads, new_seq, head_dim) new key projections.
        v_proj: (batch, n_kv_heads, new_seq, head_dim) new value projections.
        use_norm_correction: Whether to apply norm correction.
    """
    batch, num_heads, new_seq, head_dim = k_proj.shape

    keys_flat = k_proj.float().reshape(-1, head_dim)
    vals_flat = v_proj.float().reshape(-1, head_dim)

    if tq_layer.mse_only:
        # MSE-only mode
        pq = tq_layer._key_pq
        if use_norm_correction:
            key_indices, corrected_norms = compute_norm_correction(keys_flat, pq)
        else:
            key_norms = keys_flat.norm(dim=-1, keepdim=True)
            keys_normalized = keys_flat / (key_norms + 1e-8)
            key_indices = pq.quantize(keys_normalized)
            corrected_norms = key_norms.squeeze(-1)

        key_entry = {
            "mse_indices": key_indices.reshape(batch, num_heads, new_seq, head_dim),
            "vec_norm": corrected_norms.reshape(batch, num_heads, new_seq),
        }
    else:
        # Full TurboQuant (MSE + QJL)
        est = tq_layer._key_est

        # Store original norm and normalize
        vec_norm = keys_flat.norm(dim=-1, keepdim=True)
        x_normalized = keys_flat / (vec_norm + 1e-8)

        # Stage 1: MSE quantization
        mse_indices = est.polar.quantize(x_normalized)
        x_mse = est.polar.dequantize(mse_indices)

        # Norm correction: store ||k|| / ||k_hat|| instead of ||k||
        if use_norm_correction:
            recon_norm = x_mse.norm(dim=-1, keepdim=True)
            corrected_vec_norm = vec_norm / (recon_norm + 1e-8)
        else:
            corrected_vec_norm = vec_norm

        # Compute residual
        residual = x_normalized - x_mse
        residual_norm = residual.norm(dim=-1)

        # Stage 2: QJL on residual
        qjl_signs = est.qjl.project_and_sign(residual)

        key_entry = {
            "mse_indices": mse_indices.reshape(batch, num_heads, new_seq, head_dim),
            "qjl_signs": qjl_signs.reshape(batch, num_heads, new_seq, -1),
            "residual_norm": residual_norm.reshape(batch, num_heads, new_seq),
            "vec_norm": corrected_vec_norm.squeeze(-1).reshape(batch, num_heads, new_seq),
        }

    # Compress values with MSE-only PolarQuant
    val_norms = vals_flat.norm(dim=-1, keepdim=True)
    vals_normalized = vals_flat / (val_norms + 1e-8)
    val_indices = tq_layer._val_pq.quantize(vals_normalized)

    val_entry = {
        "indices": val_indices.reshape(batch, num_heads, new_seq, head_dim),
        "norms": val_norms.squeeze(-1).reshape(batch, num_heads, new_seq),
    }

    tq_layer._key_compressed.append(key_entry)
    tq_layer._value_compressed.append(val_entry)
    tq_layer._seq_len += new_seq


def _fused_gather_compressed_keys(
    tq_layer,
) -> Dict[str, torch.Tensor]:
    """Concatenate all compressed key data along the sequence dimension."""
    result = {
        "mse_indices": torch.cat(
            [e["mse_indices"] for e in tq_layer._key_compressed], dim=2
        ),
        "vec_norm": torch.cat(
            [e["vec_norm"] for e in tq_layer._key_compressed], dim=2
        ),
    }

    # QJL fields only present in non-mse_only mode
    if "qjl_signs" in tq_layer._key_compressed[0]:
        result["qjl_signs"] = torch.cat(
            [e["qjl_signs"] for e in tq_layer._key_compressed], dim=2
        )
        result["residual_norm"] = torch.cat(
            [e["residual_norm"] for e in tq_layer._key_compressed], dim=2
        )

    return result


def _fused_reconstruct_values(
    tq_layer,
) -> torch.Tensor:
    """Reconstruct all cached values via MSE dequantization."""
    all_indices = torch.cat(
        [e["indices"] for e in tq_layer._value_compressed], dim=2
    )
    all_norms = torch.cat(
        [e["norms"] for e in tq_layer._value_compressed], dim=2
    )

    batch, num_heads, total_seq, head_dim = all_indices.shape
    idx_flat = all_indices.reshape(-1, head_dim)
    norms_flat = all_norms.reshape(-1)

    val_recon = tq_layer._val_pq.dequantize(idx_flat)
    val_recon = val_recon * norms_flat.unsqueeze(-1)

    return val_recon.reshape(batch, num_heads, total_seq, head_dim)


# ---------------------------------------------------------------------------
# Model patching for fused attention
# ---------------------------------------------------------------------------

def patch_model_fused_attention(
    model: torch.nn.Module,
    cache: "TurboQuantCache",
    use_norm_correction: bool = True,
) -> torch.nn.Module:
    """Monkey-patch a HuggingFace model to use fused TurboQuant attention.

    Unlike patch_model_attention() which dequantizes keys before computing
    Q @ K^T, this patch computes attention scores DIRECTLY from compressed
    indices in float32. This eliminates the FP16 materialization step that
    destroys quality at long context.

    The patch:
    1. Intercepts Q/K/V projections
    2. Applies rotary embeddings to Q and K
    3. Compresses new K/V into the TurboQuantCache
    4. Calls fused_turboquant_attention() or fused_mse_attention() with
       compressed data + pre-computed rotation/centroid/QJL matrices
    5. Returns attention output

    Args:
        model: A HuggingFace CausalLM model (Qwen2, Llama, Mistral, etc.).
        cache: A TurboQuantCache instance.
        use_norm_correction: Whether to use norm correction (recommended).

    Returns:
        The same model object, with attention layers patched in-place.
    """
    from .custom_attention import (
        _apply_rotary_pos_emb,
        _get_attention_module,
        _get_inner_model,
        _get_layers,
        _run_qkv_projections,
    )

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

        def _make_patched_forward(lidx, attn_mod):
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
                bsz, q_len, hidden_size = hidden_states.shape

                q_proj, k_proj, v_proj = _run_qkv_projections(
                    attn_mod, hidden_states, n_q_heads, n_kv_heads, head_dim,
                )

                # Apply rotary embeddings
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

                # Ensure the layer exists in cache
                while len(cache._layers) <= lidx:
                    cache._layers.append(
                        cache._make_layer(len(cache._layers))
                    )

                tq_layer = cache._layers[lidx]
                if tq_layer._key_est is None and tq_layer._key_pq is None:
                    tq_layer._lazy_init(k_proj, v_proj)

                # Compress and store with norm correction
                _fused_compress_and_store(
                    tq_layer, k_proj, v_proj,
                    use_norm_correction=use_norm_correction,
                )

                # Gather compressed data
                compressed_keys = _fused_gather_compressed_keys(tq_layer)
                value_states = _fused_reconstruct_values(tq_layer)

                # Get rotation matrix and centroids for fused computation
                if tq_layer.mse_only:
                    rotation = tq_layer._key_pq.Pi
                    cents = tq_layer._key_pq.centroids
                else:
                    rotation = tq_layer._key_est.polar.Pi
                    cents = tq_layer._key_est.polar.centroids

                # Compute attention via fused path
                if tq_layer.mse_only:
                    attn_output = fused_mse_attention(
                        query_states=q_proj,
                        mse_indices=compressed_keys["mse_indices"],
                        vec_norms=compressed_keys["vec_norm"],
                        value_states=value_states,
                        rotation_matrix=rotation,
                        centroids=cents,
                        attention_mask=attention_mask,
                    )
                else:
                    attn_output = fused_turboquant_attention(
                        query_states=q_proj,
                        mse_indices=compressed_keys["mse_indices"],
                        qjl_signs=compressed_keys["qjl_signs"],
                        residual_norms=compressed_keys["residual_norm"],
                        vec_norms=compressed_keys["vec_norm"],
                        value_states=value_states,
                        rotation_matrix=rotation,
                        centroids=cents,
                        qjl_matrix=tq_layer._key_est.qjl.S,
                        attention_mask=attention_mask,
                    )

                # Reshape back and project
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

        attn_module.forward = _make_patched_forward(layer_idx, attn_module)

    return model
