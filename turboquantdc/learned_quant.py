"""Differentiable attention-optimal quantization.

Learns rotation angles and codebook centroids by back-propagating through
actual attention computation.  The loss is KL divergence between FP16 and
quantized softmax distributions -- the metric that actually matters for
generation quality, *not* MSE reconstruction error.

Key insight: the standard MSE objective is provably WRONG for KV cache
compression.  Better MSE does NOT mean better attention quality (our
block-diagonal rotation experiments prove this).  This module resolves
the discrepancy by directly optimising the thing we care about.

Architecture:
    - Learnable Givens rotation angles (d/2 params, naturally orthogonal)
    - Learnable codebook centroids (2^bits params, initialised from Lloyd-Max)
    - Optional learnable per-coordinate bias (d params)
    - Straight-through estimator for the non-differentiable argmin
    - KL(softmax(Q@K^T), softmax(Q@K_hat^T)) as the training loss

References:
    - TurboQuant paper (arxiv 2504.19874) -- Stage 1 baseline
    - Straight-through estimator (Bengio et al. 2013)
    - RotorQuant / block_rotation.py -- Givens parameterisation
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codebook import solve_lloyd_max


# ---------------------------------------------------------------------------
# Differentiable Givens rotation
# ---------------------------------------------------------------------------

def givens_rotate(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Apply d/2 independent Givens rotations parameterised by angles.

    Each pair (x_{2i}, x_{2i+1}) is rotated by angle theta_i:
        y_{2i}   = cos(theta_i)*x_{2i} - sin(theta_i)*x_{2i+1}
        y_{2i+1} = sin(theta_i)*x_{2i} + cos(theta_i)*x_{2i+1}

    Fully differentiable w.r.t. both x and angles (torch.sin/cos autograd).

    Args:
        x: (..., d) input vectors.
        angles: (d//2,) rotation angles.

    Returns:
        Rotated vectors, same shape as x.
    """
    d = x.shape[-1]
    n_pairs = d // 2
    n_paired = n_pairs * 2

    paired = x[..., :n_paired]
    shape = paired.shape
    pairs = paired.reshape(*shape[:-1], n_pairs, 2)

    c = angles.cos()  # (n_pairs,)
    s = angles.sin()

    v0 = pairs[..., 0]
    v1 = pairs[..., 1]

    r0 = c * v0 - s * v1
    r1 = s * v0 + c * v1

    result = torch.stack([r0, r1], dim=-1).reshape(shape)

    if n_paired < d:
        result = torch.cat([result, x[..., n_paired:]], dim=-1)
    return result


def givens_unrotate(y: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Inverse Givens rotation (transpose = negate sin).

    Args:
        y: (..., d) rotated vectors.
        angles: (d//2,) rotation angles.

    Returns:
        Unrotated vectors, same shape.
    """
    d = y.shape[-1]
    n_pairs = d // 2
    n_paired = n_pairs * 2

    paired = y[..., :n_paired]
    shape = paired.shape
    pairs = paired.reshape(*shape[:-1], n_pairs, 2)

    c = angles.cos()
    s = angles.sin()

    v0 = pairs[..., 0]
    v1 = pairs[..., 1]

    r0 = c * v0 + s * v1
    r1 = -s * v0 + c * v1

    result = torch.stack([r0, r1], dim=-1).reshape(shape)

    if n_paired < d:
        result = torch.cat([result, y[..., n_paired:]], dim=-1)
    return result


# ---------------------------------------------------------------------------
# Straight-through quantisation
# ---------------------------------------------------------------------------

def straight_through_quantize(
    x: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """Quantize with straight-through estimator for gradient flow.

    Forward:  hard assignment (argmin distance to centroids)
    Backward: gradient flows to BOTH x (via STE) and centroids (via
              soft one-hot indexing)

    The trick: we construct a soft one-hot vector from the hard indices
    using detach arithmetic, so that:
    - Forward computes centroids[hard_index] exactly
    - Backward treats the one-hot as if it were a soft assignment,
      allowing gradient to reach centroids via the dot product

    Args:
        x: (..., d) continuous values to quantise.
        centroids: (n_levels,) sorted centroid values.

    Returns:
        Quantised values, same shape as x.  Gradients flow to both
        x (via straight-through) and centroids (via one-hot matmul).
    """
    # Hard assignment: nearest centroid per element
    # x: (..., d), centroids: (n_levels,)
    dists = (x.unsqueeze(-1) - centroids).abs()  # (..., d, n_levels)
    indices = dists.argmin(dim=-1)                # (..., d)

    # One-hot encoding of hard indices -- differentiable path to centroids
    n_levels = centroids.shape[0]
    one_hot = F.one_hot(indices, n_levels).float()  # (..., d, n_levels)

    # Centroid lookup via matmul: differentiable w.r.t. centroids
    x_quant = (one_hot @ centroids)  # (..., d)

    # Combined STE: gradient flows to BOTH x and centroids.
    # Forward: x_quant (hard quantised value)
    # Backward w.r.t. x: identity (STE -- gradient passes through as if no quantisation)
    # Backward w.r.t. centroids: via one_hot @ centroids
    #
    # Decomposition:  result = (x - x.detach()) + x_quant
    # Forward: 0 + x_quant = x_quant  (correct)
    # d(result)/d(x) = 1  (STE identity)
    # d(result)/d(centroids) = d(x_quant)/d(centroids)  (centroid gradient flows)
    return (x - x.detach()) + x_quant


# ---------------------------------------------------------------------------
# LearnedQuantizer
# ---------------------------------------------------------------------------

class LearnedQuantizer(nn.Module):
    """Differentiable attention-optimal KV cache quantizer.

    Learns Givens rotation angles (and optionally codebook centroids) by
    minimising KL(softmax(Q @ K^T), softmax(Q @ K_hat^T)) -- the actual
    attention distribution divergence, not MSE.

    The optimisation loop runs during *calibration* (a short forward pass
    on a few hundred tokens).  After calibration the learned parameters are
    frozen and the quantiser operates as a standard encode/decode codec.

    By default only rotation angles are learned.  The codebook centroids
    are fixed at their Lloyd-Max optimal values, which generalise well
    since the rotation determines the distribution seen by the codebook.
    Set ``learn_centroids=True`` to also learn centroids (may overfit on
    short calibration).

    Args:
        d: Head dimension (e.g. 128).
        bits: Bits per coordinate for the codebook (1-4 typical).
        center: Subtract running mean before quantisation (shift-invariance).
        learn_centroids: Whether to include centroids in the learnable params.
        seed: Random seed for initial angle generation.
        device: Target device.
    """

    def __init__(
        self,
        d: int,
        bits: int = 3,
        center: bool = True,
        learn_centroids: bool = False,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.d = d
        self.bits = bits
        self.n_levels = 1 << bits
        self.center = center
        self.learn_centroids = learn_centroids

        n_pairs = d // 2

        # Learnable rotation angles -- initialised randomly
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        init_angles = torch.rand(n_pairs, generator=gen) * 2 * math.pi
        self.rotation_angles = nn.Parameter(init_angles.to(device))

        # Codebook centroids -- initialised from Lloyd-Max
        lm_centroids, _ = solve_lloyd_max(d, bits)
        if learn_centroids:
            self.centroids = nn.Parameter(lm_centroids.to(device))
        else:
            self.register_buffer("centroids", lm_centroids.to(device))

        # Optional learnable per-coordinate bias
        self.mean_correction = nn.Parameter(
            torch.zeros(d, device=device)
        ) if center else None

        # Running mean for center mode (not learnable)
        self.register_buffer("running_mean", torch.zeros(d, device=device))
        self.register_buffer(
            "running_count", torch.tensor(0, dtype=torch.long, device=device)
        )

    def _update_running_mean(self, x: torch.Tensor) -> None:
        """Online Welford update of running mean from new batch."""
        # x: (batch, d) or (d,)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch = x.shape[0]
        new_sum = x.sum(dim=0)
        old_n = self.running_count.item()
        new_n = old_n + batch
        if old_n == 0:
            self.running_mean.copy_(new_sum / new_n)
        else:
            self.running_mean.copy_(
                (self.running_mean * old_n + new_sum) / new_n
            )
        self.running_count.fill_(new_n)

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Quantise keys, returning compressed representation.

        Non-differentiable (hard argmin).  Use forward() for training.

        Args:
            x: (batch, d) or (d,) key vectors.

        Returns:
            Dict with: indices (int), norms (float), mean (float).
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        x = x.float()

        # Mean removal
        if self.center:
            self._update_running_mean(x)
            mean = self.running_mean.detach()
            x_c = x - mean
        else:
            mean = torch.zeros(self.d, device=x.device)
            x_c = x

        # Normalise
        norms = x_c.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x_c / norms

        # Rotate
        x_rot = givens_rotate(x_unit, self.rotation_angles.detach())

        # Hard quantise (no straight-through for inference)
        dists = (x_rot.unsqueeze(-1) - self.centroids.detach()).abs()
        indices = dists.argmin(dim=-1)

        result = {
            "indices": indices,
            "norms": norms.squeeze(-1),
            "mean": mean,
        }
        if squeeze:
            result = {k: v.squeeze(0) if v.dim() > 0 and v.shape[0] == 1 else v
                      for k, v in result.items()}
        return result

    def decode(self, compressed: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Reconstruct keys from compressed representation.

        Args:
            compressed: Output of encode().

        Returns:
            Reconstructed key vectors.
        """
        indices = compressed["indices"]
        norms = compressed["norms"]
        mean = compressed["mean"]

        squeeze = indices.dim() == 1
        if squeeze:
            indices = indices.unsqueeze(0)
            norms = norms.unsqueeze(0)

        x_quant_rot = self.centroids.detach()[indices]
        x_recon = givens_unrotate(x_quant_rot, self.rotation_angles.detach())
        x_recon = x_recon * norms.unsqueeze(-1) + mean

        if squeeze:
            x_recon = x_recon.squeeze(0)
        return x_recon

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable forward pass: quantise + dequantise.

        Uses straight-through estimator so gradients flow through the
        non-differentiable argmin to both rotation_angles and centroids.

        Args:
            x: (batch, d) key vectors in original space.

        Returns:
            x_recon: (batch, d) reconstructed key vectors.
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)

        x = x.float()

        # Mean removal
        if self.center and self.mean_correction is not None:
            mean = self.running_mean.detach() + self.mean_correction
        elif self.center:
            mean = self.running_mean.detach()
        else:
            mean = torch.zeros(self.d, device=x.device, dtype=x.dtype)

        x_c = x - mean

        # Normalise
        norms = x_c.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x_c / norms

        # Differentiable Givens rotation
        x_rot = givens_rotate(x_unit, self.rotation_angles)

        # Straight-through quantisation
        x_quant = straight_through_quantize(x_rot, self.centroids)

        # Inverse rotation
        x_recon = givens_unrotate(x_quant, self.rotation_angles)

        # Rescale + add mean
        x_recon = x_recon * norms + mean

        if squeeze:
            x_recon = x_recon.squeeze(0)
        return x_recon

    def attention_loss(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute KL divergence between FP16 and quantised attention.

        Args:
            queries: (n_q, d) query vectors.
            keys: (seq_len, d) key vectors.
            temperature: Softmax temperature (lower = sharper).

        Returns:
            Scalar KL divergence loss.
        """
        d = queries.shape[-1]
        scale = temperature / math.sqrt(d)

        # True attention
        logits_true = queries @ keys.T * scale
        attn_true = F.softmax(logits_true, dim=-1)

        # Quantised attention (differentiable)
        keys_quant = self.forward(keys)
        logits_quant = queries @ keys_quant.T * scale
        log_attn_quant = F.log_softmax(logits_quant, dim=-1)

        # KL(true || quant) = sum p_true * (log p_true - log p_quant)
        loss = F.kl_div(log_attn_quant, attn_true, reduction="batchmean")
        return loss

    def calibrate(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        lr: float = 0.01,
        steps: int = 50,
        temperature: float = 1.0,
        verbose: bool = False,
    ) -> List[float]:
        """Optimise rotation angles and centroids on calibration data.

        Args:
            queries: (n_q, d) calibration query vectors.
            keys: (seq_len, d) calibration key vectors.
            lr: Adam learning rate.
            steps: Number of optimisation steps.
            temperature: Softmax temperature.
            verbose: Print loss every 10 steps.

        Returns:
            List of loss values at each step.
        """
        queries = queries.float().to(self.rotation_angles.device)
        keys = keys.float().to(self.rotation_angles.device)

        # Update running mean from calibration keys
        self._update_running_mean(keys)

        params = [self.rotation_angles]
        if self.learn_centroids and isinstance(self.centroids, nn.Parameter):
            params.append(self.centroids)
        if self.mean_correction is not None:
            params.append(self.mean_correction)

        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps, eta_min=lr * 0.01
        )

        losses = []
        best_loss = float("inf")
        best_state = {
            "angles": self.rotation_angles.data.clone(),
        }
        if self.learn_centroids and isinstance(self.centroids, nn.Parameter):
            best_state["centroids"] = self.centroids.data.clone()
        if self.mean_correction is not None:
            best_state["correction"] = self.mean_correction.data.clone()

        for step in range(steps):
            optimizer.zero_grad()
            loss = self.attention_loss(queries, keys, temperature)
            loss.backward()

            # Gradient clipping to prevent overshooting
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

            optimizer.step()
            scheduler.step()

            current_loss = loss.item()
            losses.append(current_loss)

            # Track best parameters
            if current_loss < best_loss:
                best_loss = current_loss
                best_state["angles"] = self.rotation_angles.data.clone()
                if self.learn_centroids and isinstance(self.centroids, nn.Parameter):
                    best_state["centroids"] = self.centroids.data.clone()
                if self.mean_correction is not None:
                    best_state["correction"] = self.mean_correction.data.clone()

            if verbose and (step + 1) % 10 == 0:
                print(f"  Step {step+1:3d}/{steps}: KL loss = {current_loss:.6f}")

        # Restore best parameters
        self.rotation_angles.data.copy_(best_state["angles"])
        if "centroids" in best_state:
            self.centroids.data.copy_(best_state["centroids"])
            # Sort learned centroids for consistent inference
            with torch.no_grad():
                sorted_c, _ = self.centroids.data.sort()
                self.centroids.data.copy_(sorted_c)
        if "correction" in best_state and self.mean_correction is not None:
            self.mean_correction.data.copy_(best_state["correction"])

        return losses

    def calibrate_from_model(
        self,
        model,
        tokenizer,
        text: str = (
            "The quick brown fox jumps over the lazy dog. "
            "A large language model is a neural network trained on "
            "vast amounts of text data. The transformer architecture "
            "uses attention mechanisms to process sequences in parallel."
        ),
        layer_idx: int = 0,
        n_tokens: int = 128,
        lr: float = 0.01,
        steps: int = 50,
        verbose: bool = False,
    ) -> List[float]:
        """Extract Q/K from a real model forward pass and calibrate.

        Args:
            model: HuggingFace causal LM.
            tokenizer: Corresponding tokenizer.
            text: Calibration text.
            layer_idx: Which layer to calibrate on.
            n_tokens: Max tokens from the text.
            lr: Learning rate.
            steps: Optimisation steps.
            verbose: Print progress.

        Returns:
            List of loss values.
        """
        device = next(model.parameters()).device

        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=n_tokens,
            truncation=True,
        ).to(device)

        # Hook to capture Q and K
        captured = {}

        def hook_fn(module, input, output):
            # HF attention: output is (attn_output, attn_weights, past_kv)
            # We need to capture from input -- the Q, K, V projected tensors
            captured["output"] = output

        # Attach hook to the target attention layer
        attn_layers = []
        for name, mod in model.named_modules():
            if hasattr(mod, "q_proj") and hasattr(mod, "k_proj"):
                attn_layers.append((name, mod))

        if layer_idx >= len(attn_layers):
            raise ValueError(
                f"layer_idx={layer_idx} but model has {len(attn_layers)} attention layers"
            )

        target_name, target_attn = attn_layers[layer_idx]

        # Run forward pass with output_attentions to get KV cache
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=False,
                use_cache=True,
            )

        # Extract K and Q from the cache / by re-running the projection
        past_kv = outputs.past_key_values
        if past_kv is not None:
            # past_kv[layer_idx] = (K, V) each (batch, n_heads, seq, head_dim)
            if hasattr(past_kv, "key_cache"):
                # DynamicCache
                K_all = past_kv.key_cache[layer_idx]
                V_all = past_kv.value_cache[layer_idx]
            elif isinstance(past_kv, (list, tuple)):
                K_all = past_kv[layer_idx][0]
                V_all = past_kv[layer_idx][1]
            else:
                raise RuntimeError(f"Unsupported past_kv type: {type(past_kv)}")

            # Take head 0 for calibration: (seq, head_dim)
            K = K_all[0, 0].float()  # (seq, d)

            # Re-derive Q from input hidden states (approximate: use same projection)
            hidden = outputs.hidden_states[layer_idx] if hasattr(outputs, "hidden_states") and outputs.hidden_states else None
            if hidden is not None:
                Q_proj = target_attn.q_proj(hidden[0]).float()
                n_heads = K_all.shape[1]
                head_dim = K_all.shape[3]
                Q = Q_proj.reshape(-1, n_heads, head_dim)[:, 0, :]
            else:
                # Fallback: use K as proxy for Q (common in self-attention)
                Q = K.clone()
        else:
            raise RuntimeError("Model did not return past_key_values")

        return self.calibrate(
            queries=Q,
            keys=K,
            lr=lr,
            steps=steps,
            verbose=verbose,
        )


# ---------------------------------------------------------------------------
# Multi-layer calibration helper
# ---------------------------------------------------------------------------

def calibrate_all_layers(
    model,
    tokenizer,
    text: str = (
        "The quick brown fox jumps over the lazy dog. "
        "A large language model compresses knowledge into parameters. "
        "Attention is all you need for sequence transduction tasks."
    ),
    bits: int = 3,
    n_tokens: int = 128,
    lr: float = 0.01,
    steps: int = 50,
    device: str = "cuda",
    verbose: bool = False,
) -> Dict[int, LearnedQuantizer]:
    """Calibrate a LearnedQuantizer for every attention layer.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        text: Calibration text.
        bits: Bits per coordinate.
        n_tokens: Max calibration tokens.
        lr: Learning rate.
        steps: Optimisation steps per layer.
        device: Device.
        verbose: Print progress.

    Returns:
        Dict mapping layer_idx -> calibrated LearnedQuantizer.
    """
    model_device = next(model.parameters()).device

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=n_tokens,
        truncation=True,
    ).to(model_device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=False,
            use_cache=True,
            output_hidden_states=True,
        )

    past_kv = outputs.past_key_values

    # Count attention layers
    attn_layers = []
    for name, mod in model.named_modules():
        if hasattr(mod, "q_proj") and hasattr(mod, "k_proj"):
            attn_layers.append((name, mod))

    n_layers = len(attn_layers)
    quantizers = {}

    for layer_idx in range(n_layers):
        if hasattr(past_kv, "key_cache"):
            K_all = past_kv.key_cache[layer_idx]
        elif isinstance(past_kv, (list, tuple)):
            K_all = past_kv[layer_idx][0]
        else:
            continue

        head_dim = K_all.shape[3]
        K = K_all[0, 0].float().to(device)

        # Use K as proxy for Q (self-attention)
        Q = K.clone()

        lq = LearnedQuantizer(
            d=head_dim,
            bits=bits,
            center=True,
            seed=42 + layer_idx,
            device=device,
        )
        losses = lq.calibrate(Q, K, lr=lr, steps=steps, verbose=verbose)

        if verbose:
            print(f"Layer {layer_idx}: final KL = {losses[-1]:.6f}")

        quantizers[layer_idx] = lq

    return quantizers
