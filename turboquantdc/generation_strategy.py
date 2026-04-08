from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import torch

from .codebook import LloydMaxCodebook
from .rotation import apply_wht_rotation, generate_qjl_matrix, generate_rotation_matrix, generate_wht_rotation

_TRITON_AVAILABLE = False
try:
    from .triton_kernels import triton_quantize as _triton_quantize
    _TRITON_AVAILABLE = True
except (ImportError, RuntimeError):
    pass

try:
    from .triton_kernels import triton_dequantize, triton_dequantize_residual
except (ImportError, RuntimeError):
    pass

# ---------------------------------------------------------------------------
# Layer-adaptive anchor strategy helpers
# ---------------------------------------------------------------------------

# Valid anchor strategies
ANCHOR_STRATEGIES = ("fixed", "boundary", "gradient")


def compute_layer_key_bits(
    layer_idx: int,
    num_layers: int,
    base_bits: int = 3,
) -> int:
    """Compute key bit-width for a layer based on its distance from boundaries.

    Layers near the first/last positions of the transformer stack are more
    sensitive to quantization error (boundary layers handle embedding proximity
    and output head proximity respectively). This function assigns higher
    bit-widths to boundary layers and lower bit-widths to middle layers.

    The distance metric is normalized to [0, 0.5] where 0 means at the
    boundary and 0.5 means exact middle of the stack.

    Args:
        layer_idx: Index of the layer (0-based).
        num_layers: Total number of transformer layers.
        base_bits: Default bit-width for middle layers (default: 3).

    Returns:
        Bit-width for keys at this layer. Values in {base_bits, base_bits+1, 8}.
        Returns 8 (FP16-equivalent) for layers within 10% of boundaries.
        Returns base_bits+1 for layers within 25% of boundaries.
        Returns base_bits for all other (middle) layers.
    """
    if num_layers <= 1:
        return 8  # Single-layer model: always FP16-equivalent

    # Distance from nearest boundary, normalized to [0, 0.5]
    dist = min(layer_idx, num_layers - 1 - layer_idx) / (num_layers / 2)

    if dist < 0.1:  # Within 10% of boundary
        return 8  # FP16-equivalent for keys
    elif dist < 0.25:  # Within 25% of boundary
        return max(base_bits + 1, 4)  # One extra bit, at least 4
    else:
        return base_bits  # Base compression


def compute_anchor_schedule(
    num_layers: int,
    anchor_strategy: str = "fixed",
    anchor_interval: int = 6,
    base_key_bits: int = 3,
) -> List[Tuple[bool, int]]:
    """Compute per-layer (is_fp16, key_bits) schedule for the given strategy.

    Args:
        num_layers: Total transformer layers.
        anchor_strategy: One of "fixed", "boundary", "gradient".
        anchor_interval: Interval for "fixed" strategy (ignored by others).
        base_key_bits: Base key bit-width for compressed layers.

    Returns:
        List of (is_fp16, key_bits) tuples, one per layer.
        When is_fp16 is True, the layer stores raw FP16 (key_bits is ignored).
        When is_fp16 is False, key_bits is the bit-width for that layer's keys.
    """
    if anchor_strategy not in ANCHOR_STRATEGIES:
        raise ValueError(
            f"Unknown anchor_strategy: '{anchor_strategy}'. "
            f"Must be one of {ANCHOR_STRATEGIES}"
        )

    schedule: List[Tuple[bool, int]] = []

    if anchor_strategy == "fixed":
        for i in range(num_layers):
            is_fp16 = anchor_interval > 0 and i % anchor_interval == 0
            schedule.append((is_fp16, base_key_bits))

    elif anchor_strategy == "boundary":
        # First 2 + last 2 layers are always FP16
        for i in range(num_layers):
            is_fp16 = i < 2 or i >= num_layers - 2
            schedule.append((is_fp16, base_key_bits))

    elif anchor_strategy == "gradient":
        # Boundary FP16 + gradient bit allocation for middle layers
        for i in range(num_layers):
            key_bits = compute_layer_key_bits(i, num_layers, base_key_bits)
            is_fp16 = key_bits == 8
            schedule.append((is_fp16, key_bits))

    return schedule


