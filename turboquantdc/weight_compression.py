"""TurboQuant weight compression (TQ-W) for ultra-low-bit model deployment.

Applies TurboQuant's PolarQuant (random orthogonal rotation + Lloyd-Max
quantization) to model weights. Model weight matrices have approximately
Gaussian-distributed entries, and Lloyd-Max minimizes MSE for the exact
distribution -- so this should give better quality than GPTQ/AWQ/GGUF
at the same bit-width.

Architecture:
    CompressedLinear  -- drop-in nn.Linear replacement with quantized weights
    TurboQuantWeightCompressor -- walks a model, compresses all linear layers
    compress_model()  -- top-level API

Layer-adaptive bit allocation ("gradient" strategy):
    First 2 + last 2 layers: 4-bit  (boundary protection)
    Near-boundary layers (within 25%): 3-bit
    Middle layers: 2-bit
    This yields ~2.5 bpw on a 70B model -> ~22GB.

Reference: TurboQuant paper (arxiv 2504.19874), applied to weights
           instead of KV cache vectors.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codebook import LloydMaxCodebook
from .rotation import generate_rotation_matrix

# ---------------------------------------------------------------------------
# Bit allocation strategies
# ---------------------------------------------------------------------------

def compute_weight_bit_schedule(
    num_layers: int,
    target_bpw: float = 2.5,
    strategy: str = "gradient",
    custom_schedule: Optional[List[int]] = None,
) -> List[int]:
    """Compute per-layer bit-width schedule for weight compression.

    Strategies:
        "uniform": Same bit-width for all layers.
        "gradient": Boundary-protective allocation:
            - First 2 + last 2 layers: 4-bit
            - Layers within 25% of boundaries: 3-bit
            - Middle layers: 2-bit
        "custom": User-provided per-layer schedule.

    Args:
        num_layers: Number of transformer layers.
        target_bpw: Target average bits per weight (used for "uniform").
        strategy: One of "uniform", "gradient", "custom".
        custom_schedule: Per-layer bit-widths for "custom" strategy.

    Returns:
        List of per-layer bit-widths (integers in [2, 8]).
    """
    if strategy == "uniform":
        bits = max(2, min(8, round(target_bpw)))
        return [bits] * num_layers

    elif strategy == "gradient":
        schedule = []
        boundary_count = 2  # first 2 and last 2 layers at 4-bit
        near_fraction = 0.25  # layers within 25% of boundaries at 3-bit

        for i in range(num_layers):
            # Distance from nearest boundary (0 or num_layers-1)
            dist_from_start = i
            dist_from_end = num_layers - 1 - i
            dist = min(dist_from_start, dist_from_end)

            if dist < boundary_count:
                schedule.append(4)
            elif dist < num_layers * near_fraction:
                schedule.append(3)
            else:
                schedule.append(2)
        return schedule

    elif strategy == "custom":
        if custom_schedule is None:
            raise ValueError(
                "custom_schedule must be provided for 'custom' strategy"
            )
        if len(custom_schedule) != num_layers:
            raise ValueError(
                f"custom_schedule length ({len(custom_schedule)}) "
                f"must match num_layers ({num_layers})"
            )
        return list(custom_schedule)

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")


def effective_bpw(schedule: List[int]) -> float:
    """Compute the effective (average) bits per weight from a schedule.

    Args:
        schedule: Per-layer bit-widths.

    Returns:
        Average bits per weight across all layers.
    """
    if not schedule:
        return 0.0
    return sum(schedule) / len(schedule)


# ---------------------------------------------------------------------------
# Compressed Linear layer
# ---------------------------------------------------------------------------

class CompressedLinear(nn.Module):
    """Linear layer with TurboQuant-compressed weights.

    Stores quantized indices instead of float weights. On each forward pass,
    dequantizes weights (centroid lookup + inverse rotation) and runs the
    linear operation.

    Storage per layer:
        indices:     out_features * in_features * bits (packed)
        codebook:    2^bits centroids (float32)
        rotation:    generated from seed (not stored as matrix)
        bias:        out_features * 32 bits (if present)

    Args:
        indices: Quantized weight indices, shape (out_features, in_features).
        codebook: LloydMaxCodebook used for quantization.
        rotation_seed: Seed for deterministic rotation matrix regeneration.
        original_shape: (out_features, in_features) of original weight.
        bias: Optional bias vector.
        device: Target device.
    """

    def __init__(
        self,
        indices: torch.Tensor,
        codebook: LloydMaxCodebook,
        rotation_seed: int,
        original_shape: Tuple[int, int],
        bias: Optional[torch.Tensor] = None,
        device: str | torch.device = "cpu",
        row_norms: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.out_features, self.in_features = original_shape
        self.rotation_seed = rotation_seed
        self.codebook = codebook
        self.bits = codebook.bits

        # Store indices as buffer (integer tensor, not a parameter)
        self.register_buffer("indices", indices.to(device))

        # Store codebook centroids as buffer for fast lookup
        self.register_buffer("centroids", codebook.centroids.to(device))

        # Store row norms for rescaling (each row had its norm factored out)
        if row_norms is not None:
            self.register_buffer("row_norms", row_norms.to(device))
        else:
            self.register_buffer(
                "row_norms",
                torch.ones(self.out_features, device=device),
            )

        # Generate rotation matrix once and store as buffer
        Pi = generate_rotation_matrix(self.in_features, seed=rotation_seed, device="cpu")
        self.register_buffer("Pi", Pi.to(device))

        # Bias
        if bias is not None:
            self.register_buffer("bias", bias.to(device))
        else:
            self.bias = None

    def _dequantize(self) -> torch.Tensor:
        """Reconstruct the weight matrix from quantized indices.

        Pipeline: indices -> centroid lookup -> inverse rotation -> rescale.

        Returns:
            Reconstructed weight matrix, shape (out_features, in_features).
        """
        # Centroid lookup: (out_features, in_features) -> float values
        y_hat = self.centroids[self.indices]  # (out, in)
        # Inverse rotation: y_hat @ Pi (since Pi is orthogonal, Pi^{-1} = Pi^T)
        W = y_hat @ self.Pi  # (out, in)
        # Rescale by original row norms
        W = W * self.row_norms.unsqueeze(1)  # (out, 1) broadcast
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: dequantize weights on-the-fly and compute linear.

        Args:
            x: Input tensor, shape (..., in_features).

        Returns:
            Output tensor, shape (..., out_features).
        """
        W = self._dequantize()
        return F.linear(x, W, self.bias)

    def weight_mse(self, original_weight: torch.Tensor) -> float:
        """Compute MSE between original and reconstructed weights.

        Args:
            original_weight: Original weight matrix, shape (out, in).

        Returns:
            Mean squared error as a float.
        """
        W_hat = self._dequantize()
        return (original_weight.to(W_hat.device) - W_hat).pow(2).mean().item()

    def memory_bytes(self) -> Dict[str, int]:
        """Compute memory usage of this compressed layer.

        Returns:
            Dict with index_bits, codebook_bytes, bias_bytes, total_bytes.
        """
        n_elements = self.out_features * self.in_features
        index_bits = n_elements * self.bits
        # Actual storage: indices are stored as int8/int16/int32
        index_bytes = self.indices.numel() * self.indices.element_size()
        codebook_bytes = self.centroids.numel() * self.centroids.element_size()
        norm_bytes = self.row_norms.numel() * self.row_norms.element_size()
        bias_bytes = self.bias.numel() * self.bias.element_size() if self.bias is not None else 0
        # Pi is not counted -- in production it would be regenerated from seed
        return {
            "index_bits": index_bits,
            "index_bytes": index_bytes,
            "codebook_bytes": codebook_bytes,
            "norm_bytes": norm_bytes,
            "bias_bytes": bias_bytes,
            "total_bytes": index_bytes + codebook_bytes + norm_bytes + bias_bytes,
            "theoretical_bits": index_bits + self.out_features * 16,  # indices + norms in fp16
        }

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, bias={self.bias is not None}"
        )


# ---------------------------------------------------------------------------
# Weight compressor
# ---------------------------------------------------------------------------

class TurboQuantWeightCompressor:
    """Compresses all linear layer weights in a HuggingFace model.

    For each weight matrix W of shape (out_features, in_features):
        1. Normalize each row: store ||w_i|| and work with w_i / ||w_i||
        2. Rotate: y_i = w_i_normalized @ Pi.T  (Pi from seed, reused per width)
        3. Quantize each coordinate: idx_ij = nearest centroid
        4. Replace nn.Linear with CompressedLinear

    Rotation matrices and codebooks are generated ONCE per (dimension, bits)
    pair and reused for all layers of the same width.

    Args:
        target_bpw: Target average bits per weight.
        strategy: Bit allocation strategy ("uniform", "gradient", "custom").
        base_seed: Base seed for rotation matrix generation.
    """

    def __init__(
        self,
        target_bpw: float = 2.5,
        strategy: str = "gradient",
        base_seed: int = 42,
    ):
        self.target_bpw = target_bpw
        self.strategy = strategy
        self.base_seed = base_seed

        # Caches: avoid recomputing for same (dim, bits) pair
        self._codebook_cache: Dict[Tuple[int, int], LloydMaxCodebook] = {}
        self._rotation_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def _get_codebook(self, d: int, bits: int) -> LloydMaxCodebook:
        """Get or create a codebook for the given (dimension, bits) pair."""
        key = (d, bits)
        if key not in self._codebook_cache:
            self._codebook_cache[key] = LloydMaxCodebook(d, bits)
        return self._codebook_cache[key]

    def _get_rotation(self, d: int, seed: int) -> torch.Tensor:
        """Get or create a rotation matrix for the given (dimension, seed) pair."""
        key = (d, seed)
        if key not in self._rotation_cache:
            self._rotation_cache[key] = generate_rotation_matrix(d, seed=seed, device="cpu")
        return self._rotation_cache[key]

    def _find_linear_layers(self, model: nn.Module) -> List[Tuple[str, nn.Module, nn.Linear]]:
        """Find all nn.Linear modules in the model with their parent paths.

        Returns:
            List of (full_name, parent_module, linear_module) tuples.
        """
        result = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Find parent module
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent_name, attr_name = parts
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model
                    attr_name = name
                result.append((name, parent, attr_name, module))
        return result

    def _detect_num_layers(self, model: nn.Module) -> int:
        """Detect the number of transformer layers in a HuggingFace model.

        Looks for common patterns: model.layers, model.transformer.h,
        model.model.layers, etc.

        Returns:
            Number of layers detected, or 1 if structure is unrecognized.
        """
        # Try common HuggingFace model structures
        for attr_path in ["model.layers", "transformer.h", "model.decoder.layers", "encoder.layer"]:
            obj = model
            try:
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                return len(obj)
            except AttributeError:
                continue
        return 1

    def _layer_index_for_name(self, name: str, num_layers: int) -> int:
        """Extract transformer layer index from a module name.

        Parses patterns like 'model.layers.5.self_attn.q_proj' -> 5.
        Returns 0 for modules outside numbered layers (e.g., embeddings).

        Args:
            name: Full dotted module name.
            num_layers: Total number of layers.

        Returns:
            Layer index in [0, num_layers).
        """
        parts = name.split(".")
        for part in parts:
            if part.isdigit():
                idx = int(part)
                if 0 <= idx < num_layers:
                    return idx
        # For non-layer modules (lm_head, embed), treat as boundary
        return 0

    def compress_linear(
        self,
        linear: nn.Linear,
        bits: int,
        rotation_seed: int,
        device: str | torch.device = "cpu",
    ) -> CompressedLinear:
        """Compress a single nn.Linear layer.

        Args:
            linear: The linear layer to compress.
            bits: Number of bits per weight coordinate.
            rotation_seed: Seed for rotation matrix (deterministic).
            device: Device for the compressed layer.

        Returns:
            CompressedLinear replacement.
        """
        W = linear.weight.data.float()  # (out_features, in_features)
        out_features, in_features = W.shape

        # Step 1: Factor out row norms
        row_norms = W.norm(dim=1)  # (out_features,)
        W_normalized = W / (row_norms.unsqueeze(1) + 1e-8)  # (out, in)

        # Step 2: Rotate -- each row is a vector
        Pi = self._get_rotation(in_features, rotation_seed)
        W_rotated = W_normalized @ Pi.T  # (out, in)

        # Step 3: Quantize per-coordinate using Lloyd-Max codebook
        codebook = self._get_codebook(in_features, bits)
        indices = codebook.quantize(W_rotated)  # (out, in) integer

        # Step 4: Pack into CompressedLinear
        bias = linear.bias.data.clone() if linear.bias is not None else None

        return CompressedLinear(
            indices=indices,
            codebook=codebook,
            rotation_seed=rotation_seed,
            original_shape=(out_features, in_features),
            bias=bias,
            device=device,
            row_norms=row_norms,
        )

    def compress(
        self,
        model: nn.Module,
        custom_schedule: Optional[List[int]] = None,
    ) -> Dict[str, Union[float, int, List[int]]]:
        """Compress all linear layers in a model in-place.

        Replaces each nn.Linear with a CompressedLinear. Non-linear modules
        (embeddings, LayerNorm, etc.) are left untouched.

        Args:
            model: HuggingFace model (modified in-place).
            custom_schedule: Per-layer bit-widths for "custom" strategy.

        Returns:
            Dict with compression statistics:
                - num_compressed: Number of layers compressed.
                - schedule: Per-layer bit-widths.
                - effective_bpw: Average bits per weight.
                - original_params: Total original parameters.
                - theoretical_size_mb: Theoretical compressed size in MB.
        """
        num_layers = self._detect_num_layers(model)
        schedule = compute_weight_bit_schedule(
            num_layers=num_layers,
            target_bpw=self.target_bpw,
            strategy=self.strategy,
            custom_schedule=custom_schedule,
        )

        linear_layers = self._find_linear_layers(model)
        num_compressed = 0
        original_params = 0
        total_theoretical_bits = 0

        for name, parent, attr_name, linear in linear_layers:
            layer_idx = self._layer_index_for_name(name, num_layers)
            bits = schedule[min(layer_idx, len(schedule) - 1)]

            # Unique seed per layer for independent rotations
            rotation_seed = self.base_seed + layer_idx * 1000 + hash(name) % 1000
            device = linear.weight.device

            compressed = self.compress_linear(
                linear, bits=bits, rotation_seed=rotation_seed, device=device,
            )

            # Replace in parent module
            setattr(parent, attr_name, compressed)
            num_compressed += 1
            original_params += linear.weight.numel()
            mem = compressed.memory_bytes()
            total_theoretical_bits += mem["theoretical_bits"]

        avg_bpw = effective_bpw(schedule)
        return {
            "num_compressed": num_compressed,
            "num_layers": num_layers,
            "schedule": schedule,
            "effective_bpw": avg_bpw,
            "original_params": original_params,
            "theoretical_size_mb": total_theoretical_bits / 8 / 1024 / 1024,
        }


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

def compress_model(
    model: nn.Module,
    target_bpw: float = 2.5,
    strategy: str = "gradient",
    seed: int = 42,
    custom_schedule: Optional[List[int]] = None,
) -> Dict[str, Union[float, int, List[int]]]:
    """Compress model weights using TurboQuant.

    Replaces all nn.Linear layers with CompressedLinear layers that store
    quantized indices instead of float weights. Dequantization happens
    on-the-fly during each forward pass.

    Args:
        model: HuggingFace model (modified in-place).
        target_bpw: Target average bits per weight.
        strategy: "uniform" (same bits everywhere),
                  "gradient" (boundary-protective: 4-bit edges, 2-bit middle),
                  "custom" (provide custom_schedule).
        seed: Base seed for rotation matrices.
        custom_schedule: Per-layer bit-widths for "custom" strategy.

    Returns:
        Dict with compression statistics (see TurboQuantWeightCompressor.compress).
    """
    compressor = TurboQuantWeightCompressor(
        target_bpw=target_bpw,
        strategy=strategy,
        base_seed=seed,
    )
    return compressor.compress(model, custom_schedule=custom_schedule)


def estimate_compressed_size(
    num_params: int,
    num_layers: int,
    target_bpw: float = 2.5,
    strategy: str = "gradient",
) -> Dict[str, float]:
    """Estimate compressed model size without actually compressing.

    Useful for capacity planning (e.g., will a 70B model fit in 24GB?).

    Args:
        num_params: Total number of weight parameters.
        num_layers: Number of transformer layers.
        target_bpw: Target average bits per weight.
        strategy: Bit allocation strategy.

    Returns:
        Dict with:
            - fp16_gb: FP16 baseline size in GB.
            - compressed_gb: Estimated compressed size in GB.
            - compression_ratio: fp16 / compressed.
            - schedule: Per-layer bit-widths.
            - effective_bpw: Actual average bits per weight.
    """
    schedule = compute_weight_bit_schedule(
        num_layers=num_layers,
        target_bpw=target_bpw,
        strategy=strategy,
    )

    fp16_bits = num_params * 16
    fp16_gb = fp16_bits / 8 / 1024**3

    # Average bits from the schedule
    avg_bpw = effective_bpw(schedule)
    compressed_bits = num_params * avg_bpw
    # Add overhead: one fp16 norm per row per layer (negligible for large models)
    compressed_gb = compressed_bits / 8 / 1024**3

    return {
        "fp16_gb": fp16_gb,
        "compressed_gb": compressed_gb,
        "compression_ratio": fp16_gb / compressed_gb if compressed_gb > 0 else 0.0,
        "schedule": schedule,
        "effective_bpw": avg_bpw,
    }
