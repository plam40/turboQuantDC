from __future__ import annotations

import gc
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from .generation_cache import ANCHOR_STRATEGIES, GenerationCache
from .streaming_70b import AsyncPrefetcher
from .token_eviction import EvictionCache


KNOWN_ARCHITECTURES: Dict[str, Dict[str, Any]] = {
    # Dense models
    "llama-7b": {
        "type": "dense",
        "num_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_kv_heads": 32,
        "head_dim": 128,
        "intermediate_size": 11008,
        "vocab_size": 32000,
        "total_params_b": 7,
    },
    "llama-13b": {
        "type": "dense",
        "num_layers": 40,
        "hidden_size": 5120,
        "num_attention_heads": 40,
        "num_kv_heads": 40,
        "head_dim": 128,
        "intermediate_size": 13824,
        "vocab_size": 32000,
        "total_params_b": 13,
    },
    "llama-70b": {
        "type": "dense",
        "num_layers": 80,
        "hidden_size": 8192,
        "num_attention_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 28672,
        "vocab_size": 128256,
        "total_params_b": 70,
    },
    "llama-405b": {
        "type": "dense",
        "num_layers": 126,
        "hidden_size": 16384,
        "num_attention_heads": 128,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 53248,
        "vocab_size": 128256,
        "total_params_b": 405,
    },
    "qwen-72b": {
        "type": "dense",
        "num_layers": 80,
        "hidden_size": 8192,
        "num_attention_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 29568,
        "vocab_size": 152064,
        "total_params_b": 72,
    },
    # MoE models
    "llama4-scout": {
        "type": "moe",
        "num_layers": 48,
        "hidden_size": 5120,
        "num_attention_heads": 40,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 8192,
        "vocab_size": 202400,
        "total_params_b": 109,
        "num_experts": 16,
        "num_active_experts": 1,
        "active_params_b": 17,
    },
    "llama4-maverick": {
        "type": "moe",
        "num_layers": 48,
        "hidden_size": 5120,
        "num_attention_heads": 40,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 8192,
        "vocab_size": 202400,
        "total_params_b": 400,
        "num_experts": 128,
        "num_active_experts": 1,
        "active_params_b": 17,
    },
    "deepseek-v3": {
        "type": "moe",
        "num_layers": 61,
        "hidden_size": 7168,
        "num_attention_heads": 128,
        "num_kv_heads": 128,
        "head_dim": 128,
        "intermediate_size": 2048,  # per-expert intermediate (DeepSeek uses small experts)
        "vocab_size": 129280,
        "total_params_b": 685,
        "num_experts": 256,
        "num_active_experts": 8,
        "active_params_b": 37,
    },
    "mixtral-8x7b": {
        "type": "moe",
        "num_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 14336,
        "vocab_size": 32000,
        "total_params_b": 47,
        "num_experts": 8,
        "num_active_experts": 2,
        "active_params_b": 13,
    },
    "mixtral-8x22b": {
        "type": "moe",
        "num_layers": 56,
        "hidden_size": 6144,
        "num_attention_heads": 48,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 16384,
        "vocab_size": 32768,
        "total_params_b": 141,
        "num_experts": 8,
        "num_active_experts": 2,
        "active_params_b": 39,
    },
}




class ModelAnalyzer:
    """Analyze a model's architecture to determine streaming strategy.

    Works from either:
    1. A HuggingFace AutoConfig object
    2. A dict from KNOWN_ARCHITECTURES
    3. Manually specified parameters

    Determines:
    - Dense vs MoE architecture
    - Size per layer / per expert
    - Total params, active params
    - Minimum GPU VRAM needed
    - Optimal streaming strategy
    """

    # PCIe 4.0 x16 bandwidth in bytes/sec
    PCIE4_BANDWIDTH = 25 * (1024 ** 3)
    # PCIe 5.0 x16 bandwidth in bytes/sec
    PCIE5_BANDWIDTH = 50 * (1024 ** 3)

    def __init__(
        self,
        config: Optional[Any] = None,
        arch_dict: Optional[Dict[str, Any]] = None,
        weight_bits: int = 4,
    ):
        """Initialize from a HF config, a known architecture dict, or both.

        Args:
            config: HuggingFace AutoConfig object. If provided, arch_dict
                is ignored and values are extracted from config.
            arch_dict: Dict with architecture parameters (e.g. from
                KNOWN_ARCHITECTURES). Used when no HF config is available.
            weight_bits: Bits per weight after quantization (4 for BnB 4-bit,
                8 for BnB 8-bit, 16 for FP16).
        """
        if config is not None:
            self._from_config(config)
        elif arch_dict is not None:
            self._from_dict(arch_dict)
        else:
            raise ValueError("Either config or arch_dict must be provided")

        self.weight_bits = weight_bits

    def _from_config(self, config: Any) -> None:
        """Extract architecture from HuggingFace AutoConfig."""
        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = getattr(
            config, "num_key_value_heads", config.num_attention_heads
        )
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.num_attention_heads
        )
        self.intermediate_size = getattr(
            config, "intermediate_size",
            int(self.hidden_size * 3.5),
        )
        self.vocab_size = config.vocab_size

        # MoE detection — use hasattr to avoid MagicMock/dynamic attr issues
        self.num_experts = 0
        if hasattr(config, "num_local_experts") and isinstance(
            getattr(config, "num_local_experts", None), int
        ):
            self.num_experts = config.num_local_experts
        elif hasattr(config, "num_experts") and isinstance(
            getattr(config, "num_experts", None), int
        ):
            self.num_experts = config.num_experts

        self.num_active_experts = 0
        if hasattr(config, "num_experts_per_tok") and isinstance(
            getattr(config, "num_experts_per_tok", None), int
        ):
            self.num_active_experts = config.num_experts_per_tok
        elif hasattr(config, "num_selected_experts") and isinstance(
            getattr(config, "num_selected_experts", None), int
        ):
            self.num_active_experts = config.num_selected_experts

        self.is_moe = self.num_experts > 1

    def _from_dict(self, d: Dict[str, Any]) -> None:
        """Extract architecture from a known architecture dict."""
        self.num_layers = d["num_layers"]
        self.hidden_size = d["hidden_size"]
        self.num_attention_heads = d["num_attention_heads"]
        self.num_kv_heads = d.get("num_kv_heads", self.num_attention_heads)
        self.head_dim = d.get(
            "head_dim", self.hidden_size // self.num_attention_heads
        )
        self.intermediate_size = d.get(
            "intermediate_size", int(self.hidden_size * 3.5)
        )
        self.vocab_size = d["vocab_size"]

        self.num_experts = d.get("num_experts", 0)
        self.num_active_experts = d.get("num_active_experts", 0)
        self.is_moe = d.get("type", "dense") == "moe" or self.num_experts > 1

    # ------- Size calculations -------

    @property
    def bytes_per_param(self) -> float:
        """Bytes per parameter at the current weight quantization."""
        return self.weight_bits / 8.0

    @property
    def embedding_size_bytes(self) -> int:
        """Size of embedding + LM head in bytes (always FP16)."""
        # Embeddings and LM head are typically kept at FP16 even with
        # quantized weights, because quantizing them hurts quality badly.
        embed = self.vocab_size * self.hidden_size * 2  # FP16
        lm_head = self.vocab_size * self.hidden_size * 2
        return embed + lm_head

    @property
    def embedding_size_gb(self) -> float:
        """Size of embedding + LM head in GB."""
        return self.embedding_size_bytes / (1024 ** 3)

    @property
    def attention_size_bytes(self) -> int:
        """Size of one attention block (Q, K, V, O projections) in bytes."""
        # Q: hidden_size * num_attention_heads * head_dim
        # K, V: hidden_size * num_kv_heads * head_dim
        # O: num_attention_heads * head_dim * hidden_size
        q_size = self.hidden_size * self.num_attention_heads * self.head_dim
        kv_size = self.hidden_size * self.num_kv_heads * self.head_dim * 2
        o_size = self.num_attention_heads * self.head_dim * self.hidden_size
        total_params = q_size + kv_size + o_size
        return int(total_params * self.bytes_per_param)

    @property
    def ffn_size_bytes(self) -> int:
        """Size of one FFN block in bytes.

        For dense models: gate_proj + up_proj + down_proj.
        For MoE: this is per-expert FFN size.
        """
        # Llama-style: gate(h->i) + up(h->i) + down(i->h) = 3 * h * i
        total_params = 3 * self.hidden_size * self.intermediate_size
        return int(total_params * self.bytes_per_param)

    @property
    def expert_size_bytes(self) -> int:
        """Size of one expert (for MoE) or one FFN (for dense) in bytes."""
        return self.ffn_size_bytes

    @property
    def layer_size_bytes(self) -> int:
        """Size of one full transformer layer in bytes.

        For dense: attention + ffn.
        For MoE: attention + all experts + router.
        """
        attn = self.attention_size_bytes
        if self.is_moe and self.num_experts > 0:
            # All experts + router
            experts = self.num_experts * self.expert_size_bytes
            router = self.hidden_size * self.num_experts * 4  # router weights
            return attn + experts + router
        else:
            return attn + self.ffn_size_bytes

    @property
    def active_layer_size_bytes(self) -> int:
        """Size of one layer considering only active parameters.

        For dense: same as layer_size_bytes.
        For MoE: attention + active_experts only.
        """
        attn = self.attention_size_bytes
        if self.is_moe and self.num_active_experts > 0:
            active_ffn = self.num_active_experts * self.expert_size_bytes
            router = self.hidden_size * self.num_experts * 4
            return attn + active_ffn + router
        else:
            return self.layer_size_bytes

    @property
    def layer_size_gb(self) -> float:
        """Size of one full transformer layer in GB."""
        return self.layer_size_bytes / (1024 ** 3)

    @property
    def active_layer_size_gb(self) -> float:
        """Size of active parameters per layer in GB."""
        return self.active_layer_size_bytes / (1024 ** 3)

    @property
    def total_size_bytes(self) -> int:
        """Total model size in bytes."""
        return (
            self.embedding_size_bytes
            + self.num_layers * self.layer_size_bytes
        )

    @property
    def total_size_gb(self) -> float:
        """Total model size in GB."""
        return self.total_size_bytes / (1024 ** 3)

    @property
    def kv_bytes_per_token_fp16(self) -> int:
        """FP16 KV cache cost per token across all layers.

        Each token stores key and value vectors for every layer:
        2 (K+V) * num_kv_heads * head_dim * 2 (FP16 bytes) * num_layers
        """
        return 2 * self.num_kv_heads * self.head_dim * 2 * self.num_layers

    def summary(self) -> Dict[str, Any]:
        """Return a summary dict of the model architecture."""
        MB = 1024 ** 2
        GB = 1024 ** 3
        info = {
            "type": "moe" if self.is_moe else "dense",
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "weight_bits": self.weight_bits,
            "embedding_size_mb": round(self.embedding_size_bytes / MB, 1),
            "layer_size_mb": round(self.layer_size_bytes / MB, 1),
            "total_size_gb": round(self.total_size_gb, 1),
            "kv_bytes_per_token_fp16": self.kv_bytes_per_token_fp16,
        }
        if self.is_moe:
            info.update({
                "num_experts": self.num_experts,
                "num_active_experts": self.num_active_experts,
                "expert_size_mb": round(self.expert_size_bytes / MB, 1),
                "active_layer_size_mb": round(
                    self.active_layer_size_bytes / MB, 1
                ),
            })
        return info

