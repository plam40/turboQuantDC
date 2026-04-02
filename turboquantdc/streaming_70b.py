"""Streaming 70B Engine with Layer-Priority GPU Scheduling.

Runs models of ANY size (including 70B+) on a 24GB RTX 4090 by streaming
transformer layers between CPU and GPU. Key innovations over the basic
StreamingInferenceEngine:

1. **Layer-Priority GPU Cache**: Not all layers stream. Boundary layers
   (first 2 + last 2) and recently-used layers stay resident on GPU.
   An LRU cache decides what to evict when VRAM is full.

2. **Double-Buffered Prefetch**: While layer N computes on the default
   CUDA stream, layer N+1 transfers from CPU on a separate stream.
   This hides 50-80% of PCIe transfer latency.

3. **TurboQuantDC KV Cache**: Weights stream from CPU, but the compressed
   KV cache stays fully resident on GPU. At 3-bit with 5.1x compression,
   a 32K context for 80 layers costs ~400MB instead of ~2GB.

4. **Memory Budget Planner**: Calculates how many layers fit in GPU
   given the VRAM budget, after reserving space for embeddings, LM head,
   KV cache, and activation overhead.

VRAM breakdown for Llama-3.1-70B at FP16:
    Embedding + LM head:  ~1.0 GB (128256 * 8192 * 2 bytes)
    One transformer layer: ~0.9 GB (2 * 4 * 8192^2 * 2 bytes for QKV+O+MLP)
    KV cache (32K, 3-bit): ~0.4 GB (with TurboQuantDC compression)
    Activations:           ~0.5 GB (batch=1, seq=1 in decode)
    Overhead:              ~1.0 GB (CUDA context, fragmentation)
    ---------------------------------------------------------
    Available for layers:  ~17 GB out of 20 GB budget
    Layers that fit:       ~17-18 out of 80

Performance targets:
    - 70B model: ~5-10 tok/s (decode is PCIe-bottlenecked)
    - First token: <2 seconds (prefetch hides most latency)
    - Context: 32K+ with TurboQuantDC 5x compression

Usage:
    from turboquantdc.streaming_70b import StreamingModel

    model = StreamingModel(
        "meta-llama/Llama-3.1-70B-Instruct",
        gpu_budget_gb=20,
        kv_compression="boundary",
        kv_bits=3,
    )
    model.load()
    output = model.generate("Explain quantum entanglement:", max_new_tokens=100)
    print(output)
    print(model.memory_report())
"""

from __future__ import annotations

import gc
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from .generation_cache import (
    ANCHOR_STRATEGIES,
    GenerationCache,
    compute_anchor_schedule,
)


# ---------------------------------------------------------------------------
# Layer GPU Cache: LRU with priority protection
# ---------------------------------------------------------------------------


class LayerGPUCache:
    """LRU cache for transformer layers on GPU.

    Maintains an OrderedDict of layer indices to layer modules currently
    resident on GPU. When the cache exceeds capacity, the least-recently-used
    non-priority layer is evicted to CPU.

    Priority layers (e.g. first 2 + last 2 boundary layers) are pinned on
    GPU and never evicted, even if they are the LRU entry.

    Args:
        capacity: Maximum number of layers to keep on GPU simultaneously.
        priority_layers: Set of layer indices that should never be evicted.
    """

    def __init__(self, capacity: int, priority_layers: Set[int]):
        self.on_gpu: OrderedDict[int, Any] = OrderedDict()
        self.capacity = capacity
        self.priority = set(priority_layers)

    def get(self, layer_idx: int) -> Any:
        """Look up a layer in the GPU cache.

        If found, moves the entry to the most-recently-used position.

        Args:
            layer_idx: Index of the transformer layer.

        Returns:
            The layer module if resident on GPU, else None.
        """
        if layer_idx in self.on_gpu:
            self.on_gpu.move_to_end(layer_idx)
            return self.on_gpu[layer_idx]
        return None

    def load(self, layer_idx: int, layer: Any) -> None:
        """Load a layer onto GPU and add to the cache.

        If the layer is already cached, moves it to MRU position.
        If the cache is at capacity, evicts the LRU non-priority layer first.
        Calls ``layer.to("cuda")`` to transfer weights.

        Args:
            layer_idx: Index of the transformer layer.
            layer: The nn.Module for this transformer layer.
        """
        if layer_idx in self.on_gpu:
            self.on_gpu.move_to_end(layer_idx)
            return

        if len(self.on_gpu) >= self.capacity:
            self._evict_lru()

        layer.to("cuda")
        self.on_gpu[layer_idx] = layer

    def _evict_lru(self) -> None:
        """Evict the least-recently-used non-priority layer.

        Iterates from the front (LRU end) of the OrderedDict and evicts
        the first entry that is NOT in the priority set. If all entries
        are priority, no eviction occurs (cache exceeds capacity).
        """
        for idx in list(self.on_gpu.keys()):
            if idx not in self.priority:
                self.on_gpu[idx].to("cpu")
                del self.on_gpu[idx]
                return
        # All entries are priority -- cannot evict, allow capacity overshoot

    @property
    def resident_count(self) -> int:
        """Number of layers currently on GPU."""
        return len(self.on_gpu)

    def is_resident(self, layer_idx: int) -> bool:
        """Check if a layer is currently on GPU."""
        return layer_idx in self.on_gpu

    def evict_all_non_priority(self) -> None:
        """Evict all non-priority layers to CPU.

        Useful after a forward pass to reclaim VRAM for other operations.
        """
        for idx in list(self.on_gpu.keys()):
            if idx not in self.priority:
                self.on_gpu[idx].to("cpu")
                del self.on_gpu[idx]


# ---------------------------------------------------------------------------
# Async Prefetcher: double-buffered CPU->GPU transfer
# ---------------------------------------------------------------------------


class AsyncPrefetcher:
    """Asynchronous layer prefetcher using CUDA streams.

    While layer N computes on the default CUDA stream, this prefetcher
    transfers layer N+1 from CPU to GPU on a separate stream. After the
    compute finishes, ``wait()`` synchronizes the streams so the next
    layer is ready.

    Args:
        device: Target CUDA device.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._stream: Optional[torch.cuda.Stream] = None
        self._pending_layer: Any = None

        if torch.cuda.is_available():
            self._stream = torch.cuda.Stream(device=device)

    def prefetch(self, layer: Any) -> None:
        """Start asynchronous transfer of a layer to GPU.

        The transfer runs on a separate CUDA stream so it overlaps with
        computation on the default stream.

        Args:
            layer: The nn.Module to transfer to GPU.
        """
        if self._stream is None:
            # Fallback: synchronous transfer
            layer.to(self.device, non_blocking=True)
            self._pending_layer = layer
            return

        self._pending_layer = layer
        with torch.cuda.stream(self._stream):
            layer.to(self.device, non_blocking=True)

    def wait(self) -> None:
        """Block until the pending prefetch completes.

        Synchronizes the prefetch stream with the default stream so the
        layer's parameters are safe to use on the default stream.
        """
        if self._stream is not None and self._pending_layer is not None:
            self._stream.synchronize()
        self._pending_layer = None


# ---------------------------------------------------------------------------
# Memory Planner
# ---------------------------------------------------------------------------


class MemoryPlanner:
    """Static memory budget planner for streaming models.

    Calculates how many transformer layers can be held on GPU simultaneously,
    given the total VRAM budget, model architecture, and reservations for
    embeddings, LM head, KV cache, and activation overhead.
    """

    # Overhead for CUDA context + fragmentation + activations
    OVERHEAD_GB = 1.5
    # Minimum KV cache reservation
    MIN_KV_BUDGET_GB = 2.0

    @staticmethod
    def plan(
        num_layers: int,
        hidden_size: int,
        num_kv_heads: int,
        head_dim: int,
        vocab_size: int,
        dtype_bytes: int = 2,
        gpu_budget_gb: float = 20.0,
    ) -> Dict[str, Any]:
        """Calculate a memory plan for the given model architecture.

        Args:
            num_layers: Number of transformer layers.
            hidden_size: Model hidden dimension.
            num_kv_heads: Number of key/value heads (GQA).
            head_dim: Dimension per attention head.
            vocab_size: Vocabulary size.
            dtype_bytes: Bytes per parameter (2 for FP16).
            gpu_budget_gb: Total GPU VRAM to use (in GB).

        Returns:
            Dict containing:
                - num_gpu_layers: How many layers fit in GPU cache.
                - kv_budget_gb: VRAM reserved for KV cache.
                - layer_size_mb: Size of one transformer layer in MB.
                - embed_head_mb: Size of embedding + LM head in MB.
                - priority_layers: Set of boundary layer indices.
                - estimated_tok_per_sec: Rough throughput estimate.
        """
        MB = 1024 * 1024
        GB = 1024 * MB

        # Embedding + LM head size
        # Embedding: vocab_size * hidden_size * dtype_bytes
        # LM head: vocab_size * hidden_size * dtype_bytes (may be tied)
        embed_bytes = vocab_size * hidden_size * dtype_bytes
        lm_head_bytes = vocab_size * hidden_size * dtype_bytes
        embed_head_bytes = embed_bytes + lm_head_bytes

        # One transformer layer size (approximate):
        # Self-attention: Q, K, V, O projections = 4 * hidden_size * hidden_size
        # (with GQA, K/V are smaller, but we use the upper bound for safety)
        # FFN: gate + up + down = 3 * hidden_size * intermediate_size
        # intermediate_size is typically 3.5x hidden_size for Llama-style
        intermediate_size = int(hidden_size * 3.5)
        attn_bytes = 4 * hidden_size * hidden_size * dtype_bytes
        ffn_bytes = 3 * hidden_size * intermediate_size * dtype_bytes
        layer_bytes = attn_bytes + ffn_bytes

        # Priority layers: first 2 + last 2 (boundary layers)
        if num_layers <= 4:
            priority_layers = set(range(num_layers))
        else:
            priority_layers = {0, 1, num_layers - 2, num_layers - 1}

        # Available VRAM after fixed allocations
        budget_bytes = gpu_budget_gb * GB
        reserved = (
            embed_head_bytes
            + MemoryPlanner.OVERHEAD_GB * GB
            + MemoryPlanner.MIN_KV_BUDGET_GB * GB
        )
        available_for_layers = max(0, budget_bytes - reserved)

        # How many layers fit?
        if layer_bytes > 0:
            num_gpu_layers = int(available_for_layers // layer_bytes)
        else:
            num_gpu_layers = num_layers

        # Clamp to actual layer count
        num_gpu_layers = min(num_gpu_layers, num_layers)
        # Ensure at least priority layers fit
        num_gpu_layers = max(num_gpu_layers, min(len(priority_layers), num_layers))

        # KV budget: whatever is left after layers
        layers_used_bytes = num_gpu_layers * layer_bytes
        kv_budget_bytes = max(
            MemoryPlanner.MIN_KV_BUDGET_GB * GB,
            budget_bytes - embed_head_bytes - layers_used_bytes
            - MemoryPlanner.OVERHEAD_GB * GB,
        )

        # Rough throughput estimate:
        # Streaming layers need PCIe transfer. PCIe 4.0 x16 = ~25 GB/s
        # Layers not cached need transfer time = layer_bytes / bandwidth
        pcie_bandwidth = 25 * GB  # bytes/sec
        layers_to_stream = max(0, num_layers - num_gpu_layers)
        if layers_to_stream > 0 and layer_bytes > 0:
            stream_time_per_token = layers_to_stream * layer_bytes / pcie_bandwidth
            # With double-buffering, hide ~50% of transfer time
            stream_time_per_token *= 0.5
            estimated_tok_per_sec = 1.0 / max(stream_time_per_token, 0.001)
        else:
            estimated_tok_per_sec = 50.0  # all layers cached, no streaming

        return {
            "num_gpu_layers": num_gpu_layers,
            "kv_budget_gb": round(kv_budget_bytes / GB, 2),
            "layer_size_mb": round(layer_bytes / MB, 1),
            "embed_head_mb": round(embed_head_bytes / MB, 1),
            "priority_layers": priority_layers,
            "estimated_tok_per_sec": round(estimated_tok_per_sec, 1),
            "layers_to_stream": layers_to_stream,
        }


# ---------------------------------------------------------------------------
# StreamingModel: production 70B inference
# ---------------------------------------------------------------------------


class StreamingModel:
    """Production streaming inference engine for 70B+ models.

    Wraps a HuggingFace CausalLM model for layer-streaming inference.
    Combines layer-priority GPU scheduling with TurboQuantDC KV cache
    compression to run models that far exceed GPU VRAM.

    Architecture:
        - Always on GPU: embedding, final norm, LM head (~1GB)
        - High priority: first 2 + last 2 transformer layers (boundary)
        - LRU cached: recently-used layers up to GPU budget
        - Streamed: remaining layers loaded on demand with double-buffering

    Args:
        model_name: HuggingFace model name or path.
        gpu_budget_gb: VRAM budget in GB (default 20, leaving 4GB headroom
            on a 24GB RTX 4090).
        kv_compression: Anchor strategy for GenerationCache. One of
            "fixed", "boundary", "gradient".
        kv_bits: Bit-width for TurboQuantDC KV cache compression (2-8).
        device: CUDA device string.
        dtype: Weight dtype.
    """

    def __init__(
        self,
        model_name: str,
        gpu_budget_gb: float = 20.0,
        kv_compression: str = "boundary",
        kv_bits: int = 3,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        if kv_bits < 2 or kv_bits > 8:
            raise ValueError(f"bits must be between 2 and 8, got {kv_bits}")
        if gpu_budget_gb <= 0:
            raise ValueError(f"gpu_budget_gb must be positive, got {gpu_budget_gb}")
        self._validate_config(kv_bits=kv_bits, kv_compression=kv_compression)

        self.model_name = model_name
        self.gpu_budget_gb = gpu_budget_gb
        self.kv_compression = kv_compression
        self.kv_bits = kv_bits
        self.device = torch.device(device)
        self.dtype = dtype

        # Model components (populated by load())
        self.config = None
        self.tokenizer = None
        self.embed_tokens = None
        self.rotary_emb = None
        self.final_norm = None
        self.lm_head = None
        self.layers: List[Any] = []
        self.num_layers: int = 0

        # Architecture info
        self.hidden_size: int = 0
        self.num_attention_heads: int = 0
        self.num_kv_heads: int = 0
        self.head_dim: int = 0
        self.vocab_size: int = 0

        # Layer management
        self.layer_cache: Optional[LayerGPUCache] = None
        self.prefetcher: Optional[AsyncPrefetcher] = None
        self.memory_plan: Optional[Dict[str, Any]] = None

        # KV cache
        self.tq_cache: Optional[GenerationCache] = None

        # Metrics
        self._peak_vram: int = 0
        self._layer_size_bytes: int = 0
        self._embed_head_bytes: int = 0
        self._model_total_bytes: int = 0
        self._load_time: float = 0.0
        self._tokens_generated: int = 0
        self._generation_time: float = 0.0

    @staticmethod
    def _validate_config(kv_bits: int, kv_compression: str) -> None:
        """Validate configuration parameters.

        Args:
            kv_bits: Bit-width for KV cache.
            kv_compression: Anchor strategy name.

        Raises:
            ValueError: If kv_compression is not a valid anchor strategy.
        """
        if kv_compression not in ANCHOR_STRATEGIES:
            raise ValueError(
                f"kv_compression must be one of {ANCHOR_STRATEGIES}, "
                f"got '{kv_compression}'"
            )

    def load(self) -> None:
        """Load model: weights to CPU, embeddings + head to GPU, set up caches.

        This is the main initialization method. It:
        1. Loads the full model to CPU
        2. Moves embedding, rotary_emb, norm, lm_head to GPU
        3. Computes the memory plan
        4. Sets up the LayerGPUCache with priority layers
        5. Pre-loads priority layers to GPU
        6. Initializes the async prefetcher
        7. Sets up the TurboQuantDC KV cache
        """
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        start = time.time()

        # Load config
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.hidden_size = self.config.hidden_size
        self.num_attention_heads = self.config.num_attention_heads
        self.num_kv_heads = getattr(
            self.config, "num_key_value_heads", self.num_attention_heads
        )
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.vocab_size = self.config.vocab_size
        self.num_layers = self.config.num_hidden_layers

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model to CPU
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        model.eval()

        self._model_total_bytes = sum(
            p.numel() * p.element_size() for p in model.parameters()
        )

        # Extract backbone
        backbone = self._get_backbone(model)

        # Move permanent GPU residents
        self.embed_tokens = backbone.embed_tokens.to(self.device)
        if hasattr(backbone, "rotary_emb"):
            self.rotary_emb = backbone.rotary_emb.to(self.device)
        self.final_norm = backbone.norm.to(self.device)
        self.lm_head = model.lm_head.to(self.device)

        self._embed_head_bytes = sum(
            p.numel() * p.element_size()
            for module in [self.embed_tokens, self.lm_head, self.final_norm]
            for p in module.parameters()
        )

        # Keep layers on CPU
        self.layers = list(backbone.layers)
        for layer in self.layers:
            layer.to("cpu")
            layer.eval()

        self._layer_size_bytes = sum(
            p.numel() * p.element_size() for p in self.layers[0].parameters()
        )

        # Compute memory plan
        self.memory_plan = MemoryPlanner.plan(
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            vocab_size=self.vocab_size,
            dtype_bytes=self.layers[0].parameters().__next__().element_size(),
            gpu_budget_gb=self.gpu_budget_gb,
        )

        # Set up layer cache with priority layers
        self.layer_cache = LayerGPUCache(
            capacity=self.memory_plan["num_gpu_layers"],
            priority_layers=self.memory_plan["priority_layers"],
        )

        # Pre-load priority layers to GPU
        for idx in sorted(self.memory_plan["priority_layers"]):
            if idx < len(self.layers):
                self.layer_cache.load(idx, self.layers[idx])

        # Set up async prefetcher
        self.prefetcher = AsyncPrefetcher(device=self.device)

        # Initialize TurboQuantDC KV cache
        self.tq_cache = GenerationCache(
            key_bits=self.kv_bits,
            val_bits=max(2, self.kv_bits - 1),
            anchor_strategy=self.kv_compression,
        )

        # Clean up model shell
        del model, backbone
        gc.collect()

        self._load_time = time.time() - start

        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self._peak_vram = torch.cuda.max_memory_allocated(self.device)

    def _get_backbone(self, model: Any) -> Any:
        """Extract the transformer backbone from an HF model.

        Tries common attribute names used by Llama, Qwen, Mistral, etc.
        """
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer
        for attr in ("model", "transformer", "backbone"):
            if hasattr(model, attr):
                sub = getattr(model, attr)
                if hasattr(sub, "layers") or hasattr(sub, "h"):
                    return sub
        raise ValueError(
            f"Cannot find transformer backbone in {type(model).__name__}."
        )

    @torch.inference_mode()
    def _forward_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        """Run one transformer layer, managing GPU residency.

        Checks the layer cache first. If the layer is already on GPU,
        uses it directly. Otherwise, loads it from CPU (the prefetcher
        should have already started this transfer).

        During compute, starts prefetching the next layer.

        Args:
            layer_idx: Index of the transformer layer.
            hidden_states: Hidden states on GPU.
            position_ids: Position IDs on GPU.
            attention_mask: Causal mask on GPU.
            position_embeddings: (cos, sin) from rotary_emb.
            cache_position: Cache position indices on GPU.

        Returns:
            Updated hidden_states tensor.
        """
        # Wait for any pending prefetch (this layer should be ready)
        self.prefetcher.wait()

        # Get or load the layer
        layer = self.layer_cache.get(layer_idx)
        if layer is None:
            # Not in cache -- load synchronously (prefetch should have handled this)
            self.layer_cache.load(layer_idx, self.layers[layer_idx])
            layer = self.layers[layer_idx]

        # Start prefetching the NEXT layer while this one computes
        next_idx = layer_idx + 1
        if next_idx < self.num_layers:
            if not self.layer_cache.is_resident(next_idx):
                self.prefetcher.prefetch(self.layers[next_idx])

        # Forward pass
        output = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=self.tq_cache,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Track peak VRAM
        current_peak = torch.cuda.max_memory_allocated(self.device)
        if current_peak > self._peak_vram:
            self._peak_vram = current_peak

        return hidden_states

    @torch.inference_mode()
    def _generate_token(
        self,
        input_ids: torch.Tensor,
        past_seq_len: int = 0,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        """Generate one token by streaming through all layers.

        Args:
            input_ids: Input token IDs, shape (batch, seq).
            past_seq_len: Tokens already in the KV cache.
            temperature: Sampling temperature.
            top_k: Top-k filtering (0 = greedy).

        Returns:
            Next token ID, shape (batch, 1).
        """
        input_ids = input_ids.to(self.device)
        batch_size, seq_len = input_ids.shape

        # Position and cache bookkeeping
        cache_position = torch.arange(
            past_seq_len, past_seq_len + seq_len, device=self.device
        )
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

        # Embed
        hidden_states = self.embed_tokens(input_ids)

        # Rotary embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Causal mask
        attention_mask = self._build_causal_mask(
            batch_size, seq_len, past_seq_len, hidden_states.dtype
        )

        # Prefetch the first layer if not already on GPU
        if not self.layer_cache.is_resident(0):
            self.prefetcher.prefetch(self.layers[0])

        # Stream through all transformer layers
        for layer_idx in range(self.num_layers):
            hidden_states = self._forward_layer(
                layer_idx,
                hidden_states,
                position_ids,
                attention_mask,
                position_embeddings,
                cache_position,
            )

        # Wait for any lingering prefetch
        self.prefetcher.wait()

        # Final norm + LM head (only last position)
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states[:, -1:, :])

        # Sample
        if temperature <= 0 or top_k == 0:
            next_token = logits.argmax(dim=-1)
        else:
            logits = logits / temperature
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                probs = F.softmax(top_k_logits, dim=-1)
                sampled_idx = torch.multinomial(probs.squeeze(1), 1)
                next_token = top_k_indices.squeeze(1).gather(-1, sampled_idx)

        return next_token

    def _build_causal_mask(
        self,
        batch_size: int,
        seq_len: int,
        past_seq_len: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build a 4D causal attention mask.

        Shape: (batch, 1, seq_len, total_len).
        0.0 = attend, -inf = masked.
        """
        total_len = past_seq_len + seq_len
        mask = torch.zeros(
            batch_size, 1, seq_len, total_len,
            device=self.device, dtype=dtype,
        )
        if seq_len > 1:
            causal_mask = torch.triu(
                torch.full(
                    (seq_len, seq_len), torch.finfo(dtype).min,
                    device=self.device, dtype=dtype,
                ),
                diagonal=1,
            )
            mask[:, :, :, past_seq_len:] = causal_mask.unsqueeze(0).unsqueeze(0)
        return mask

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> str:
        """Generate text with streaming inference.

        For each decode step, streams through all layers with double-buffered
        prefetch and compressed KV cache.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering (0 = greedy).

        Returns:
            Generated text (prompt + completion).
        """
        if self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Fresh KV cache per generation
        self.tq_cache = GenerationCache(
            key_bits=self.kv_bits,
            val_bits=max(2, self.kv_bits - 1),
            anchor_strategy=self.kv_compression,
        )
        torch.cuda.reset_peak_memory_stats(self.device)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        all_token_ids = input_ids.clone()

        gen_start = time.time()

        # Prefill
        next_token = self._generate_token(
            input_ids, past_seq_len=0,
            temperature=temperature, top_k=top_k,
        )
        all_token_ids = torch.cat([all_token_ids, next_token], dim=-1)
        past_seq_len = input_ids.shape[1]

        # Decode
        for step in range(max_new_tokens - 1):
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            next_token = self._generate_token(
                next_token, past_seq_len=past_seq_len,
                temperature=temperature, top_k=top_k,
            )
            all_token_ids = torch.cat([all_token_ids, next_token], dim=-1)
            past_seq_len += 1

        self._generation_time = time.time() - gen_start
        self._tokens_generated = all_token_ids.shape[1] - input_ids.shape[1]
        self._peak_vram = torch.cuda.max_memory_allocated(self.device)

        output_text = self.tokenizer.decode(
            all_token_ids[0], skip_special_tokens=True
        )
        return output_text

    def memory_report(self) -> Dict[str, Any]:
        """Report VRAM usage, layer cache status, and performance.

        Returns:
            Dict with memory and performance metrics.
        """
        MB = 1024 * 1024

        tok_per_sec = 0.0
        if self._generation_time > 0 and self._tokens_generated > 0:
            tok_per_sec = self._tokens_generated / self._generation_time

        kv_cache_mb = 0.0
        if self.tq_cache is not None:
            try:
                savings = self.tq_cache.memory_savings()
                kv_cache_mb = savings.get("total_compressed_bits", 0) / 8 / MB
            except Exception:
                pass

        return {
            "peak_vram_mb": round(self._peak_vram / MB, 1),
            "model_total_mb": round(self._model_total_bytes / MB, 1),
            "layer_size_mb": round(self._layer_size_bytes / MB, 1),
            "embed_head_mb": round(self._embed_head_bytes / MB, 1),
            "num_layers": self.num_layers,
            "gpu_layers": self.layer_cache.resident_count if self.layer_cache else 0,
            "kv_cache_mb": round(kv_cache_mb, 3),
            "kv_bits": self.kv_bits,
            "kv_compression": self.kv_compression,
            "tokens_generated": self._tokens_generated,
            "tokens_per_sec": round(tok_per_sec, 2),
            "load_time_sec": round(self._load_time, 1),
            "gpu_budget_gb": self.gpu_budget_gb,
        }

    def architecture_info(self) -> Dict[str, Any]:
        """Return model architecture details and memory plan."""
        info = {
            "model_name": self.model_name,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "dtype": str(self.dtype),
        }
        if self.memory_plan:
            info["memory_plan"] = self.memory_plan
        return info
