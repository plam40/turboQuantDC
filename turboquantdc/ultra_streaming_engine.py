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

from .ultra_streaming_analyzer import ModelAnalyzer
from .ultra_streaming_weights import WeightManager
from .ultra_streaming_kv import KVManager
from .ultra_streaming_planning import plan_memory, format_plan_report



class UltraStreamingEngine:
    """Run arbitrarily large models on limited GPU VRAM.

    Supports both dense models (full layer streaming) and MoE models
    (expert-level offloading with caching).

    The GPU holds:
    1. Embedding + LM head (always resident, ~1GB)
    2. Active layer(s) being computed (1-3 layers, ~1-3GB)
    3. TurboQuantDC compressed KV cache (rest of VRAM)

    CPU RAM holds:
    - All model weights (quantized)
    - KV cache overflow (if needed)

    Args:
        model_name: HuggingFace model name or path.
        gpu_budget_gb: VRAM budget in GB (default 22, leaving 2GB system headroom).
        cpu_budget_gb: CPU RAM budget in GB (default 64).
        kv_compression: Anchor strategy for KV cache.
        kv_bits: Bit-width for KV cache compression (2-8).
        weight_bits: Weight quantization bits (4 or 8).
        device: CUDA device string.
        dtype: Weight dtype.
    """

    def __init__(
        self,
        model_name: str,
        gpu_budget_gb: float = 22.0,
        cpu_budget_gb: float = 64.0,
        kv_compression: str = "boundary",
        kv_bits: int = 3,
        weight_bits: int = 4,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        if kv_bits < 2 or kv_bits > 8:
            raise ValueError(f"kv_bits must be between 2 and 8, got {kv_bits}")
        if gpu_budget_gb <= 0:
            raise ValueError(f"gpu_budget_gb must be positive, got {gpu_budget_gb}")
        if cpu_budget_gb <= 0:
            raise ValueError(f"cpu_budget_gb must be positive, got {cpu_budget_gb}")
        if weight_bits not in (4, 8, 16):
            raise ValueError(f"weight_bits must be 4, 8, or 16, got {weight_bits}")
        if kv_compression not in ANCHOR_STRATEGIES:
            raise ValueError(
                f"kv_compression must be one of {ANCHOR_STRATEGIES}, "
                f"got '{kv_compression}'"
            )

        self.model_name = model_name
        self.gpu_budget_gb = gpu_budget_gb
        self.cpu_budget_gb = cpu_budget_gb
        self.kv_compression = kv_compression
        self.kv_bits = kv_bits
        self.weight_bits = weight_bits
        self.device = torch.device(device)
        self.dtype = dtype

        # Components (populated by load())
        self.analyzer: Optional[ModelAnalyzer] = None
        self.weight_mgr: Optional[WeightManager] = None
        self.kv_mgr: Optional[KVManager] = None
        self.prefetcher: Optional[AsyncPrefetcher] = None
        self._plan: Optional[Dict[str, Any]] = None

        # Model components (populated by load())
        self.config = None
        self.tokenizer = None
        self.embed_tokens = None
        self.rotary_emb = None
        self.final_norm = None
        self.lm_head = None
        self.layers: List[Any] = []
        self.num_layers: int = 0

        # Metrics
        self._load_time: float = 0.0
        self._tokens_generated: int = 0
        self._generation_time: float = 0.0
        self._peak_vram: int = 0

    def analyze(self, config: Optional[Any] = None) -> Dict[str, Any]:
        """Analyze what's possible with this model + GPU combination.

        Can be called without loading the model -- uses config or
        KNOWN_ARCHITECTURES for offline analysis.

        Args:
            config: Optional HuggingFace AutoConfig. If None, tries to
                load from model_name or match against known architectures.

        Returns:
            Memory plan dict from plan_memory().
        """
        if config is not None:
            self.analyzer = ModelAnalyzer(
                config=config, weight_bits=self.weight_bits,
            )
        elif self.analyzer is None:
            # Try to match against known architectures
            arch = self._match_known_architecture(self.model_name)
            if arch is not None:
                self.analyzer = ModelAnalyzer(
                    arch_dict=arch, weight_bits=self.weight_bits,
                )
            else:
                # Must download config
                from transformers import AutoConfig
                hf_config = AutoConfig.from_pretrained(self.model_name)
                self.analyzer = ModelAnalyzer(
                    config=hf_config, weight_bits=self.weight_bits,
                )

        self._plan = plan_memory(
            self.analyzer,
            gpu_budget_gb=self.gpu_budget_gb,
            cpu_budget_gb=self.cpu_budget_gb,
        )
        return self._plan

    @staticmethod
    def _match_known_architecture(model_name: str) -> Optional[Dict[str, Any]]:
        """Try to match a model name against KNOWN_ARCHITECTURES.

        Matches on substrings in the model name, case-insensitive.

        Args:
            model_name: HuggingFace model name or path.

        Returns:
            Architecture dict if matched, None otherwise.
        """
        name_lower = model_name.lower()
        # Direct key match
        for key, arch in KNOWN_ARCHITECTURES.items():
            if key in name_lower:
                return arch
        # Common pattern matching
        patterns = {
            "405b": "llama-405b",
            "70b": "llama-70b",
            "13b": "llama-13b",
            "7b": "llama-7b",
            "72b": "qwen-72b",
            "scout": "llama4-scout",
            "maverick": "llama4-maverick",
            "deepseek-v3": "deepseek-v3",
            "deepseek_v3": "deepseek-v3",
            "mixtral-8x22": "mixtral-8x22b",
            "mixtral-8x7": "mixtral-8x7b",
        }
        for pattern, key in patterns.items():
            if pattern in name_lower:
                return KNOWN_ARCHITECTURES.get(key)
        return None

    def load(self) -> None:
        """Load model to CPU and set up GPU streaming infrastructure.

        1. Loads model config and tokenizer
        2. Loads weights to CPU (quantized via BitsAndBytes)
        3. Moves embeddings + LM head to GPU
        4. Sets up WeightManager for layer/expert caching
        5. Sets up KVManager for compressed KV cache
        6. Initializes async prefetcher
        """
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        start = time.time()

        # Load config
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.analyzer = ModelAnalyzer(
            config=self.config, weight_bits=self.weight_bits,
        )

        # Plan memory
        self._plan = plan_memory(
            self.analyzer,
            gpu_budget_gb=self.gpu_budget_gb,
            cpu_budget_gb=self.cpu_budget_gb,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model to CPU
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": self.dtype,
            "device_map": "cpu",
            "low_cpu_mem_usage": True,
        }

        # Add quantization config if needed
        if self.weight_bits in (4, 8):
            try:
                from transformers import BitsAndBytesConfig
                if self.weight_bits == 4:
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    load_kwargs["device_map"] = "auto"
                    load_kwargs["max_memory"] = {
                        0: f"{int(self.gpu_budget_gb)}GiB",
                        "cpu": f"{int(self.cpu_budget_gb)}GiB",
                    }
                else:
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
                    load_kwargs["device_map"] = "auto"
                    load_kwargs["max_memory"] = {
                        0: f"{int(self.gpu_budget_gb)}GiB",
                        "cpu": f"{int(self.cpu_budget_gb)}GiB",
                    }
            except ImportError:
                pass  # BnB not available, load at FP16

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **load_kwargs,
        )
        model.eval()

        # Extract backbone
        backbone = self._get_backbone(model)
        self.num_layers = self.analyzer.num_layers

        # Move permanent GPU residents
        self.embed_tokens = backbone.embed_tokens.to(self.device)
        if hasattr(backbone, "rotary_emb"):
            self.rotary_emb = backbone.rotary_emb.to(self.device)
        self.final_norm = backbone.norm.to(self.device)
        self.lm_head = model.lm_head.to(self.device)

        # Keep layers on CPU
        self.layers = list(backbone.layers)
        for layer in self.layers:
            layer.to("cpu")
            layer.eval()

        # Set up WeightManager
        weight_cache_bytes = int(self._plan["weight_cache_gb"] * (1024 ** 3))
        self.weight_mgr = WeightManager(
            gpu_budget_bytes=weight_cache_bytes,
            device=self.device,
            is_moe=self.analyzer.is_moe,
        )

        # Set up KVManager
        self.kv_mgr = KVManager(
            kv_budget_gb=self._plan["kv_budget_gb"],
            num_layers=self.num_layers,
            kv_bytes_per_token_fp16=self.analyzer.kv_bytes_per_token_fp16,
            kv_compression=self.kv_compression,
            kv_bits=self.kv_bits,
        )
        self.kv_mgr.create_cache()

        # Set up async prefetcher
        self.prefetcher = AsyncPrefetcher(device=self.device)

        # Clean up model shell
        del model, backbone
        gc.collect()

        self._load_time = time.time() - start

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            self._peak_vram = torch.cuda.max_memory_allocated(self.device)

    def _get_backbone(self, model: Any) -> Any:
        """Extract the transformer backbone from an HF model."""
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
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> str:
        """Generate text with ultra-streaming inference.

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
        self.kv_mgr.create_cache()
        cache = self.kv_mgr.cache

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        all_token_ids = input_ids.clone()

        gen_start = time.time()

        # Prefill
        next_token = self._generate_token(
            input_ids, past_seq_len=0, cache=cache,
            temperature=temperature, top_k=top_k,
        )
        all_token_ids = torch.cat([all_token_ids, next_token], dim=-1)
        past_seq_len = input_ids.shape[1]

        # Decode
        for step in range(max_new_tokens - 1):
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            next_token = self._generate_token(
                next_token, past_seq_len=past_seq_len, cache=cache,
                temperature=temperature, top_k=top_k,
            )
            all_token_ids = torch.cat([all_token_ids, next_token], dim=-1)
            past_seq_len += 1

        self._generation_time = time.time() - gen_start
        self._tokens_generated = all_token_ids.shape[1] - input_ids.shape[1]

        if torch.cuda.is_available():
            self._peak_vram = torch.cuda.max_memory_allocated(self.device)

        return self.tokenizer.decode(
            all_token_ids[0], skip_special_tokens=True,
        )

    @torch.inference_mode()
    def _generate_token(
        self,
        input_ids: torch.Tensor,
        past_seq_len: int,
        cache: Any,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        """Generate one token by streaming through all layers.

        Uses WeightManager for layer caching and async prefetch.
        """
        input_ids = input_ids.to(self.device)
        batch_size, seq_len = input_ids.shape

        cache_position = torch.arange(
            past_seq_len, past_seq_len + seq_len, device=self.device,
        )
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.embed_tokens(input_ids)

        if self.rotary_emb is not None:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
        else:
            position_embeddings = None

        attention_mask = self._build_causal_mask(
            batch_size, seq_len, past_seq_len, hidden_states.dtype,
        )

        # Stream through all transformer layers
        for layer_idx in range(self.num_layers):
            layer_key = (layer_idx,)

            # Check weight cache
            layer = self.weight_mgr.get(layer_key)
            if layer is None:
                # Not cached -- load from CPU
                layer_module = self.layers[layer_idx]
                layer_size = sum(
                    p.numel() * p.element_size()
                    for p in layer_module.parameters()
                )
                self.weight_mgr.load(layer_key, layer_module, layer_size)
                layer = layer_module

            # Start prefetching next layer
            next_idx = layer_idx + 1
            if next_idx < self.num_layers:
                next_key = (next_idx,)
                if not self.weight_mgr.is_cached(next_key):
                    self.prefetcher.prefetch(self.layers[next_idx])

            # Forward pass through layer
            output = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

        # Wait for any pending prefetch
        self.prefetcher.wait()

        # Final norm + LM head
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
        """Build a 4D causal attention mask."""
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

    def analysis_report(self) -> str:
        """Generate a human-readable analysis report.

        Returns:
            Multi-line report string.
        """
        if self.analyzer is None or self._plan is None:
            self.analyze()
        return format_plan_report(self.model_name, self.analyzer, self._plan)

    def memory_report(self) -> Dict[str, Any]:
        """Report current memory usage and performance metrics."""
        MB = 1024 ** 2

        tok_per_sec = 0.0
        if self._generation_time > 0 and self._tokens_generated > 0:
            tok_per_sec = self._tokens_generated / self._generation_time

        report = {
            "peak_vram_mb": round(self._peak_vram / MB, 1),
            "load_time_sec": round(self._load_time, 1),
            "tokens_generated": self._tokens_generated,
            "tokens_per_sec": round(tok_per_sec, 2),
            "gpu_budget_gb": self.gpu_budget_gb,
            "cpu_budget_gb": self.cpu_budget_gb,
            "kv_bits": self.kv_bits,
            "kv_compression": self.kv_compression,
            "weight_bits": self.weight_bits,
        }

        if self.weight_mgr is not None:
            report["weight_cache"] = self.weight_mgr.stats()

        if self.kv_mgr is not None and self.kv_mgr.strategy is not None:
            report["kv_strategy"] = self.kv_mgr.strategy["description"]

        if self._plan is not None:
            report["model_size_gb"] = self._plan["model_size_gb"]
            report["is_moe"] = self._plan["is_moe"]
            report["estimated_tok_per_sec"] = self._plan["estimated_tok_per_sec"]

        return report
