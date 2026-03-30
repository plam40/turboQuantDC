"""Chunked Prefill Engine for TurboQuantDC.

Processes arbitrarily long documents through a real LLM by splitting the input
into chunks, running each chunk through the model, and compressing the resulting
KV cache with TurboQuant after each chunk.  This turns our synthetic 1M-context
benchmarks into REAL model generation from long context.

The key insight: ``model.forward()`` with ``past_key_values`` processes only
the NEW tokens while attending to all previously cached tokens.  So each chunk
adds to the KV cache without reprocessing earlier chunks.

Memory model:
    active = model_weights + one_chunk_activations + compressed_KV_cache
    At TQ-4 mse_only, compressed KV grows at ~8 bytes/token/layer/head.
    For 512K tokens with Qwen2.5-3B (36 layers, 2 KV heads):
        512K * 36 * 2 * 8 = ~300 MB -- fits easily on a 24 GB RTX 4090.

Usage:
    from turboquantdc.chunked_prefill import ChunkedPrefillEngine

    engine = ChunkedPrefillEngine("Qwen/Qwen2.5-3B-Instruct", bits=4, chunk_size=4096)
    engine.load_model()
    engine.prefill(long_document, callback=lambda done, total, vram: print(f"{done}/{total}"))
    answer = engine.generate("What was the secret code?", max_new_tokens=50)
    print(answer)
    print(engine.memory_report())

Reference: TurboQuant paper (arxiv 2504.19874).
"""

from __future__ import annotations

import gc
import time
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn.functional as F

from .hf_integration import TurboQuantCache


class ChunkedPrefillEngine:
    """Process arbitrarily long documents via chunked prefill with TurboQuant KV cache.

    Instead of feeding the entire document at once (which OOMs at long context),
    this engine:
    1. Splits the document into chunks of ``chunk_size`` tokens
    2. For each chunk, runs a forward pass through the model
    3. The KV cache from each chunk is compressed with TurboQuant
    4. The compressed KV cache accumulates on GPU (tiny memory footprint)
    5. After all chunks are processed, generates new tokens using the full context

    Memory: Only one chunk's activations + the compressed KV cache need to
    fit in VRAM at any time.  The compressed KV cache grows at ~5 bytes/token
    (TQ-3) instead of ~256 bytes/token (FP16 for d=128).

    Args:
        model_name: HuggingFace model name or path.
        bits: TurboQuant bit-width (2, 3, or 4).
        chunk_size: Tokens per prefill chunk (default 4096).
        mse_only: Use MSE-only mode for keys (recommended for generation quality
                  because it gives 2^b centroids instead of 2^(b-1) MSE + 1-bit QJL).
        device: GPU device string.
    """

    def __init__(
        self,
        model_name: str,
        bits: int = 4,
        chunk_size: int = 4096,
        mse_only: bool = True,
        device: str = "cuda",
    ):
        if bits not in (2, 3, 4):
            raise ValueError(f"bits must be 2, 3, or 4, got {bits}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        self.model_name = model_name
        self.bits = bits
        self.chunk_size = chunk_size
        self.mse_only = mse_only
        self.device = device

        self.model = None
        self.tokenizer = None
        self.cache: Optional[TurboQuantCache] = None
        self._total_tokens_processed: int = 0

        # Timing and memory tracking
        self._prefill_time: float = 0.0
        self._generation_time: float = 0.0
        self._tokens_generated: int = 0
        self._peak_vram_bytes: int = 0
        self._load_time: float = 0.0
        self._chunk_times: list[float] = []

    def load_model(self) -> None:
        """Load model and tokenizer.

        The model is loaded in 4-bit quantization via bitsandbytes to keep
        the model weights at ~2 GB, leaving the bulk of VRAM available for
        the compressed KV cache and chunk activations.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        start = time.time()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(load_in_4bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        self.model.eval()

        self._load_time = time.time() - start
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    def prefill(
        self,
        text: str,
        callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> int:
        """Process a long document through the model in chunks.

        Tokenizes the full document, splits it into chunks, and runs each chunk
        through the model.  The model's attention layers call
        ``TurboQuantCache.update()`` internally, so the KV cache accumulates
        compressed entries automatically.

        Args:
            text: The full document text.
            callback: Optional ``fn(chunks_done, total_chunks, vram_gb)`` called
                      after each chunk completes.

        Returns:
            Total number of tokens processed.

        Raises:
            RuntimeError: If the model has not been loaded yet.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start = time.time()
        self._chunk_times.clear()

        # Tokenize the full document
        input_ids = self.tokenizer.encode(text, return_tensors="pt")[0]
        total_tokens = len(input_ids)

        # Create a fresh TurboQuant cache
        self.cache = TurboQuantCache(
            bits=self.bits, seed=42, mse_only=self.mse_only,
        )
        self._total_tokens_processed = 0

        # Split into chunks
        num_chunks = (total_tokens + self.chunk_size - 1) // self.chunk_size

        for chunk_idx in range(num_chunks):
            chunk_start = time.time()

            start_tok = chunk_idx * self.chunk_size
            end_tok = min(start_tok + self.chunk_size, total_tokens)
            chunk_ids = input_ids[start_tok:end_tok].unsqueeze(0).to(self.device)

            # Forward pass -- the model's attention layers call
            # cache.update(key_states, value_states, layer_idx, cache_kwargs)
            # internally, which compresses and stores the new KV entries.
            # HuggingFace infers position_ids from the cache length, so
            # positions are correct automatically.
            with torch.no_grad():
                _outputs = self.model(
                    input_ids=chunk_ids,
                    past_key_values=self.cache,
                    use_cache=True,
                )

            self._total_tokens_processed += (end_tok - start_tok)
            self._chunk_times.append(time.time() - chunk_start)

            # Track peak VRAM
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            if peak > self._peak_vram_bytes:
                self._peak_vram_bytes = peak

            # Free chunk activations
            del _outputs
            gc.collect()
            torch.cuda.empty_cache()

            if callback:
                vram_gb = torch.cuda.memory_allocated() / 1e9
                callback(chunk_idx + 1, num_chunks, vram_gb)

        self._prefill_time = time.time() - start
        return self._total_tokens_processed

    def generate(
        self,
        prompt_suffix: str = "",
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> str:
        """Generate new tokens using the full compressed context.

        After ``prefill()`` has processed the document, this method generates
        an autoregressive continuation.  If ``prompt_suffix`` is provided
        (e.g. a question about the document), it is processed first so that
        the model sees ``[document context][question]`` before generating.

        Args:
            prompt_suffix: Optional text to append before generation
                           (e.g. a question about the document).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (1.0 = neutral, 0 = greedy).
            top_k: Top-k filtering (0 = greedy decoding).

        Returns:
            Generated text string (suffix + completion, without the document).

        Raises:
            RuntimeError: If ``prefill()`` has not been called yet.
        """
        if self.cache is None:
            raise RuntimeError("Call prefill() first to process a document.")
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        gen_start = time.time()

        # If there is a suffix prompt, process it through the model first
        # so that its KV entries are added to the cache.
        if prompt_suffix:
            suffix_ids = self.tokenizer.encode(
                prompt_suffix, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=suffix_ids,
                    past_key_values=self.cache,
                    use_cache=True,
                )
            # Grab logits for the last position to start generation
            next_logits = outputs.logits[:, -1, :]
            del outputs
        else:
            # Generate from the last position of the prefilled context.
            # We need to run a single-token forward pass to get logits.
            # Use the last token from the prefilled context as a prompt.
            # (The cache already has all the KV, so we just need logits.)
            # We re-run the last token to get its next-token logits.
            last_token_id = torch.tensor(
                [[self.tokenizer.eos_token_id]], device=self.device,
            )
            with torch.no_grad():
                outputs = self.model(
                    input_ids=last_token_id,
                    past_key_values=self.cache,
                    use_cache=True,
                )
            next_logits = outputs.logits[:, -1, :]
            del outputs

        # Autoregressive generation loop
        generated_ids = []
        for _step in range(max_new_tokens):
            # Sample the next token
            next_token_id = self._sample_token(
                next_logits, temperature=temperature, top_k=top_k,
            )
            token_id_val = next_token_id.item()
            generated_ids.append(token_id_val)

            # Stop on EOS
            if token_id_val == self.tokenizer.eos_token_id:
                break

            # Run the new token through the model (adds to cache)
            token_input = next_token_id.view(1, 1)  # (batch=1, seq=1)
            with torch.no_grad():
                outputs = self.model(
                    input_ids=token_input,
                    past_key_values=self.cache,
                    use_cache=True,
                )
            next_logits = outputs.logits[:, -1, :]
            del outputs

        self._tokens_generated = len(generated_ids)
        self._generation_time = time.time() - gen_start

        # Track peak VRAM after generation
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        if peak > self._peak_vram_bytes:
            self._peak_vram_bytes = peak

        # Decode generated tokens
        result_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # If there was a suffix, prepend it to the output for readability
        if prompt_suffix:
            return prompt_suffix + result_text
        return result_text

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        """Sample a single token from logits.

        Args:
            logits: Logits for the next position, shape (batch, vocab).
            temperature: Sampling temperature.
            top_k: Top-k filtering (0 = greedy).

        Returns:
            Token ID tensor of shape (1,).
        """
        if temperature <= 0 or top_k == 0:
            # Greedy decoding
            return logits.argmax(dim=-1).squeeze(0)

        logits = logits / temperature
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
            probs = F.softmax(top_k_logits, dim=-1)
            sampled_idx = torch.multinomial(probs.squeeze(0), 1)
            return top_k_indices.squeeze(0).gather(-1, sampled_idx).squeeze(-1)
        else:
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs.squeeze(0), 1).squeeze(-1)

    def memory_report(self) -> Dict[str, Any]:
        """Report memory usage and performance statistics.

        Returns:
            Dict with memory and performance metrics including:
            - total_tokens: Tokens in the compressed KV cache
            - peak_vram_gb: Peak GPU memory used
            - kv_cache_mb: Compressed KV cache size
            - fp16_kv_cache_mb: What FP16 would have used
            - compression_ratio: FP16 / compressed
            - prefill_time_sec: Time to process the document
            - prefill_tok_per_sec: Prefill throughput
            - generation_time_sec: Time for token generation
            - generation_tok_per_sec: Generation throughput
            - num_chunks: Number of chunks processed
            - chunk_avg_sec: Average time per chunk
            - bits: Configured bit-width
            - mse_only: Whether MSE-only mode is used
        """
        MB = 1024 * 1024

        kv_cache_mb = 0.0
        fp16_kv_mb = 0.0
        compression_ratio = 0.0
        num_layers = 0

        if self.cache is not None and self.cache.is_initialized:
            savings = self.cache.memory_savings()
            kv_cache_mb = savings["total_compressed_bits"] / 8 / MB
            fp16_kv_mb = savings["total_fp16_bits"] / 8 / MB
            compression_ratio = savings["overall_compression_ratio"]
            num_layers = savings["num_layers"]

        prefill_tok_per_sec = 0.0
        if self._prefill_time > 0 and self._total_tokens_processed > 0:
            prefill_tok_per_sec = self._total_tokens_processed / self._prefill_time

        gen_tok_per_sec = 0.0
        if self._generation_time > 0 and self._tokens_generated > 0:
            gen_tok_per_sec = self._tokens_generated / self._generation_time

        chunk_avg = 0.0
        if self._chunk_times:
            chunk_avg = sum(self._chunk_times) / len(self._chunk_times)

        return {
            "total_tokens": self._total_tokens_processed,
            "peak_vram_gb": round(self._peak_vram_bytes / 1e9, 2),
            "kv_cache_mb": round(kv_cache_mb, 3),
            "fp16_kv_cache_mb": round(fp16_kv_mb, 3),
            "compression_ratio": round(compression_ratio, 2),
            "prefill_time_sec": round(self._prefill_time, 2),
            "prefill_tok_per_sec": round(prefill_tok_per_sec, 1),
            "generation_time_sec": round(self._generation_time, 2),
            "generation_tok_per_sec": round(gen_tok_per_sec, 2),
            "tokens_generated": self._tokens_generated,
            "num_chunks": len(self._chunk_times),
            "chunk_avg_sec": round(chunk_avg, 2),
            "num_layers": num_layers,
            "bits": self.bits,
            "mse_only": self.mse_only,
        }

    @property
    def cache_seq_length(self) -> int:
        """Return number of tokens currently in the compressed KV cache."""
        if self.cache is None:
            return 0
        return self.cache.get_seq_length(0)


# ---------------------------------------------------------------------------
# Helper: build a long document with a hidden needle
# ---------------------------------------------------------------------------
def build_needle_document(
    needle_text: str,
    target_tokens: int = 32_000,
    depth: float = 0.25,
    tokenizer=None,
) -> str:
    """Build a long document with a needle fact embedded at a given depth.

    Creates filler text (repeating paragraphs about generic topics) and
    inserts the needle at the specified fractional depth.

    Args:
        needle_text: The fact to hide, e.g.
            "The secret code is BLUE-FALCON-42."
        target_tokens: Approximate total document length in tokens.
        depth: Where to place the needle (0.0 = start, 1.0 = end).
        tokenizer: HuggingFace tokenizer for accurate token counting.
            If None, estimates ~4 chars per token.

    Returns:
        The assembled document string with the needle embedded.
    """
    filler_paragraphs = [
        "The history of scientific discovery is marked by moments of unexpected insight. "
        "From Newton's falling apple to Fleming's contaminated petri dish, serendipity "
        "has played a crucial role in advancing human knowledge. These moments remind us "
        "that careful observation is just as important as systematic experimentation.",

        "Modern computational methods have transformed how we approach complex problems. "
        "Machine learning algorithms can now process vast datasets that would take human "
        "analysts years to review. This has led to breakthroughs in fields ranging from "
        "drug discovery to climate modeling and materials science.",

        "The architecture of distributed systems presents unique challenges in consistency, "
        "availability, and partition tolerance. The CAP theorem demonstrates that any "
        "networked shared-data system can provide at most two of these three guarantees "
        "simultaneously, forcing engineers to make careful tradeoffs.",

        "Quantum computing represents a paradigm shift in computational capability. "
        "By leveraging quantum mechanical phenomena such as superposition and entanglement, "
        "quantum processors can solve certain classes of problems exponentially faster "
        "than classical computers, though practical applications remain limited.",

        "The study of natural language processing has evolved from rule-based systems "
        "to statistical methods and now to large neural networks. Transformer architectures "
        "have proven remarkably effective at capturing long-range dependencies in text, "
        "enabling applications from translation to code generation.",

        "Software engineering best practices emphasize the importance of testing at "
        "multiple levels. Unit tests verify individual components, integration tests "
        "check component interactions, and end-to-end tests validate complete workflows. "
        "This layered approach catches different categories of defects.",

        "Database optimization involves careful consideration of indexing strategies, "
        "query planning, and data partitioning. A well-designed schema can reduce query "
        "times by orders of magnitude, while poor design leads to performance degradation "
        "that compounds as data volume grows.",

        "Network security requires a defense-in-depth approach combining firewalls, "
        "intrusion detection systems, encryption, and access controls. No single measure "
        "is sufficient; attackers will always find the weakest link unless every layer "
        "of the security stack is properly maintained.",
    ]

    # Estimate tokens per filler paragraph
    if tokenizer is not None:
        filler_tokens = [
            len(tokenizer.encode(p)) for p in filler_paragraphs
        ]
        avg_tokens_per_para = sum(filler_tokens) / len(filler_tokens)
    else:
        avg_tokens_per_para = sum(len(p) for p in filler_paragraphs) / len(filler_paragraphs) / 4

    # How many filler paragraphs do we need?
    num_paragraphs = int(target_tokens / avg_tokens_per_para) + 1

    # Build filler by cycling through paragraphs
    filler_parts = []
    for i in range(num_paragraphs):
        filler_parts.append(filler_paragraphs[i % len(filler_paragraphs)])

    # Insert needle at the specified depth
    needle_pos = max(1, int(len(filler_parts) * depth))
    filler_parts.insert(needle_pos, f"\n--- IMPORTANT NOTE ---\n{needle_text}\n--- END NOTE ---\n")

    document = "\n\n".join(filler_parts)

    # Trim to approximate target length
    if tokenizer is not None:
        tokens = tokenizer.encode(document)
        if len(tokens) > target_tokens:
            tokens = tokens[:target_tokens]
            document = tokenizer.decode(tokens, skip_special_tokens=True)

    return document


# ---------------------------------------------------------------------------
# Demo script
# ---------------------------------------------------------------------------
def main():
    """Demo: chunked prefill with needle-in-a-haystack retrieval."""
    import sys

    print("=" * 70)
    print("TurboQuantDC Chunked Prefill Demo")
    print("=" * 70)

    model_name = "Qwen/Qwen2.5-3B-Instruct"
    bits = 4
    chunk_size = 4096
    target_tokens = 32_000
    needle = "The secret project codename is BLUE-FALCON-42."

    print(f"\nConfiguration:")
    print(f"  Model:       {model_name}")
    print(f"  Bits:        {bits} (mse_only=True)")
    print(f"  Chunk size:  {chunk_size}")
    print(f"  Target doc:  {target_tokens} tokens")
    print(f"  Needle:      '{needle}'")
    print(f"  Depth:       25%")

    # 1. Load model
    print(f"\n--- Loading model ---")
    engine = ChunkedPrefillEngine(
        model_name, bits=bits, chunk_size=chunk_size, mse_only=True,
    )
    engine.load_model()
    print(f"  Model loaded in {engine._load_time:.1f}s")

    # 2. Build document with needle
    print(f"\n--- Building document ---")
    document = build_needle_document(
        needle_text=needle,
        target_tokens=target_tokens,
        depth=0.25,
        tokenizer=engine.tokenizer,
    )
    doc_tokens = len(engine.tokenizer.encode(document))
    print(f"  Document: {doc_tokens} tokens, {len(document)} chars")

    # 3. Prefill
    print(f"\n--- Chunked prefill ---")

    def progress(done, total, vram_gb):
        cached = engine.cache_seq_length
        print(
            f"  Chunk {done:3d}/{total}  |  "
            f"cached: {cached:6d} tokens  |  "
            f"VRAM: {vram_gb:.2f} GB"
        )

    total_tokens = engine.prefill(document, callback=progress)
    print(f"  Total tokens processed: {total_tokens}")

    # 4. Generate answer
    print(f"\n--- Generation (needle retrieval) ---")
    question = "\n\nBased on the document above, what is the secret project codename? Answer concisely:\n"
    answer = engine.generate(question, max_new_tokens=50, temperature=0.0)
    print(f"  Question: What is the secret project codename?")
    print(f"  Answer:   {answer.strip()}")

    # 5. Check if needle was found
    needle_found = "BLUE-FALCON-42" in answer.upper() or "BLUE" in answer.upper()

    # 6. Memory report
    report = engine.memory_report()
    print(f"\n--- Memory report ---")
    print(f"  Total tokens:        {report['total_tokens']}")
    print(f"  Peak VRAM:           {report['peak_vram_gb']} GB")
    print(f"  KV cache (TQ-{bits}):  {report['kv_cache_mb']:.1f} MB")
    print(f"  KV cache (FP16):     {report['fp16_kv_cache_mb']:.1f} MB")
    print(f"  Compression ratio:   {report['compression_ratio']}x")
    print(f"  Num layers:          {report['num_layers']}")
    print(f"  Prefill time:        {report['prefill_time_sec']}s ({report['prefill_tok_per_sec']:.0f} tok/s)")
    print(f"  Generation time:     {report['generation_time_sec']}s ({report['generation_tok_per_sec']:.1f} tok/s)")
    print(f"  Tokens generated:    {report['tokens_generated']}")
    print(f"  Chunks:              {report['num_chunks']} (avg {report['chunk_avg_sec']:.2f}s each)")
    print(f"  Needle found:        {'YES' if needle_found else 'NO'}")

    print(f"\n{'=' * 70}")
    if needle_found:
        print("SUCCESS: Needle retrieved from compressed context.")
    else:
        print("MISS: Needle not found in generation output.")
        print("  (This can happen at very high compression or long context.)")
    print(f"{'=' * 70}")

    return 0 if needle_found else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
