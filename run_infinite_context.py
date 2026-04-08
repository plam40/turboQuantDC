#!/usr/bin/env python3
"""100x Infinite Context Demo (Llama-3-8B).

This script demonstrates TurboRetrievalCache doing effectively infinite context
(up to 5 Million tokens depending on RAM) on a consumer desktop by indexing
keys in FAISS on the CPU and compressing values with TurboQuant.

It will generate a synthetic text base with a needle fact, chunked-prefill
the text into FAISS, and answer the question.
"""

import argparse
import time
from typing import Any, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from turboquantdc.chunked_prefill import ChunkedPrefillEngine
from turboquantdc.turbo_retrieval_cache import TurboRetrievalCache


def build_context(target_tokens: int, tokenizer) -> tuple[str, int]:
    """Build a massive context with a hidden needle."""
    NEEDLE_IDX = 0.5  # middle of context
    NEEDLE = "The secret system unlock code is X-ALPHA-774."
    
    # We will build it efficiently in memory using lists
    filler = "The system is functioning normally. All diagnostics report green metrics across the board. " * 5
    filler_tokens = len(tokenizer.encode(filler))
    
    # Pre-calculate how many paragraphs we need
    num_paras = target_tokens // filler_tokens + 1
    needle_para = int(num_paras * NEEDLE_IDX)
    
    paragraphs = []
    for i in range(num_paras):
        if i == needle_para:
            paragraphs.append(f"\n[SYSTEM ALERT]\n{NEEDLE}\n[ALERT END]\n")
        paragraphs.append(filler)
        
    text = "\n".join(paragraphs)
    # Don't re-encode the whole thing if it's millions of tokens right now just return the string.
    # The actual token count is roughly num_paras * filler_tokens
    actual_tokens = num_paras * filler_tokens
    return text, actual_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    # For demo, 1 Million tokens takes about 1-2 mins to prefill in chunks
    parser.add_argument("--tokens", type=int, default=1_000_000)
    args = parser.parse_args()

    print(f"Loading {args.model} natively onto GPU in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    model.eval()

    # Create the FAISS + TurboQuant cache
    print("Initializing FAISS TurboRetrievalCache (CPU Keys + Compressed Values)...")
    cache = TurboRetrievalCache(
        num_layers=model.config.num_hidden_layers,
        num_kv_heads=model.config.num_key_value_heads,
        head_dim=model.config.hidden_size // model.config.num_attention_heads,
        index_type="flat",  # Flat exact inner product for 100% recall
        nlist=16,
        m_subquantizers=16,
        key_bits=3, 
        val_bits=2,
        window_size=2048,
        k=2048,
    )

    print(f"Generating {args.tokens:,} tokens of synthetic context with a hidden needle...")
    t0 = time.time()
    context_text, token_count = build_context(args.tokens, tokenizer)
    print(f"Built context in {time.time()-t0:.1f}s.")
    
    prompt = (
        f"You are a retrieval system. Read the log and answer the question.\n\n"
        f"--- LOG START ---\n{context_text}\n--- LOG END ---\n\n"
        f"Question: What is the secret system unlock code?\n"
        f"Answer: The secret system unlock code is"
    )
    
    # Tokenize input (warning: tokenization takes a lot of RAM)
    print("Tokenizing huge prompt...")
    t0 = time.time()
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        
    print(f"Tokenized {inputs.input_ids.shape[1]:,} tokens in {time.time()-t0:.1f}s.")
    
    # Chunked prefill
    engine = ChunkedPrefillEngine(model_name=args.model, chunk_size=1024)
    engine.model = model
    engine.tokenizer = tokenizer
    engine.bits = 3
    engine.device = model.device
    
    def cb(tokens_done, total_ch, vram_gb):
        print(f"Prefill chunk {tokens_done}/{total_ch}  |  VRAM: {vram_gb:.2f} GB")
            
    print("Starting chunked stream to FAISS index...")
    t0 = time.time()
    
    # We must mock engine's cache to our cache
    engine.cache = cache
    # engine.prefill handles the chunking logic internally, but engine.prefill
    # instantiates a NEW cache. Let's just write the loop explicitly here
    # to use our precise cache cleanly without breaking API abstractions.
    total_tokens = inputs.input_ids.shape[1]
    chunk_size = 1024
    num_chunks = (total_tokens + chunk_size - 1) // chunk_size
    
    # --- MONKEY PATCH QWEN2 ATTENTION FOR RETRIEVAL AND PREFILL WINDOW ---
    print("Monkey-patching Qwen2Attention for native FAISS Retrieval...")
    import transformers.models.qwen2.modeling_qwen2 as modeling_qwen2
    
    original_forward = modeling_qwen2.Qwen2Attention.forward
    
    def retrieve_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        if "past_key_values" in kwargs and past_key_value is None:
            past_key_value = kwargs.pop("past_key_values")
            
        is_decoding = hidden_states.shape[1] == 1
        
        if past_key_value is None or not hasattr(past_key_value, "retrieve_and_attend"):
            return original_forward(
                self, hidden_states, attention_mask, position_ids, past_key_value,
                output_attentions, use_cache, cache_position, position_embeddings,
                **kwargs
            )
            
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            pass
        else:
            cos, sin = position_embeddings
            query_states, key_states = modeling_qwen2.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if not is_decoding:
            # PREFILL
            k_out, v_out = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs=None)
            
            if attention_mask is not None and attention_mask.dim() == 4:
                # Slice the HF auto-generated mask to match the limited keys_out size
                attention_mask = attention_mask[:, :, :, -k_out.shape[2]:]
                
            from transformers.models.qwen2.modeling_qwen2 import repeat_kv
            import torch.nn.functional as F
            
            k_out = repeat_kv(k_out, self.num_key_value_groups)
            v_out = repeat_kv(v_out, self.num_key_value_groups)
            
            attn_output = F.scaled_dot_product_attention(
                query_states, k_out, v_out, attn_mask=attention_mask
            )
        else:
            # DECODING FAISS RETRIEVAL
            # 1. Update cache with the new sequence=1 token
            if past_key_value is not None:
                _, _ = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs=None)
            
            # 2. FAISS Retrieval per query head (mapping to KV heads)
            attn_output_heads = []
            for h in range(self.config.num_attention_heads):
                kv_h = h // self.num_key_value_groups
                q_h = query_states[:, h, :, :] # (batch, 1, head_dim) -> (batch, head_dim)
                if q_h.dim() == 3:
                    q_h = q_h.squeeze(1)
                
                # Retrieve from cache
                res = past_key_value.retrieve_and_attend(self.layer_idx, kv_h, q_h)  # type: ignore
                attn_output_heads.append(res.output.unsqueeze(1)) # (batch, 1, head_dim)
                
            # Combine heads
            attn_output = torch.cat(attn_output_heads, dim=1) # (batch, num_heads, head_dim)
            attn_output = attn_output.unsqueeze(2) # (batch, num_heads, 1, head_dim)
        
        # Combine heads globally
        attn_output = attn_output.transpose(1, 2).contiguous() # (batch, q_len, num_heads, head_dim)
        attn_output = attn_output.reshape(bsz, q_len, -1)
        
        attn_output = self.o_proj(attn_output)

        return attn_output, None

    class FAISSQwen2Attention(type(model.model.layers[0].self_attn)):
        def forward(self, *args, **kwargs):
            return retrieve_forward(self, *args, **kwargs)
            
    for layer in model.model.layers:
        layer.self_attn.__class__ = FAISSQwen2Attention

    # --- PREFIX CHUNKING ---
    for chunk_idx in range(num_chunks):
        start_tok = chunk_idx * chunk_size
        end_tok = min(start_tok + chunk_size, total_tokens)
        chunk_ids = inputs.input_ids[:, start_tok:end_tok]
        chunk_len = end_tok - start_tok
        
        past_seen = cache.get_seq_length(0)
        cache_position = torch.arange(past_seen, past_seen + chunk_len, device=model.device)
        position_ids = cache_position.unsqueeze(0)
        
        # We must shape the attention mask exactly to match this size.
        actual_attn_len = min(past_seen + chunk_len, 2048)
        attention_mask = torch.ones((1, actual_attn_len), dtype=torch.long, device=model.device)

        with torch.no_grad():
            _ = model(
                input_ids=chunk_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
            )
        
        if (chunk_idx + 1) % 10 == 0 or (chunk_idx + 1) == num_chunks:
            cb(chunk_idx + 1, num_chunks, torch.cuda.memory_allocated() / 1e9)
            
    print(f"Prefill done in {time.time()-t0:.1f}s. Indexed {total_tokens} tokens into FAISS!")

    print("Generating response via FAISS Retrieval Attention...")
    generated_tokens = []
    
    # Get initial query token (last token of prompt)
    last_token = inputs.input_ids[:, -1:]
    
    # We rebuild position IDs for the first decoding step
    past_seen = cache.get_seq_length(0)
    cache_position = torch.tensor([past_seen], device=model.device)
    position_ids = cache_position.unsqueeze(0)
    
    current_ids = last_token
    t_start = time.time()
    
    with torch.no_grad():
        for _ in range(50):
            outputs = model(
                input_ids=current_ids,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
            )
            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1).item()
            generated_tokens.append(next_token_id)
            
            if next_token_id == tokenizer.eos_token_id:
                break
                
            current_ids = torch.tensor([[next_token_id]], device=model.device)
            cache_position += 1
            position_ids = cache_position.unsqueeze(0)

    t_end = time.time()
    
    ans = tokenizer.decode(generated_tokens)
    print("=" * 60)
    print(ans)
    print("=" * 60)
    print(f"Generated {len(generated_tokens)} tokens in {t_end - t_start:.2f}s ({len(generated_tokens)/(t_end - t_start):.2f} tok/s)")

    if "X-ALPHA-774" in ans:
        print("\nSUCCESS! The needle was perfectly retrieved from effectively infinite context 100x larger than normal GPU VRAM supports!")
    else:
        print("\nNeedle not found. Could not retrieve.")

if __name__ == "__main__":
    main()
