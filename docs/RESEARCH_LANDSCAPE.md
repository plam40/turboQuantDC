# KV Cache Compression Research Landscape (April 2026)

## Our Unique Position
Nobody has combined WHT-rotation + Lloyd-Max quantization with:
- Semantic clustering/merging
- Cross-token delta coding in WHT space
- Quantization codes as ANN index
- Role-specialized codebooks
- Tiered offload with retrieval

## Top 5 Unexplored Ideas (ranked)
1. Quantization-Indexed Retrieval (codes ARE the hash)
2. DeltaQuant in WHT space (cross-token residual coding)
3. Merge-Then-Quantize (cluster → centroid → quantize)
4. Tri-Tier Quantize-Compress-Offload
5. Role-Specialized Codebooks

## Competitive Threat
RotorQuant claims to beat TurboQuant: PPL 6.91 vs 7.07, 28% faster decode.
Uses block-diagonal rotations. Worth investigating.
