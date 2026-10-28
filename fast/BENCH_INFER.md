# CPU inference: prefill and token generation

Single-threaded decode (no thread pool yet), greedy sampling, `-G`.
Prefill and decode are timed separately by the engine.

Important caveat: **prefill is not batched**. The prompt is fed one token
per forward pass, exactly like decode, so prefill tok/s tracks decode tok/s
instead of running an order of magnitude faster. Batching the prompt into
a real GEMM is the single biggest prefill win available and is not done yet.

```
=== active scheme ===
Power Scheme GUID: 04387b98-d565-4b25-8de1-89f8902490c1  (LiGGUF Bench (clock pinned))power-mode overlay: ded574b5-45a0-4f42-8737-46345c09c238

1-thread :  3.39 GHz   [0,10=199%]
4-thread :  1.95 GHz   [0,0=115%  0,14=114%  0,1=114%  0,10=114%]
```

| model | prefill tok/s | decode tok/s | detail |
|---|---|---|---|
| smollm-135m-instruct-add-basics-q8_0.gguf | 44.6 | 31.4 | 490 tok prefill in 10988.5 ms, 127 tok decode in 4050.4 ms [c=2048 n=128] |
| tinyllama-15M-stories-Q2_K.gguf | 312.7 | 302 | 184 tok prefill in 588.5 ms, 47 tok decode in 155.6 ms [c=256 n=48 short] |
| TinyLLama-v0.Q8_0.gguf | 291.9 | 159.6 | 1065 tok prefill in 3648.2 ms, 127 tok decode in 795.6 ms [c=2048 n=128] |
| Tiny-LLM.Q4_K_M.gguf | 1431.1 | 1741 | 181 tok prefill in 126.5 ms, 47 tok decode in 27.0 ms [c=256 n=48 short] |
| Tiny-Moe.Q4_K_M.gguf | 45 | 25.9 | 1317 tok prefill in 29270.1 ms, 127 tok decode in 4905.9 ms [c=2048 n=128] |
| tinystories-gpt-0.1-3m.Q2_K.gguf | 435.5 | 297.5 | 490 tok prefill in 1125.2 ms, 127 tok decode in 426.8 ms [c=2048 n=128] |
| qwen2.5-0.5b-instruct-q2_k.gguf | 12.2 | 11.1 | 483 tok prefill in 39724.0 ms, 127 tok decode in 11487.7 ms [c=2048 n=128] |
| tinyllama-1.1b-chat-v1.0-q2_k.gguf | 15.4 | 9.9 | 1037 tok prefill in 67111.6 ms, 127 tok decode in 12829.2 ms [c=2048 n=128] |
| MiniCPM5-1B-Q8_0.gguf | 12.3 | 9.9 | 483 tok prefill in 39291.2 ms, 63 tok decode in 6366.7 ms [c=2048 n=64] |
| MiniCPM5-1B.i1-Q6_K.gguf | 13.1 | 13.4 | 483 tok prefill in 36955.3 ms, 63 tok decode in 4689.9 ms [c=2048 n=64] |
| Llama-3.2-1B-Instruct-Q8_0.gguf | 12.4 | 11.4 | 484 tok prefill in 39117.5 ms, 63 tok decode in 5512.0 ms [c=2048 n=64] |
| Qwen2.5-1.5B-MoE-A0.9B-Q4_K_M.gguf | 10.4 | 9.5 | 483 tok prefill in 46271.2 ms, 63 tok decode in 6650.5 ms [c=2048 n=64] |
| Llama-3.2-1B-4B-Quad-MoE.Q4_K_M.gguf | 9.6 | 9.7 | 484 tok prefill in 50218.9 ms, 63 tok decode in 6482.6 ms [c=2048 n=64] |
| Phi-3-mini-4k-instruct-q4.gguf | 4.1 | 3.1 | 1037 tok prefill in 254886.2 ms, 31 tok decode in 9934.4 ms [c=2048 n=32] |

Not swept here:
- Uraion-Agent-Steer-Q4_K_M.gguf (4.4 GB, supported but slow)
- Phi-4-reasoning-plus-Q3_K_M.gguf (6.9 GB, supported but slow)
- Ternary-Bonsai / Qwen3.5-9B / Qwen3.8-4B / MiniCPM-V (qwen35, GDN not implemented)
- Firefly-v5 / gemma-4-E4B (gemma4, no loader)
- Mellum2.1-12B (measured separately: 1.16 tok/s prefill, 3.07 tok/s decode)

