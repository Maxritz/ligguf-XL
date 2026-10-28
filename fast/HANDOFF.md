# LiGGUF FAST - Windows/MSVC correctness pass

Status after this pass: the engine now produces correct output for the LLaMA
family (llama / qwen2 / qwen2moe, dense and MoE) across the quant types
tested. The remaining gap is the Qwen3.5/Qwen3.8 `qwen35` hybrid (GDN linear
attention), plus gemma4/phi3/gpt2/mellum loaders.

## Build

    build_win.bat            # release (/O2)
    build_win.bat debug      # /Od /Zi

Ships `lil_quants.cpp` and `hadamard.cpp` in the compile list now.

## Critical bugs found and fixed

1. RoPE convention. The engine rotated adjacent pairs `(2m, 2m+1)`. LLaMA/ggml
   uses half-split rotary: the pair is `(m, m + rope_dim/2)` (GGML_ROPE_TYPE_NORM,
   `is_neox = false`). At position 0 RoPE is the identity, which is exactly why
   position 0 matched a reference and every later position diverged.

2. Data race in the attention loop. `kc_row` / `vc_row` were declared outside the
   OpenMP region and reassigned inside the per-head loop, so all threads stomped
   the same pointers. Multi-threaded output was nondeterministic garbage;
   single-threaded was correct. Fixed by making them per-iteration locals.

3. Q5_0 / Q5_1 codec. Both used an interleaved nibble layout. ggml is split-half:
   `qs[j]` low nibble is element `j`, high nibble is element `j + 16`, and bit
   `j` of the 32-bit `qh` is the 5th bit of element `j`. Q5_0 was also missing the
   `-16` bias (5-bit signed), and Q5_1 dropped the 5th bit and halved its
   `m * sum(q)` term. Fixed in both the dequantizer and the AVX2 dot.

4. `vocab_size` was resolved from the tokenizer array *after* `allocate()`, so
   models without a `*.vocab_size` KV sized the logits buffer to 0 and then wrote
   `n_vocab` rows into it. Now derived from `token_embd.weight` dims before
   allocation.

5. `head_dim` ignored `attention.key_length` / `value_length`. MiniCPM5-1B has
   head_dim 128 while `n_embed/n_heads` is 96.

6. Attention context width. `attn_output` input is `n_heads * head_dim`, which is
   not always `n_embed`. The context buffer was sized to `n_embed` and projected
   with `n = n_embed`.

7. Bias tensors were never loaded. The block-tensor loop only accepted keys
   containing `.weight`, so `attn_q.bias` / `attn_k.bias` / `attn_v.bias` were
   dropped and the bias path was dead. Every qwen2 model was wrong because of it.

8. GGUF version 2 was rejected. ggml parses v2 and v3 identically; only v1 is
   special. Now accepted.

9. `matmul` / `matmul2` / `matmul3` were three duplicated switches and the
   on-demand activation pointer was ignored, so a NULL `qx8` caused a NULL
   dereference. Replaced with one `row_dot` dispatcher plus `prepare_activation`,
   which resolves Q8_0 and/or Q8_K once per call and no longer aborts on mixed
   weight types.

## Features added

- IQ4_NL codec (dequant + fused dot).
- MoE FFN: softmax router over all experts, top-k, optional weight
  renormalization (llama) or not (qwen2moe), SiLU experts, optional shared expert
  with a sigmoid gate. Mirrors llama.cpp `build_moe_ffn`.
- Architecture-prefix driven hparam reading, so `qwen2`, `qwen2moe`, `phi3`,
  `gemma*` and `llama` all go through one loader.
- Attention biases, tied output embeddings, GGUF v2.

## Quality gates

- `-Z` numerical self-test. For every quant type present in the model it compares
  the fused activation x weight dot against a fully dequantized double-precision
  reference, and reports the fp16 round-trip error. All quant types currently
  pass at ~1e-7..1e-5 relative error.
- `-Y a,b,c` prints top-10 logits after feeding raw token ids.
- `-i "text"` prints the tokenizer output (ids + pieces).
- `FLOAT_DOT=1` forces the slow dequantize + float-FMA path; use it to separate
  codec bugs from graph bugs.
- `ref_forward.py` is an independent numpy oracle: it parses the GGUF directly
  and runs the graph, either printing top-10 logits per position or generating
  greedily. Use it to bisect any future numerical regression.

## Remaining work

- qwen35 hybrid: GDN linear attention (causal depthwise conv1d, L2 norm of q/k,
  delta rule with ssm_a/beta/alpha, gated RMSNorm, ssm_out) interleaved with full
  attention. Needed for Bonsai 2B/27B, Qwen3.5-9B, Qwen3.8-4B, MiniCPM-V.
- Ternary PTQ1_0 + the PrismML Hadamard fold (`hadamard.cpp` exists but is not
  wired into the matmul paths). Required for the Bonsai models.
- gemma4 / phi3 / gpt2 / mellum loaders.
- Q4_0 or f16 KV cache; a sane default context length (the default of
  `context_length` from the file allocated ~4.3 GB for a 131072-ctx model and
  thrashed); prefill (prompt-processing) speed.
