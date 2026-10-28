# qwen35 / GDN linear attention: the spec, extracted from the oracle

`qwen35` is the last architecture in `C:\models` with no loader. It is the carrier
for Ternary-Bonsai (PTQ1_0), Qwen3.5-9B, Qwen3.8-4B and MiniCPM-V, so it also
unblocks four quant codecs that currently have no testable model.

References, all local:

| what | where |
|---|---|
| model + graph | `experiment\llama.cpp-prism\src\models\qwen35.cpp` (63 KB) |
| the recurrence | `experiment\llama.cpp-prism\src\models\delta-net-base.cpp` (29 KB) |
| MoE variant | `experiment\llama.cpp-prism\src\models\qwen35moe.cpp` |
| PrismML fold + quant | `experiment\llama.cpp-prism\src\llama-quant.cpp`, `llama-model.cpp` |
| books/benchmarks | `experiment\Bonsai-demo\` (docs + whitepapers, no implementation) |

The prism fork is the authoritative one: it is a superset of upstream and it has
the `prism.hadamard.*` handling we also need.

## Shape of the model

Hybrid stack. Every `full_attention_interval`-th layer is ordinary causal
attention; the rest are gated delta net (linear attention with a recurrent state).

```c
is_recr[i] = (i < n_layer) && ((i + 1) % full_attention_interval != 0);
```

With `full_attention_interval = 4` that is 3 linear layers per 1 full layer, and
the last layer is full attention. Layers flagged `is_recr` use the recurrent
state instead of the KV cache.

### Hyperparameters

| key | meaning |
|---|---|
| `qwen35.block_count` | n_layer |
| `qwen35.embedding_length` | n_embd |
| `qwen35.attention.head_count` / `head_count_kv` | n_head / n_head_kv |
| `qwen35.attention.key_length` / `value_length` | head_dim (not n_embd/n_head) |
| `qwen35.full_attention_interval` | full-attention period |
| `qwen35.rope.dimension_count` | n_rot (partial rotary, < head_dim) |
| `qwen35.rope.dimension_sections` | MRoPE sections, 4 ints |
| `qwen35.ssm.conv_kernel` | ssm_d_conv |
| `qwen35.ssm.inner_size` | ssm_d_inner |
| `qwen35.ssm.state_size` | head_k_dim |
| `qwen35.ssm.group_count` | num_k_heads |
| `qwen35.ssm.time_step_rank` | num_v_heads |

Derived:

```c
head_v_dim = ssm_d_inner / num_v_heads;
qkv_dim    = 2 * head_k_dim * num_k_heads + head_v_dim * num_v_heads;
```

Worked example, `MiniCPM-V-4_6-Q8_0.gguf` (the first test target, 0.76 GB):

```
n_layer 24, n_embd 1024, n_head 8, n_head_kv 2, head_dim 256,
n_rot 64, sections [11,11,10,0], n_ff 3584,
conv_kernel 4, inner 2048, state 128, group 16, time_step_rank 16
=> head_v_dim = 2048/16 = 128
=> qkv_dim    = 2*128*16 + 128*16 = 6144   (matches attn_qkv [1024,6144] and
                                            ssm_conv1d [4,6144])
=> is_recr = layers 0,1,2,4,5,6,8,9,10,12,... ; full attention at 3,7,11,15,19,23
```

`MiniCPM-V` is the right first target: Q8_0 (a codec already proven at 1.4e-5),
small enough to iterate on, and it exercises both layer kinds.

## Tensors

Top level: `token_embd.weight`, `output_norm.weight`, `output.weight` (tied to
token_embd when absent).

Full-attention layer: `attn_norm`, `attn_q` (carries query **and** its gate),
`attn_k`, `attn_v`, `attn_q_norm`, `attn_k_norm`, `attn_output`.

Linear layer: `attn_norm`, `attn_qkv` (fused q,k,v), `attn_gate` (z),
`ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_dt` (bias), `ssm_a`, `ssm_norm`,
`ssm_out`, and `post_attention_norm` on both kinds.

All of these already exist as fields on `trans_block`; the loader binds nothing
yet and the graph skips linear layers entirely.

## Full-attention layer

Ordinary causal attention except for two things:

1. `attn_q` is `2 * head_dim * n_head` wide. The query is the **strided** view
   `[head_dim, n_head]` with stride `2*head_dim` and offset 0; the gate is the
   view at offset `head_dim` with the same stride. So `Q|gate` is interleaved per
   head, not concatenated. This is why Qwen3 sets `has_qgate`.
2. The gate is applied **after** attention: `out = attn_out * sigmoid(gate)`,
   then `attn_output` projection.

Then `q_norm`/`k_norm` are RMSNorm over `head_dim` before RoPE, and RoPE is MRoPE
over `n_rot` with the given `sections`. For text-only input all three position
streams are identical, so MRoPE degenerates to the plain half-split RoPE our
`rope()` already implements (it already clamps to `rope_dim` for partial rotary).

## Linear (GDN) layer

Given `cur` = RMSNorm'd input, for one token:

```
qkv   = attn_qkv   @ cur          -> qkv_dim
z     = attn_gate  @ cur          -> d_inner        (the output gate)
beta  = sigmoid(ssm_beta  @ cur)  -> num_v_heads
alpha = ssm_alpha  @ cur          -> num_v_heads
g     = ssm_a * softplus(alpha + ssm_dt)          -> num_v_heads
```

`ssm_a` already stores `-exp(A_log)`, so `g` is the (negative) log decay.

Depthwise causal conv over the whole `qkv` stream, kernel `ssm_d_conv`:

```
qkv_c = silu( conv1d( qkv, ssm_conv1d, conv_state ) )
```

`conv_state` is a rolling buffer of the previous `ssm_d_conv - 1` rows of `qkv`.

Split, then a **joint** L2 normalisation of q and k (one norm over the two head
groups viewed as `[head_k_dim, 2*num_k_heads]`):

```
qk = l2norm( concat(q_conv, k_conv), eps )
q  = qk[:, :num_k_heads]
k  = qk[:, num_k_heads:]
v  = v_conv
if (num_k_heads != num_v_heads)  repeat q and k out to num_v_heads
```

Then the GDN recurrence, autoregressive form for one token
(`delta-net-base.cpp:289`, `build_delta_net_autoregressive`). With state
`s[head_v_dim][head_v_dim]` per value head, `q` scaled by `1/sqrt(head_k_dim)`:

```
g  = exp(g)                       // per head scalar (GDA) or per key dim
s  = s * g                        // decay the state
sk = rowsum(s * k)                // [head_v_dim]
d  = (v - sk) * beta              // [head_v_dim]
s  = s + outer(k, d)
o  = rowsum(s * q)                // [head_v_dim]
```

Then a gated RMSNorm and the output projection:

```
o = rmsnorm(o, ssm_norm) * silu(z)      // ssm_norm is [head_v_dim]
out = ssm_out @ o                       -> n_embd
```

Residual: `x += out`, and `attn_post_norm` is a post-norm on the attention
output before the FFN, matching the existing `attn_post_norm` field.

## State

Two buffers per layer, both persistent across tokens:

- `conv_state`: `[ssm_d_conv - 1][qkv_dim]` rolling window, float.
- `ssm_state`: `[num_v_heads][head_v_dim][head_v_dim]` float. For MiniCPM-V that
  is `16 * 128 * 128 * 4` = 1 MB per layer, 18 linear layers = ~18 MB. Fine.

The KV cache is used only by the full-attention layers, so `allocate()` should
size `kc`/`vc` by the count of full-attention layers, not `n_layers`.

## PTQ1_0 and the PrismML Hadamard fold (Ternary-Bonsai)

Ternary-Bonsai adds two things on top, both already stubbed in the engine
(`hadamard_*` fields, `build_hadamard()`, `hadamard.cpp`):

- **PTQ1_0** (id 143) is the TQ1_0 ternary codec at group size 128. Dequant and
  dot are already implemented and pass the self test on the 2B model's tensors.
- **The fold**: `PTQ1_0` and `PQ2_0` weights are only valid in a
  Hadamard-rotated basis, so activations must be rotated by the same block
  diagonal before the matmul. `syntax`: `prism.hadamard.*` metadata names which
  weights are folded (`hadamard_weight_blocks`), the block size, and a sign
  vector. There is one special case: `hadamard_gdn_v_grouped` in
  `llama-model.cpp:608` applies to `ssm_out.weight`, because the GDN value
  grouping changes the rotation layout.

Order matters: rotate the activation, then the matmul, and use the
inverse-rotated output where the oracle does.

## Plan

1. Loader: read the hparams above, bind the tensors, set `is_recr` per layer,
   build the derived dims, and size the recurrent state. Verify by loading
   MiniCPM-V.
2. Honest refusal: until step 3 lands, `qwen35` must keep erroring rather than
   running a half graph. The current access violation is worse than an error.
3. Graph: the full-attention layer (Q-gate view, qk-norm, RoPE, sigmoid gate).
4. Graph: the linear layer (conv + GDN + gated norm + out).
5. Verify against MiniCPM-V: `-Z` self test, then coherent text, then compare a
   logits probe against llama.cpp-prism on the same prompt.
6. Then the Hadamard fold and PTQ1_0, verified on Ternary-Bonsai-2-2B.

Steps 3 and 4 are the bulk: roughly 300-400 lines across the loader, the graph
and the state buffers. They are mechanical now that the spec is pinned down, but
they need the verification loop in step 5, which is not something to rush in the
same pass as the writing.
