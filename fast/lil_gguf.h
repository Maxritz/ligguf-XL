/* LiGGUF - a small, dependency-free LLaMA inference engine with direct GGUF support
 * (C) Dmitry 'sciloaf' Solovyev aka MatrixS_Master, 2025-2026
 * */

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "platform.h"
#include <vector>
#include <string>
#include <map>

// Base constants
#define ALIGNMENT 32
#define MAXTENSORS 4000
#define MAXMETAKV 500
#define MAXNAMELEN 1024
#define QK8_0 32
#define QK8_1 32
#define QK_K 256
#define K_SCALE_SIZE 12
#define QK4_0 32
#define QK4_1 32
#define QK2_0 64
#define QK1_0 128
#define QK4_NL 32
#define QK_MXFP4 32

// ggml/gguf.h enum ggml_type — values are ABI, must match upstream exactly.
// Ids >= 128 are vendor-private (PrismML fork).
enum gguf_type {
    F32 = 0, F16 = 1,
    Q4_0 = 2, Q4_1 = 3,
    Q5_0 = 6, Q5_1 = 7,
    Q8_0 = 8, Q8_1 = 9,
    Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
    IQ2_XXS = 16, IQ2_XS = 17, IQ3_XXS = 18,
    IQ1_S = 19, IQ4_NL = 20, IQ3_S = 21, IQ2_S = 22, IQ4_XS = 23,
    I8 = 24, I16 = 25, I32 = 26, I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34, TQ2_0 = 35,
    MXFP4 = 39,
    NVFP4 = 40,
    Q1_0 = 41,   // upstream 1-bit ternary, QK1_0 = 128
    Q2_0 = 42,   // upstream 2-bit, QK2_0 = 64
    // vendor-private ids (PrismML fork)
    PQ2_0  = 142, // Q2_0 codec, group 128
    PTQ1_0 = 143, // TQ1_0 codec, group 128
    // LiGGUF-private legacy 1-bit formats (32-value blocks). Ids chosen so they
    // can never collide with a real GGUF file; old checkpoints keep loading.
    LQ1_0 = 200,
    LQ1_G = 201,
    GGUF_TYPE_COUNT
};

enum gguf_val_type {
    GUINT8, GINT8, GUINT16, GINT16, GUINT32, GINT32,
    GFLOAT32, GBOOL, GSTRING, GARRAY, GUINT64, GINT64, GFLOAT64,
    GGUF_VAL_TYPE_COUNT
};

typedef uint16_t gguf_half;

struct gguf_kv {
    uint64_t off;
    gguf_val_type tag;
};

struct gguf_tensor {
    uint64_t off;
    std::vector<uint64_t> dims;
    gguf_type type;
};

struct block_q8_0 {
    uint16_t d;
    int8_t qs[QK8_0];
};

struct block_q2_K {
    uint8_t scales[QK_K/16];
    uint8_t qs[QK_K/4];
    gguf_half d;
    gguf_half dmin;
};

struct block_q3_K {
    uint8_t hmask[QK_K/8];
    uint8_t qs[QK_K/4];
    uint8_t scales[12];
    gguf_half d;
};

struct block_q4_K {
    gguf_half d;
    gguf_half dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K/2];
};

struct block_q5_K {
    gguf_half d;
    gguf_half dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K/8];
    uint8_t qs[QK_K/2];
};

struct block_q6_K {
    uint8_t ql[QK_K/2];
    uint8_t qh[QK_K/4];
    int8_t scales[QK_K/16];
    gguf_half d;
};

/* Q5_0/Q5_1 blocks (from GGUF spec: QK8_0=32 per block) */
struct block_q5_0 {
    gguf_half d;
    uint8_t qh[QK8_0/8];   /* high bit for each of 32 values (bit-packed) */
    uint8_t qs[QK8_0/2];   /* lower 4 bits for each of 32 values (nibble-packed) */
};

struct block_q5_1 {
    gguf_half d;
    gguf_half m;           /* per-block min */
    uint8_t qh[QK8_0/8];
    uint8_t qs[QK8_0/2];
};

/* ---- legacy 4/8-bit blocks (ggml-common.h) ---- */

struct block_q4_0 {
    gguf_half d;
    uint8_t qs[QK4_0/2];   /* nibble-packed: low nibble = j, high nibble = j + qk/2 */
};

struct block_q4_1 {
    gguf_half d;
    gguf_half m;
    uint8_t qs[QK4_1/2];
};

struct block_q8_1 {
    gguf_half d;           /* delta */
    gguf_half s;           /* d * sum(qs) */
    int8_t qs[QK8_1];
};

/* Q2_0: 2 bits per weight, 64 per block, code 0..3 -> (code - 1) * d */
struct block_q2_0 {
    gguf_half d;
    uint8_t qs[QK2_0/4];
};

/* PQ2_0: identical codec to Q2_0 but 128 weights per block (PrismML fork) */
struct block_pq2_0 {
    gguf_half d;
    uint8_t qs[128/4];
};

/* MXFP4: E8M0 block scale + 32 E2M1 nibbles */
struct block_mxfp4 {
    uint8_t e;
    uint8_t qs[QK_MXFP4/2];
};

/* IQ4_NL: 16-entry non-linear 4-bit codebook, 32 values per block.
 * qs[j] low nibble -> element j, high nibble -> element j + 16. */
struct block_iq4_nl {
    gguf_half d;
    uint8_t qs[QK4_NL/2];
};

/* TQ1_0: ternary base-3 packing, 256 per block, 1.6875 bpw.
 * qs packs 5 trits per byte (base-3 digits), qh packs 4 trits per byte.
 * Decode: xi = ((byte * pow3[n]) * 3 >> 8) - 1, value = (xi - 1) * d. */
struct block_tq1_0 {
    uint8_t qs[(QK_K - 4*QK_K/64)/5];  /* 48 B -> 240 values */
    uint8_t qh[QK_K/64];               /*  4 B ->  16 values */
    gguf_half d;
};

/* PTQ1_0: same codec as TQ1_0 but 128 weights per block (PrismML fork). */
struct block_ptq1_0 {
    uint8_t qs[(128 - 4*128/64)/5];    /* 24 B -> 120 values */
    uint8_t qh[128/64];                /*  2 B ->   8 values */
    gguf_half d;
};

/* TQ2_0: 2 bits per weight, 256 per block, code 0..3 -> (code - 1) * d */
struct block_tq2_0 {
    uint8_t qs[QK_K/4];
    gguf_half d;
};

/* Q1_0: 1 bit per weight, 128 per block (upstream) */
struct block_q1_0 {
    gguf_half d;
    uint8_t qs[QK1_0/8];
};

/* LiGGUF-private legacy 1-bit formats, 32 values per block */
struct block_lq1_0 {
    gguf_half d;
    uint8_t qs[QK8_0/8];
};

struct block_lq1_G {
    gguf_half d;
    uint8_t qs[128/8];
};

struct block_q8_K {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K/16];
};

typedef std::vector<block_q8_0> qtensor;
typedef std::vector<block_q8_K> ktensor;
typedef std::vector<gguf_half> htensor;
typedef std::vector<float> ftensor;

struct wtensor {
    uint8_t* ptr;
    gguf_type type;
    uint32_t rsz;
    int rows = 0;   // number of rows (output dim) when known
    int cols = 0;   // row width (input dim) when known
};

/* Layer kind, detected from which tensors the GGUF actually contains. */
enum layer_kind {
    LAYER_UNKNOWN = 0,
    LAYER_ATTENTION,   // full attention: attn_q/k/v (+ attn_output)
    LAYER_LINEAR,      // gated delta net: attn_qkv + attn_gate + ssm_*
};

/* One decoder layer. Weight pointers alias into the mmap'd tensor block and are
 * never freed; NULL means "tensor absent in this checkpoint". */
struct trans_block {
    layer_kind kind;    // full attention vs linear/SSM
    bool is_full_attn = true; // mirror of `kind`, kept for the legacy loaders

    float* att_norm;        // pre-attention RMSNorm
    float* att_post_norm;   // post-attention RMSNorm (Qwen3.5 hybrid)
    wtensor att_q;          // full attention Q (may carry the Q gate)
    wtensor att_k;
    wtensor att_v;
    wtensor att_out;        // attention output projection
    float* att_q_norm;      // per-head Q RMSNorm
    float* att_k_norm;      // per-head K RMSNorm
    float* att_q_bias;      // Q projection bias (qwen2)
    float* att_k_bias;
    float* att_v_bias;

    /* linear attention (gated delta net) */
    wtensor att_qkv;        // fused QKV projection
    wtensor att_gate;       // z / output gate
    wtensor ssm_conv1d;     // causal depthwise conv kernel
    wtensor ssm_dt;         // dt bias
    wtensor ssm_a;          // -A_log.exp() per head
    wtensor ssm_beta;       // beta projection
    wtensor ssm_alpha;      // alpha projection
    float* ssm_norm;        // per-head output RMSNorm
    wtensor ssm_out;        // linear attention output projection

    wtensor ffn_norm;
    wtensor ffn_up;
    wtensor ffn_down;
    wtensor ffn_gate;

    /* MoE (llama/qwen*-moe style). ffn_gate_inp is the router logits. */
    bool is_moe = false;
    wtensor ffn_gate_inp;   // router: [n_embd, n_expert]
    wtensor ffn_up_exps;    // fused [n_expert, n_embd, n_ff]
    wtensor ffn_gate_exps;  // fused [n_expert, n_embd, n_ff]
    wtensor ffn_down_exps;  // fused [n_expert, n_ff, n_embd]
    wtensor ffn_up_shexp;   // shared expert (optional)
    wtensor ffn_gate_shexp;
    wtensor ffn_down_shexp;
    wtensor ffn_gate_inp_shexp; // shared-expert scalar gate (qwen2moe)
    int n_expert = 0;
    int n_expert_used = 0;
    int moe_ff = 0;         // expert intermediate width
    int shexp_ff = 0;       // shared-expert intermediate width
};

struct bert_block {
    wtensor attn_q, attn_k, attn_v, attn_out;
    float* attn_q_bias;
    float* attn_k_bias;
    float* attn_v_bias;
    float* attn_out_bias;
    float* attn_norm_w;
    float* attn_norm_b;
    wtensor ffn_up, ffn_down;
    float* ffn_up_bias;
    float* ffn_down_bias;
    float* ffn_norm_w;
    float* ffn_norm_b;
};

struct gguf_model {
    intptr_t file = -1; // model file handle
    uint8_t* base = NULL; // base mmap address
    uint64_t fsize = 0; // model file size
    uint8_t* tensors_off = NULL; // aligned offset of the start of tensors block
    std::string arch; // model architecture

    int vocab_size; // size of the vocabulary
    int n_layers; // number of layers
    int n_heads; // number of Query heads
    int n_kv_heads; // number of Key/Value heads
    int n_embed; // input embedding size
    int n_context; // size of the context window
    int tok_bos, tok_eos; // BOS/EOS token IDs
    int tok_unk, tok_sep, tok_pad, tok_cls, tok_mask; // extra tokenizer IDs
    bool add_bos = true; // tokenizer BOS policy
    int head_dim; // head size (computed)
    float rms_epsilon; // epsilon for RMS norm
    float rope_base; // RoPE base frequency
    int rope_dim; // RoPE dimension, clamped to head_dim during inference
    std::string rope_scaling;
    float rope_scale = 1.0f;
    int rope_orig_context = 0;
    int n_ff; // feed-forward hidden size
    int kv_dim; // total K/V width across KV heads
    int q_proj_dim; // Q projection output dimension (may be 2x for Qwen3.5 gate)
    bool has_qgate; // Qwen3.5: Q projection outputs [Q | gate]

    /* hybrid attention (Qwen3.5 / Qwen3.8 / qwen35 arch) */
    int full_attention_interval; // 0 = all layers are full attention
    std::vector<char> is_recr;    // per layer: 1 = linear/SSM, 0 = full attention

    /* SSM / gated delta net hyperparameters */
    int ssm_d_conv;      // conv kernel width
    int ssm_d_inner;     // value inner size
    int ssm_d_state;     // key/state head dim
    int ssm_dt_rank;     // number of value heads (== time-step rank)
    int ssm_n_group;     // number of key heads
    int rope_dim_sections[4]; // MRoPE sections

    /* MoE */
    int n_expert;
    int n_expert_used;
    bool moe_norm_w = true;   // renormalize selected expert weights (llama); false for qwen2moe
    bool moe_shexp = false;   // shared expert present

    /* PrismML Hadamard folding (prism.hadamard.*) */
    bool  hadamard_enabled = false;
    bool  hadamard_tied_output = false;
    int   hadamard_block_size = 0;
    std::vector<std::string> hadamard_weights;   // names needing the forward transform
    std::vector<std::string> hadamard_inverse;   // names needing the inverse-after-lookup
    std::vector<float> hadamard_signs;           // explicit sign vector (empty = identity)
    int   hadamard_sign_width = 0;               // width the sign vector applies to
    bool  hadamard_gdn_v_grouped = false;
    ftensor hadamard_rot;   // [block_size * block_size] rotation matrix, row-major

    std::vector<std::string> tokens; // vector of known tokens (pos == index)
    std::map<std::string,int> tokens_rev; // reverse token lookup by string
    ftensor tokscores; // tokenizer scores from GGUF
    std::vector<std::string> merges;
    std::map<std::string,int> merge_rank;
    std::string tok_model;
    std::string tok_pre;
    std::map<std::string,gguf_kv> meta_kv; // model metadata key/value pairs
    std::map<std::string,gguf_tensor> tensors; // all tensors described in the GGUF
    wtensor t_embed; // token embedding tensor (const)
    wtensor t_out; // output classifier tensor (const)
    float* t_outnorm; // final RMS norm weights (const)
    std::vector<trans_block> tr; // transformer blocks/layers (const)
    wtensor bert_tok_embd; // BERT token embedding tensor (const)
    wtensor bert_pos_embd; // BERT position embedding tensor (const)
    wtensor bert_tok_types; // BERT token-type embedding tensor (const)
    float* bert_emb_norm_w = NULL; // BERT embedding layernorm weights (const)
    float* bert_emb_norm_b = NULL; // BERT embedding layernorm bias (const)
    std::vector<bert_block> bert; // BERT encoder blocks/layers (const)

    uint32_t kvrd32(const std::string& key) const;
    float kvrdf32(const std::string& key) const;
    std::string kvrdstr(const std::string& key) const;
    bool kvrdbool(const std::string& key) const;
    uint32_t kv32(const std::string& key, uint32_t def) const;   // missing key -> def
    float    kvf32(const std::string& key, float def) const;

    bool open_mmap(const char* fn);
    void close_mmap();
    bool read_tokenizer();
    bool read_gguf();
    bool read_llama();
    bool read_qwen3();
    bool read_bert();
    bool read_arch();     // generic, dimension-driven loader (preferred)
    bool finalize();      // post-load fixups (tied output, validation)
    bool validate() const; // post-load consistency gate
    void build_hadamard();
    // returns NULL when the tensor is absent
    const gguf_tensor* find_tensor(const std::string& name) const;
};

uint64_t row_size(gguf_type type, int len);
