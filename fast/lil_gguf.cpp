/* LiGGUF - a small, dependency-free LLaMA inference engine with direct GGUF support
 * (C) Dmitry 'sciloaf' Solovyev aka MatrixS_Master, 2025-2026
 * */

#include <assert.h>
#include <time.h>
#include "platform.h"
#include "common.h"
#include "lil_gguf.h"

// Standard GGUF keys
#define TOKENS_KEY "tokenizer.ggml.tokens"
#define TOKENS_SCORE_KEY "tokenizer.ggml.scores"
#define TOKENS_TYPE_KEY "tokenizer.ggml.token_type"
#define TOKENS_MERGES_KEY "tokenizer.ggml.merges"
#define TOKENS_MODEL_KEY "tokenizer.ggml.model"
#define TOKENS_PRE_KEY "tokenizer.ggml.pre"
#define TOKENS_EMBED_KEY "token_embd.weight"
#define TOKENS_BOS_KEY "tokenizer.ggml.bos_token_id"
#define TOKENS_EOS_KEY "tokenizer.ggml.eos_token_id"
#define TOKENS_UNK_KEY "tokenizer.ggml.unknown_token_id"
#define TOKENS_SEP_KEY "tokenizer.ggml.seperator_token_id"
#define TOKENS_PAD_KEY "tokenizer.ggml.padding_token_id"
#define TOKENS_CLS_KEY "tokenizer.ggml.cls_token_id"
#define TOKENS_MASK_KEY "tokenizer.ggml.mask_token_id"
#define TOKENS_ADD_BOS_KEY "tokenizer.ggml.add_bos_token"
#define ARCH_KEY "general.architecture"
#define OUTPUT_KEY "output.weight"
#define OUTPUT_NORM_KEY "output_norm.weight"

// LLaMA-specific keys
#define VOCAB_SIZE_KEY "llama.vocab_size"
#define CONTEXT_LEN_KEY "llama.context_length"
#define EMBED_LEN_KEY "llama.embedding_length"
#define HEAD_COUNT_KEY "llama.attention.head_count"
#define HEAD_KV_COUNT_KEY "llama.attention.head_count_kv"
#define BLOCK_COUNT_KEY "llama.block_count"
#define RMS_EPSILON_KEY "llama.attention.layer_norm_rms_epsilon"
#define ROPE_BASE_KEY "llama.rope.freq_base"
#define ROPE_DIMS_KEY "llama.rope.dimension_count"
#define FF_LEN_KEY "llama.feed_forward_length"
#define KEY_LEN_KEY "llama.attention.key_length"
#define VALUE_LEN_KEY "llama.attention.value_length"

// Qwen3-specific keys
#define QWEN3_BLOCK_COUNT_KEY "qwen3.block_count"
#define QWEN3_CONTEXT_LEN_KEY "qwen3.context_length"
#define QWEN3_EMBED_LEN_KEY "qwen3.embedding_length"
#define QWEN3_HEAD_COUNT_KEY "qwen3.attention.head_count"
#define QWEN3_HEAD_KV_COUNT_KEY "qwen3.attention.head_count_kv"
#define QWEN3_RMS_EPSILON_KEY "qwen3.attention.layer_norm_rms_epsilon"
#define QWEN3_ROPE_BASE_KEY "qwen3.rope.freq_base"
#define QWEN3_ROPE_SCALING_TYPE_KEY "qwen3.rope.scaling.type"
#define QWEN3_ROPE_SCALING_FACTOR_KEY "qwen3.rope.scaling.factor"
#define QWEN3_ROPE_SCALING_ORIG_CTX_KEY "qwen3.rope.scaling.original_context_length"
#define QWEN3_KEY_LEN_KEY "qwen3.attention.key_length"
#define QWEN3_VALUE_LEN_KEY "qwen3.attention.value_length"
#define QWEN3_FF_LEN_KEY "qwen3.feed_forward_length"

//BERT-specific keys
#define BERT_BLOCK_COUNT_KEY "bert.block_count"
#define BERT_CONTEXT_LEN_KEY "bert.context_length"
#define BERT_EMBED_LEN_KEY "bert.embedding_length"
#define BERT_HEAD_COUNT_KEY "bert.attention.head_count"
#define BERT_LN_EPSILON_KEY "bert.attention.layer_norm_epsilon"
#define BERT_FF_LEN_KEY "bert.feed_forward_length"
#define BERT_POS_EMBED_KEY "position_embd.weight"
#define BERT_TYPE_EMBED_KEY "token_types.weight"
#define BERT_EMB_NORM_W_KEY "token_embd_norm.weight"
#define BERT_EMB_NORM_B_KEY "token_embd_norm.bias"

using namespace std;

#define RDMEM(T,N) T inline N (uint8_t** pos)\
{\
    T r = 0;\
    memcpy(&r,*pos,sizeof(T));\
    *pos += sizeof(T);\
    return r;\
}

RDMEM(uint8_t,rd8)
RDMEM(uint16_t,rd16)
RDMEM(uint32_t,rd32)
RDMEM(uint64_t,rd64)
RDMEM(float,rdf32)

static string rdstr(uint8_t** pos)
{
    uint64_t l = rd64(pos);
    string res((const char*)*pos,l);
    *pos += l;
    return res;
}

uint32_t gguf_model::kvrd32(const string& key) const
{
    if (!meta_kv.count(key)) {
        ERR("Key '%s' not found!",key.c_str());
        return 0;
    }
    uint8_t* p = base + meta_kv.at(key).off;
    return rd32(&p);
}

float gguf_model::kvrdf32(const string& key) const
{
    if (!meta_kv.count(key)) {
        ERR("Key '%s' not found!",key.c_str());
        return 0;
    }
    uint8_t* p = base + meta_kv.at(key).off;
    return rdf32(&p);
}

string gguf_model::kvrdstr(const string& key) const
{
    if (!meta_kv.count(key)) {
        ERR("Key '%s' not found!",key.c_str());
        return "";
    }
    uint8_t* p = base + meta_kv.at(key).off;
    return rdstr(&p);
}

bool gguf_model::kvrdbool(const string& key) const{
    if (!meta_kv.count(key)) {
        ERR("Key '%s' not found!",key.c_str());
        return false;
    }
    uint8_t* p = base + meta_kv.at(key).off;
    return rd8(&p) != 0;
}

uint32_t gguf_model::kv32(const string& key, uint32_t def) const
{
    return meta_kv.count(key) ? kvrd32(key) : def;
}

float gguf_model::kvf32(const string& key, float def) const
{
    return meta_kv.count(key) ? kvrdf32(key) : def;
}

static uint64_t skipper(gguf_val_type t, uint8_t* pos)
{
    switch (t) {
        case GUINT8:
        case GINT8:
        case GBOOL:
            return 1;
        case GUINT16:
        case GINT16:
            return 2;
        case GUINT32:
        case GINT32:
        case GFLOAT32:
            return 4;
        case GUINT64:
        case GINT64:
        case GFLOAT64:
            return 8;
        case GSTRING: {
            uint64_t l = rd64(&pos);
            return l + 8;
        }
        case GARRAY: {
            uint8_t* org = pos;
            gguf_val_type nt = (gguf_val_type)rd32(&pos);
            uint64_t ne = rd64(&pos);
            for (uint64_t i = 0; i < ne; i++) pos += skipper(nt,pos);
            return pos - org;
        }
        default:
            ERR("Unknown/unsupported tag %d",t);
            abort();
    }
}

static inline bool supported_qweight(gguf_type t)
{
    switch (t) {
        case Q8_0: case Q8_1:
        case Q4_0: case Q4_1:
        case Q5_0: case Q5_1:
        case Q2_0:
        case IQ4_NL:
        case Q1_0: case LQ1_0: case LQ1_G:
        case Q2_K: case Q3_K: case Q4_K: case Q5_K: case Q6_K:
        case PQ2_0: case TQ1_0: case PTQ1_0: case TQ2_0:
        case MXFP4:
            return true;
        default:
            return false;
    }
}

static inline bool supported_rowweight(gguf_type t)
{
    return t == F32 || t == F16 || t == BF16 || t == F64 ||
           t == I8 || t == I16 || t == I32 || t == I64 ||
           supported_qweight(t);
}

/* Byte size of one quantized row of `len` elements. Every supported type has a
 * fixed block width, so this is a closed-form expression, not a lookup. */
static uint64_t row_size_of(gguf_type type, int len)
{
    switch (type) {
        case F32:  return sizeof(float) * len;
        case F16:  return sizeof(gguf_half) * len;
        case BF16: return sizeof(uint16_t) * len;
        case F64:  return sizeof(double) * len;
        case I8:   return len;
        case I16:  return sizeof(int16_t) * len;
        case I32:  return sizeof(int32_t) * len;
        case I64:  return sizeof(int64_t) * len;

        case Q8_0: assert(len % QK8_0 == 0);  return sizeof(block_q8_0)  * (len / QK8_0);
        case Q8_1: assert(len % QK8_1 == 0);  return sizeof(block_q8_1)  * (len / QK8_1);
        case Q4_0: assert(len % QK4_0 == 0);  return sizeof(block_q4_0)  * (len / QK4_0);
        case Q4_1: assert(len % QK4_1 == 0);  return sizeof(block_q4_1)  * (len / QK4_1);
        case Q5_0: assert(len % QK8_0 == 0);  return sizeof(block_q5_0)  * (len / QK8_0);
        case Q5_1: assert(len % QK8_0 == 0);  return sizeof(block_q5_1)  * (len / QK8_0);

        case Q2_0: assert(len % QK2_0 == 0);  return sizeof(block_q2_0)  * (len / QK2_0);
        case IQ4_NL:assert(len % QK4_NL == 0);return sizeof(block_iq4_nl)* (len / QK4_NL);
        case PQ2_0:assert(len % 128  == 0);   return sizeof(block_pq2_0) * (len / 128);
        case MXFP4:assert(len % QK_MXFP4 == 0);return sizeof(block_mxfp4)* (len / QK_MXFP4);
        case TQ1_0:assert(len % QK_K == 0);   return sizeof(block_tq1_0) * (len / QK_K);
        case PTQ1_0:assert(len % 128 == 0);   return sizeof(block_ptq1_0)* (len / 128);
        case TQ2_0:assert(len % QK_K == 0);   return sizeof(block_tq2_0) * (len / QK_K);
        case Q1_0: assert(len % QK1_0 == 0);  return sizeof(block_q1_0)  * (len / QK1_0);
        case LQ1_0:assert(len % QK8_0 == 0);  return sizeof(block_lq1_0) * (len / QK8_0);
        case LQ1_G:assert(len % 128  == 0);   return sizeof(block_lq1_G) * (len / 128);

        case Q2_K: assert(len % QK_K == 0);   return sizeof(block_q2_K)  * (len / QK_K);
        case Q3_K: assert(len % QK_K == 0);   return sizeof(block_q3_K)  * (len / QK_K);
        case Q4_K: assert(len % QK_K == 0);   return sizeof(block_q4_K)  * (len / QK_K);
        case Q5_K: assert(len % QK_K == 0);   return sizeof(block_q5_K)  * (len / QK_K);
        case Q6_K: assert(len % QK_K == 0);   return sizeof(block_q6_K)  * (len / QK_K);

        default:
            ERR("Unsupported row type %d",type);
            abort();
    }
}

static wtensor mkweight(uint8_t* ptr, const gguf_tensor& tz)
{
    wtensor r;
    r.ptr = ptr;
    r.type = tz.type;
    r.rsz = (uint32_t)row_size(tz.type,(int)tz.dims[0]);
    r.cols = (int)tz.dims[0];
    r.rows = tz.dims.size() > 1 ? (int)tz.dims[1] : 1;
    return r;
}

uint64_t row_size(gguf_type type, int len)
{
    return row_size_of(type,len);
}

bool gguf_model::open_mmap(const char* fn)
{
    fsize = 0;
    base = NULL;
    HANDLE hFile = open_wrap(fn);
    file = (intptr_t)hFile;
    if (file == (intptr_t)-1) {
        ERR("Can't open file %s",fn);
        return false;
    }

    fsize = file_size_wrap(hFile);
    base = (uint8_t*)mmap_wrap(hFile, fsize);
    if (base == MAP_FAILED) {
        ERR("Can't mmap() file %s",fn);
        close_mmap();
        return false;
    }

    return true;
}

void gguf_model::close_mmap()
{
    if (base && base != MAP_FAILED && fsize) munmap_wrap(base, fsize);
    if (file != (intptr_t)-1) close_wrap((HANDLE)file);
    base = NULL;
    fsize = 0;
    file = -1;
}

bool gguf_model::read_tokenizer()
{
    if (!meta_kv.count(TOKENS_KEY)) {
        ERR("Tokens array %s not found!",TOKENS_KEY);
        return false;
    }

    uint8_t* pos = base + meta_kv[TOKENS_KEY].off;
    gguf_val_type nt = (gguf_val_type)rd32(&pos);
    if (nt != GSTRING) {
        ERR("Unexpected tokens array type %d",nt);
        return false;
    }

    uint64_t ne = rd64(&pos);
    if (!vocab_size) vocab_size = (int)ne;
    if ((int)ne != vocab_size) {
        ERR("Unexpected number of tokens %I64u", (unsigned __int64)ne);
        return false;
    }
    tokens.resize(ne);

    for (uint64_t i = 0; i < ne; i++) {
        tokens[i] = rdstr(&pos);
        tokens_rev[tokens[i]] = (int)i;
    }

    tok_model = meta_kv.count(TOKENS_MODEL_KEY)? kvrdstr(TOKENS_MODEL_KEY) : "";
    tok_pre = meta_kv.count(TOKENS_PRE_KEY)? kvrdstr(TOKENS_PRE_KEY) : "";
    add_bos = meta_kv.count(TOKENS_ADD_BOS_KEY)? kvrdbool(TOKENS_ADD_BOS_KEY) : true;

    if (meta_kv.count(TOKENS_SCORE_KEY)) {
        pos = base + meta_kv[TOKENS_SCORE_KEY].off;
        nt = (gguf_val_type)rd32(&pos);
        if (nt != GFLOAT32) {
            ERR("Unexpected token scores type %d",nt);
            return false;
        }

        ne = rd64(&pos);
        tokscores.resize(ne);
        for (uint64_t i = 0; i < ne; i++) tokscores[i] = rdf32(&pos);
    }

    if (meta_kv.count(TOKENS_MERGES_KEY)) {
        pos = base + meta_kv[TOKENS_MERGES_KEY].off;
        nt = (gguf_val_type)rd32(&pos);
        if (nt != GSTRING) {
            ERR("Unexpected merge array type %d",nt);
            return false;
        }

        ne = rd64(&pos);
        merges.resize(ne);
        merge_rank.clear();
        for (uint64_t i = 0; i < ne; i++) {
            merges[i] = rdstr(&pos);
            merge_rank[merges[i]] = (int)i;
        }
    }

    return true;
}

bool gguf_model::read_gguf()
{
    uint8_t* p = base;

    if (memcmp(p,"GGUF",4)) {
        ERR("Wrong magic %4.4s",p);
        return false;
    }
    p += 4;

    uint32_t ver = rd32(&p);
    if (ver != 2 && ver != 3) {
        ERR("Wrong file version (%u)",ver);
        return false;
    }

    uint64_t nten = rd64(&p);
    if (!nten || nten > MAXTENSORS) {
        ERR("Suspicious number of tensors: %I64u", (unsigned __int64)nten);
        return false;
    }

    uint64_t meta = rd64(&p);
    if (!meta || meta > MAXMETAKV) {
        ERR("Suspicious number of KV pairs: %I64u", (unsigned __int64)meta);
        return false;
    }

    gguf_kv kv;
    for (uint64_t i = 0; i < meta; i++) {
        string key = rdstr(&p);
        kv.tag = (gguf_val_type)rd32(&p);
        kv.off = p - base;
        meta_kv[key] = kv;
        p += skipper(kv.tag,p);
    }

    for (uint64_t i = 0; i < nten; i++) {
        string key = rdstr(&p);
        uint32_t ndim = rd32(&p);

        gguf_tensor tz;
        tz.dims.resize(ndim);
        for (uint32_t j = 0; j < ndim; j++) tz.dims[j] = rd64(&p);
        tz.type = (gguf_type)rd32(&p);
        tz.off = rd64(&p);
        tensors[key] = tz;
    }

    uint64_t off = p - base;
    tensors_off = base + (off + (ALIGNMENT - (off % ALIGNMENT)) % ALIGNMENT);

    arch = kvrdstr(ARCH_KEY);
    tok_bos = meta_kv.count(TOKENS_BOS_KEY)? kvrd32(TOKENS_BOS_KEY) : 0;
    tok_eos = meta_kv.count(TOKENS_EOS_KEY)? kvrd32(TOKENS_EOS_KEY) : 0;
    tok_unk = meta_kv.count(TOKENS_UNK_KEY)? kvrd32(TOKENS_UNK_KEY) : 0;
    tok_sep = meta_kv.count(TOKENS_SEP_KEY)? kvrd32(TOKENS_SEP_KEY) : 0;
    tok_pad = meta_kv.count(TOKENS_PAD_KEY)? kvrd32(TOKENS_PAD_KEY) : 0;
    tok_cls = meta_kv.count(TOKENS_CLS_KEY)? kvrd32(TOKENS_CLS_KEY) : 0;
    tok_mask = meta_kv.count(TOKENS_MASK_KEY)? kvrd32(TOKENS_MASK_KEY) : 0;

    if (arch == "llama" || arch == "qwen" || arch == "qwen2" || arch == "qwen2.5" ||
        arch == "qwen2moe" || arch == "phi3") return read_llama();
    else if (arch == "qwen3" || arch == "qwen35" || arch == "qwen3.5" || arch == "qwen3.8") return read_qwen3();
    else if (arch == "gpt2") return read_gpt2();
    else if (arch == "bert") return read_bert();

    /* These must NOT fall through to read_llama(). They used to be listed there,
     * and because the loader silently binds whatever tensors it recognises, the
     * graph then ran with null weights and emitted confident, incoherent text.
     *
     *   gemma2/3/4  - RMSNorm is x*rsqrt(ms)*(1+w), not *w; the FFN is GeGLU, not
     *                 SwiGLU; attention is sliding-window; and both the attention
     *                 logits and the final logits are softcapped. gemma4 adds
     *                 per-layer feed_forward_length[], attention.shared_kv_layers,
     *                 attention.sliding_window_pattern[], 512-wide heads
     *                 (key_length/value_length) and
     *                 embedding_length_per_layer_input.
     *   mellum      - attn_q_norm/attn_k_norm plus a 64-expert MoE layout that the
     *                 llama path would bind to the wrong shapes, and YaRN rope.
     *
     * Refusing is the honest behaviour here: a wrong answer is worse than none. */
    ERR("Unsupported architecture '%s' - there is no loader for it. Refusing "
        "rather than running the LLaMA path and emitting garbage.",arch.c_str());
    return false;
}

bool gguf_model::finalize()
{
    // Tied output embeddings: many checkpoints omit output.weight and reuse the
    // token embedding matrix for the final projection.
    if (!t_out.ptr) t_out = t_embed;
    if (!t_out.ptr) {
        ERR("Model has neither output.weight nor token_embd.weight");
        return false;
    }
    // The vocab size is needed before buffers are allocated. Prefer the KV entry
    // but fall back to the token embedding width, which is authoritative.
    if (!vocab_size) {
        map<string,gguf_tensor>::const_iterator te = tensors.find(TOKENS_EMBED_KEY);
        if (te != tensors.end() && te->second.dims.size() > 1)
            vocab_size = (int)te->second.dims[1];
    }
    if (!vocab_size) {
        ERR("Cannot determine vocabulary size");
        return false;
    }
    return true;
}

bool gguf_model::read_llama()
{
    /* All LLaMA-family hparams live under "<arch>." and are dimension-driven. */
    const string p = arch + ".";
    const char* pfx = p.c_str();

    vocab_size = kv32(p + "vocab_size", 0);
    n_layers   = kv32(p + "block_count", 0);
    n_embed    = kv32(p + "embedding_length", 0);
    n_heads    = kv32(p + "attention.head_count", 0);
    n_kv_heads = kv32(p + "attention.head_count_kv", n_heads);
    n_context  = kv32(p + "context_length", 2048);
    rms_epsilon= kvf32(p + "attention.layer_norm_rms_epsilon", 1e-5f);
    rope_base  = kvf32(p + "rope.freq_base", 10000.0f);
    rope_dim   = kv32(p + "rope.dimension_count", 0);
    n_ff       = kv32(p + "feed_forward_length", 0);

    /* head_dim is not always n_embed/n_heads (e.g. MiniCPM5: 128 vs 96). Honour
     * the explicit key/value lengths when the checkpoint provides them. */
    head_dim = kv32(p + "attention.key_length", n_heads ? n_embed / n_heads : 0);
    const int value_dim = kv32(p + "attention.value_length", head_dim);
    kv_dim = n_kv_heads * value_dim;
    q_proj_dim = 0;
    has_qgate = false;
    moe_norm_w = (arch != "qwen2moe");

    if (!n_layers || !n_embed || !n_heads) {
        ERR("Missing LLaMA hparams for arch '%s'",pfx);
        return false;
    }
    if (!rope_dim) rope_dim = head_dim;

    tr.resize(n_layers);
    memset(tr.data(),0,n_layers * sizeof(trans_block));

    for (map<string,gguf_tensor>::const_iterator it = tensors.begin(); it != tensors.end(); ++it) {
        const string& key = it->first;
        uint8_t* ptr = tensors_off + it->second.off;
        if (!key.compare(0,4,"blk.") && (key.find(".weight") != string::npos || key.find(".bias") != string::npos)) {
            int id = atoi(key.c_str()+4);
            assert(id >= 0 && id < n_layers);

            /* Check combined QKV first (matches attn_q/k but not attn_qkv/proj) */
            if (key.find("qkv_proj") != string::npos || key.find("attn_qkv") != string::npos || key.find("query_key_value") != string::npos) {
                tr[id].att_q = mkweight(ptr,it->second);
                tr[id].att_k = tr[id].att_q;  /* placeholder, split later */
                tr[id].att_v = tr[id].att_q;
            }
            else if (key.find("attn_norm") != string::npos) tr[id].att_norm = (float*)ptr;
            else if (key.find("attn_q.bias") != string::npos) tr[id].att_q_bias = (float*)ptr;
            else if (key.find("attn_k.bias") != string::npos) tr[id].att_k_bias = (float*)ptr;
            else if (key.find("attn_v.bias") != string::npos) tr[id].att_v_bias = (float*)ptr;
            else if (key.find("attn_q") != string::npos) {
                tr[id].att_q = mkweight(ptr,it->second);
                /* the Q projection may be wider than n_embed (separate head_dim) */
                if (it->second.dims.size() > 1) q_proj_dim = (int)it->second.dims[1];
            }
            else if (key.find("attn_k") != string::npos) tr[id].att_k = mkweight(ptr,it->second);
            else if (key.find("attn_v") != string::npos) tr[id].att_v = mkweight(ptr,it->second);
            else if (key.find("attn_output") != string::npos) tr[id].att_out = mkweight(ptr,it->second);
            else if (key.find("ffn_norm") != string::npos) tr[id].ffn_norm.ptr = (uint8_t*)ptr;
            else if (key.find("ffn_gate_inp_shexp") != string::npos) tr[id].ffn_gate_inp_shexp = mkweight(ptr,it->second);
            else if (key.find("ffn_gate_inp") != string::npos) tr[id].ffn_gate_inp = mkweight(ptr,it->second);
            else if (key.find("ffn_up_exps") != string::npos) tr[id].ffn_up_exps = mkweight(ptr,it->second);
            else if (key.find("ffn_gate_exps") != string::npos) tr[id].ffn_gate_exps = mkweight(ptr,it->second);
            else if (key.find("ffn_down_exps") != string::npos) tr[id].ffn_down_exps = mkweight(ptr,it->second);
            else if (key.find("ffn_up_shexp") != string::npos) tr[id].ffn_up_shexp = mkweight(ptr,it->second);
            else if (key.find("ffn_gate_shexp") != string::npos) tr[id].ffn_gate_shexp = mkweight(ptr,it->second);
            else if (key.find("ffn_down_shexp") != string::npos) tr[id].ffn_down_shexp = mkweight(ptr,it->second);
            else if (key.find("ffn_up") != string::npos) tr[id].ffn_up = mkweight(ptr,it->second);
            else if (key.find("ffn_down") != string::npos) tr[id].ffn_down = mkweight(ptr,it->second);
            else if (key.find("ffn_gate") != string::npos) tr[id].ffn_gate = mkweight(ptr,it->second);
        }
        else if (key == TOKENS_EMBED_KEY) t_embed = mkweight(ptr,it->second);
        else if (key == OUTPUT_KEY) t_out = mkweight(ptr,it->second);
        else if (key == OUTPUT_NORM_KEY) t_outnorm = (float*)ptr;
    }

    /* Split combined QKV tensors into Q/K/V views */
    for (int id = 0; id < n_layers; id++) {
        if (tr[id].att_q.ptr && tr[id].att_k.ptr == tr[id].att_q.ptr && tr[id].att_v.ptr == tr[id].att_q.ptr) {
            uint64_t row_bytes = tr[id].att_q.rsz;
            tr[id].att_k.ptr = tr[id].att_q.ptr + (uint64_t)(n_heads * head_dim) * row_bytes;
            tr[id].att_v.ptr = tr[id].att_k.ptr + (uint64_t)kv_dim * row_bytes;
            tr[id].att_k.type = tr[id].att_q.type; tr[id].att_k.rsz = tr[id].att_q.rsz;
            tr[id].att_v.type = tr[id].att_q.type; tr[id].att_v.rsz = tr[id].att_q.rsz;
        }
    }

    if (q_proj_dim <= 0) q_proj_dim = n_heads * head_dim;
    has_qgate = (q_proj_dim == 2 * n_heads * head_dim);

    /* Phi-3/Phi-4 fuse the SwiGLU gate and up into one blk.N.ffn_up.weight of
     * width 2*n_ff (gate first, matching ggml's GLU split). Without this the
     * graph reads a null ffn_gate and produces nonsense. */
    for (int id = 0; id < n_layers; id++) {
        if (tr[id].ffn_up.ptr && !tr[id].ffn_gate.ptr && n_ff > 0 &&
            tr[id].ffn_up.rows == 2 * n_ff) {
            tr[id].ffn_up_fused = true;
            ffn_up_fused = true;
        }
    }

    /* MoE: derive expert counts from the KV pairs, falling back to tensor shapes. */
    n_expert      = kv32(p + "expert_count", 0);
    n_expert_used = kv32(p + "expert_used_count", 1);
    moe_shexp = false;
    for (int id = 0; id < n_layers; id++) {
        trans_block& b = tr[id];
        if (!b.ffn_gate_inp.ptr) continue;
        b.is_moe = true;
        if (n_expert <= 0) n_expert = b.ffn_gate_inp.rows;
        b.n_expert = n_expert;
        b.n_expert_used = n_expert_used;
        b.moe_ff = b.ffn_up_exps.rows ? b.ffn_up_exps.rows : n_ff;
        if (b.ffn_up_shexp.ptr) { b.shexp_ff = b.ffn_up_shexp.rows; moe_shexp = true; }
    }

    assert(supported_rowweight(t_embed.type));
    assert(supported_rowweight(t_out.type));
    return true;
}

bool gguf_model::read_qwen3()
{
    n_layers = kvrd32(meta_kv.count(QWEN3_BLOCK_COUNT_KEY)? QWEN3_BLOCK_COUNT_KEY : (meta_kv.count("qwen35.block_count")? "qwen35.block_count" : "qwen3.block_count"));
    n_embed = kvrd32(meta_kv.count(QWEN3_EMBED_LEN_KEY)? QWEN3_EMBED_LEN_KEY : "qwen35.embedding_length");
    n_heads = kvrd32(meta_kv.count(QWEN3_HEAD_COUNT_KEY)? QWEN3_HEAD_COUNT_KEY : "qwen35.attention.head_count");
    n_kv_heads = kvrd32(meta_kv.count(QWEN3_HEAD_KV_COUNT_KEY)? QWEN3_HEAD_KV_COUNT_KEY : "qwen35.attention.head_count_kv");
    n_context = kvrd32(meta_kv.count(QWEN3_CONTEXT_LEN_KEY)? QWEN3_CONTEXT_LEN_KEY : "qwen35.context_length");
    rms_epsilon = kvrdf32(meta_kv.count(QWEN3_RMS_EPSILON_KEY)? QWEN3_RMS_EPSILON_KEY : "qwen35.attention.layer_norm_rms_epsilon");
    rope_base = meta_kv.count(QWEN3_ROPE_BASE_KEY)? kvrdf32(QWEN3_ROPE_BASE_KEY) : (meta_kv.count("qwen35.rope.freq_base")? kvrdf32("qwen35.rope.freq_base") : 1000000.0f);
    rope_scaling = meta_kv.count(QWEN3_ROPE_SCALING_TYPE_KEY)? kvrdstr(QWEN3_ROPE_SCALING_TYPE_KEY) : (meta_kv.count("qwen35.rope.scaling.type")? kvrdstr("qwen35.rope.scaling.type") : "");
    rope_scale = meta_kv.count(QWEN3_ROPE_SCALING_FACTOR_KEY)? kvrdf32(QWEN3_ROPE_SCALING_FACTOR_KEY) : (meta_kv.count("qwen35.rope.scaling.factor")? kvrdf32("qwen35.rope.scaling.factor") : 1.0f);
    rope_orig_context = meta_kv.count(QWEN3_ROPE_SCALING_ORIG_CTX_KEY)? kvrd32(QWEN3_ROPE_SCALING_ORIG_CTX_KEY) : (meta_kv.count("qwen35.rope.scaling.original_context_length")? kvrd32("qwen35.rope.scaling.original_context_length") : 0);
    head_dim = meta_kv.count(QWEN3_KEY_LEN_KEY)? kvrd32(QWEN3_KEY_LEN_KEY) : (meta_kv.count("qwen35.attention.key_length")? kvrd32("qwen35.attention.key_length") : (n_embed / n_heads));
    rope_dim = meta_kv.count("qwen35.rope.dimension_count")? kvrd32("qwen35.rope.dimension_count") : head_dim;
    n_ff = kvrd32(meta_kv.count(QWEN3_FF_LEN_KEY)? QWEN3_FF_LEN_KEY : "qwen35.feed_forward_length");
    int value_dim = meta_kv.count(QWEN3_VALUE_LEN_KEY)? kvrd32(QWEN3_VALUE_LEN_KEY) : (meta_kv.count("qwen35.attention.value_length")? kvrd32("qwen35.attention.value_length") : head_dim);
    kv_dim = n_kv_heads * value_dim;
    q_proj_dim = 0;
    has_qgate = false;
    vocab_size = 0;

    int full_attn_iv = meta_kv.count("qwen35.full_attention_interval") ? kvrd32("qwen35.full_attention_interval") : 0;

    tr.resize(n_layers);
    memset(tr.data(),0,n_layers * sizeof(trans_block));

    for (int id = 0; id < n_layers; id++) {
        tr[id].is_full_attn = (full_attn_iv > 0) ? ((id % full_attn_iv) == (full_attn_iv - 1)) : true;
    }

    for (map<string,gguf_tensor>::const_iterator it = tensors.begin(); it != tensors.end(); ++it) {
        const string& key = it->first;
        uint8_t* ptr = tensors_off + it->second.off;
        if (!key.compare(0,4,"blk.") && (key.find(".weight") != string::npos || key.find(".bias") != string::npos)) {
            int id = atoi(key.c_str()+4);
            if (id < 0 || id >= n_layers) continue;

            if (key.find("attn_qkv") != string::npos || key.find("qkv_proj") != string::npos || key.find("query_key_value") != string::npos) {
                tr[id].att_q = mkweight(ptr,it->second);
                tr[id].att_k = tr[id].att_q;
                tr[id].att_v = tr[id].att_q;
            }
            else if (key.find("attn_norm") != string::npos && key.find("post") == string::npos && key.find("pre") == string::npos) tr[id].att_norm = (float*)ptr;
            else if (key.find("attn_post_norm") != string::npos || key.find("post_attention_norm") != string::npos) tr[id].att_post_norm = (float*)ptr;
            else if (key.find("attn_q_norm") != string::npos) tr[id].att_q_norm = (float*)ptr;
            else if (key.find("attn_k_norm") != string::npos) tr[id].att_k_norm = (float*)ptr;
            else if (key.find("attn_q") != string::npos && key.find("attn_qkv") == string::npos && key.find("q_norm") == string::npos) tr[id].att_q = mkweight(ptr,it->second);
            else if (key.find("attn_k") != string::npos && key.find("attn_qkv") == string::npos && key.find("k_norm") == string::npos) tr[id].att_k = mkweight(ptr,it->second);
            else if (key.find("attn_v") != string::npos && key.find("attn_qkv") == string::npos) tr[id].att_v = mkweight(ptr,it->second);
            else if ((key.find("attn_output") != string::npos || (key.find("attn_out") != string::npos)) && key.find("blk.") != string::npos) tr[id].att_out = mkweight(ptr,it->second);
            else if (key.find("ffn_norm") != string::npos) tr[id].ffn_norm = mkweight(ptr,it->second);
            else if (key.find("ffn_up") != string::npos) tr[id].ffn_up = mkweight(ptr,it->second);
            else if (key.find("ffn_down") != string::npos) tr[id].ffn_down = mkweight(ptr,it->second);
            else if (key.find("ffn_gate") != string::npos) tr[id].ffn_gate = mkweight(ptr,it->second);
        }
        else if (key == TOKENS_EMBED_KEY || key.find("token_embd") != string::npos) {
            if (!t_embed.ptr) t_embed = mkweight(ptr,it->second);
            if (!vocab_size && it->second.dims.size() > 1) vocab_size = (int)it->second.dims[1];
        }
        else if (key == OUTPUT_KEY || (key.find("output.weight") != string::npos && key.find("blk.") == string::npos)) {
            t_out = mkweight(ptr,it->second);
            if (!vocab_size && it->second.dims.size() > 1) vocab_size = (int)it->second.dims[1];
        }
        else if (key == OUTPUT_NORM_KEY || key.find("output_norm") != string::npos) {
            if (!t_outnorm) t_outnorm = (float*)ptr;
        }
    }

    for (int id = 0; id < n_layers; id++) {
        if (tr[id].att_q.ptr && tr[id].att_k.ptr == tr[id].att_q.ptr && tr[id].att_v.ptr == tr[id].att_q.ptr) {
            uint64_t row_bytes = tr[id].att_q.rsz;
            if (tr[id].is_full_attn) {
                tr[id].att_k.ptr = tr[id].att_q.ptr + (uint64_t)(n_heads * head_dim * 2) * row_bytes;
                tr[id].att_v.ptr = tr[id].att_k.ptr + (uint64_t)kv_dim * row_bytes;
                q_proj_dim = n_heads * head_dim * 2;
                has_qgate = true;
            } else {
                /* linear attention: need to figure out dimensions from tensor */
                /* For now, use reasonable defaults */
                tr[id].att_k.ptr = tr[id].att_q.ptr + (uint64_t)(n_heads * head_dim * 2) * row_bytes;
                tr[id].att_v.ptr = tr[id].att_k.ptr + (uint64_t)kv_dim * row_bytes;
            }
            tr[id].att_k.type = tr[id].att_q.type; tr[id].att_k.rsz = tr[id].att_q.rsz;
            tr[id].att_v.type = tr[id].att_q.type; tr[id].att_v.rsz = tr[id].att_q.rsz;
        }
    }
    if (q_proj_dim == 0) q_proj_dim = n_heads * head_dim;

    assert(supported_rowweight(t_embed.type));
    assert(supported_rowweight(t_out.type));
    return true;
}


/* GPT-2 is a different graph from LLaMA: LayerNorm with bias instead of RMSNorm,
 * learned absolute position embeddings instead of RoPE, GELU instead of SwiGLU,
 * and a bias on every projection. Q/K/V live in one fused [Q|K|V] tensor, which
 * is why the split below mirrors read_llama's. */
bool gguf_model::read_gpt2()
{
    const string p = arch + ".";
    const char* pfx = p.c_str();

    vocab_size = kv32(p + "vocab_size", 0);
    n_layers   = kv32(p + "block_count", 0);
    n_embed    = kv32(p + "embedding_length", 0);
    n_heads    = kv32(p + "attention.head_count", 0);
    n_kv_heads = kv32(p + "attention.head_count_kv", n_heads);
    n_context  = kv32(p + "context_length", 1024);
    rms_epsilon= kvf32(p + "attention.layer_norm_epsilon", 1e-5f);
    n_ff       = kv32(p + "feed_forward_length", 0);

    rope_base  = 0.0f; // no RoPE: positions are learned
    rope_dim   = 0;
    n_expert   = 0;
    n_expert_used = 0;
    has_qgate  = false;
    is_gpt2    = true;

    if (!n_layers || !n_embed || !n_heads || !n_ff) {
        ERR("Missing GPT-2 hparams for arch '%s'",pfx);
        return false;
    }

    head_dim   = n_embed / n_heads;
    q_proj_dim = n_heads * head_dim;
    kv_dim     = n_kv_heads * head_dim;

    tr.resize(n_layers);
    memset(tr.data(),0,n_layers * sizeof(trans_block));

    for (map<string,gguf_tensor>::const_iterator it = tensors.begin(); it != tensors.end(); ++it) {
        const string& key = it->first;
        uint8_t* ptr = tensors_off + it->second.off;

        if (!key.compare(0,4,"blk.") && (key.find(".weight") != string::npos || key.find(".bias") != string::npos)) {
            int id = atoi(key.c_str()+4);
            assert(id >= 0 && id < n_layers);

            /* every .bias test must precede the matching .weight substring test */
            if (key.find("attn_qkv.bias") != string::npos) {
                tr[id].att_q_bias = (float*)ptr; /* split into K/V biases below */
            }
            else if (key.find("attn_qkv") != string::npos) {
                tr[id].att_q = mkweight(ptr,it->second);
                tr[id].att_k = tr[id].att_q; /* placeholder, split below */
                tr[id].att_v = tr[id].att_q;
            }
            else if (key.find("attn_norm.bias") != string::npos) tr[id].att_norm_b = (float*)ptr;
            else if (key.find("attn_norm") != string::npos) tr[id].att_norm = (float*)ptr;
            else if (key.find("attn_output.bias") != string::npos) tr[id].att_out_bias = (float*)ptr;
            else if (key.find("attn_output") != string::npos) tr[id].att_out = mkweight(ptr,it->second);
            else if (key.find("ffn_norm.bias") != string::npos) tr[id].ffn_norm_b = (float*)ptr;
            else if (key.find("ffn_norm") != string::npos) tr[id].ffn_norm.ptr = ptr;
            else if (key.find("ffn_up.bias") != string::npos) tr[id].ffn_up_bias = (float*)ptr;
            else if (key.find("ffn_up") != string::npos) tr[id].ffn_up = mkweight(ptr,it->second);
            else if (key.find("ffn_down.bias") != string::npos) tr[id].ffn_down_bias = (float*)ptr;
            else if (key.find("ffn_down") != string::npos) tr[id].ffn_down = mkweight(ptr,it->second);
        }
        else if (key.find("token_embd") != string::npos) {
            t_embed = mkweight(ptr,it->second);
            if (!vocab_size && it->second.dims.size() > 1) vocab_size = (int)it->second.dims[1];
        }
        else if (key.find("position_embd") != string::npos) {
            t_posemb = mkweight(ptr,it->second);
            has_pos_embd = true;
        }
        else if (key.find("output_norm.bias") != string::npos) t_outnorm_b = (float*)ptr;
        else if (key.find("output_norm") != string::npos) t_outnorm = (float*)ptr;
        else if (key.find("output.weight") != string::npos) {
            t_out = mkweight(ptr,it->second);
            if (!vocab_size && it->second.dims.size() > 1) vocab_size = (int)it->second.dims[1];
        }
    }

    /* Split the fused [Q|K|V] projection and bias into per-part views. */
    for (int id = 0; id < n_layers; id++) {
        if (tr[id].att_q.ptr && tr[id].att_k.ptr == tr[id].att_q.ptr && tr[id].att_v.ptr == tr[id].att_q.ptr) {
            const uint64_t row_bytes = tr[id].att_q.rsz;
            const int q_width = n_heads * head_dim;
            tr[id].att_k.ptr = tr[id].att_q.ptr + (uint64_t)q_width * row_bytes;
            tr[id].att_v.ptr = tr[id].att_k.ptr + (uint64_t)kv_dim  * row_bytes;
            tr[id].att_k.type = tr[id].att_q.type; tr[id].att_k.rsz = tr[id].att_q.rsz;
            tr[id].att_v.type = tr[id].att_q.type; tr[id].att_v.rsz = tr[id].att_q.rsz;
        }
        if (tr[id].att_q_bias && !tr[id].att_k_bias && !tr[id].att_v_bias) {
            const int q_width = n_heads * head_dim;
            tr[id].att_k_bias = tr[id].att_q_bias + q_width;
            tr[id].att_v_bias = tr[id].att_k_bias + kv_dim;
        }
    }

    if (!t_out.ptr) t_out = t_embed; // tied output embeddings

    assert(supported_rowweight(t_embed.type));
    assert(supported_rowweight(t_out.type));
    return true;
}


bool gguf_model::read_bert()
{
    n_layers = kvrd32(BERT_BLOCK_COUNT_KEY);
    n_embed = kvrd32(BERT_EMBED_LEN_KEY);
    n_heads = kvrd32(BERT_HEAD_COUNT_KEY);
    n_kv_heads = n_heads;
    n_context = kvrd32(BERT_CONTEXT_LEN_KEY);
    rms_epsilon = kvrdf32(BERT_LN_EPSILON_KEY);
    rope_base = 0.0f;
    rope_dim = 0;
    n_ff = kvrd32(BERT_FF_LEN_KEY);
    head_dim = n_embed / n_heads;
    kv_dim = n_embed;

    bert.resize(n_layers);
    memset(bert.data(),0,bert.size() * sizeof(bert_block));

    for (map<string,gguf_tensor>::const_iterator it = tensors.begin(); it != tensors.end(); ++it) {
        const string& key = it->first;
        uint8_t* ptr = tensors_off + it->second.off;

        if (!key.compare(0,4,"blk.")) {
            if (key.find(".weight") == string::npos && key.find(".bias") == string::npos) continue;
            int id = atoi(key.c_str()+4);
            assert(id >= 0 && id < n_layers);

            bert_block &b = bert[id];
            if (key.find("attn_q.w") != string::npos) b.attn_q = mkweight(ptr,it->second);
            else if (key.find("attn_q.b") != string::npos) b.attn_q_bias = (float*)ptr;
            else if (key.find("attn_k.w") != string::npos) b.attn_k = mkweight(ptr,it->second);
            else if (key.find("attn_k.b") != string::npos) b.attn_k_bias = (float*)ptr;
            else if (key.find("attn_v.w") != string::npos) b.attn_v = mkweight(ptr,it->second);
            else if (key.find("attn_v.b") != string::npos) b.attn_v_bias = (float*)ptr;
            else if (key.find("attn_output.w") != string::npos) b.attn_out = mkweight(ptr,it->second);
            else if (key.find("attn_output.b") != string::npos) b.attn_out_bias = (float*)ptr;
            else if (key.find("attn_output_norm.w") != string::npos) b.attn_norm_w = (float*)ptr;
            else if (key.find("attn_output_norm.b") != string::npos) b.attn_norm_b = (float*)ptr;
            else if (key.find("ffn_up.w") != string::npos) b.ffn_up = mkweight(ptr,it->second);
            else if (key.find("ffn_up.b") != string::npos) b.ffn_up_bias = (float*)ptr;
            else if (key.find("ffn_down.w") != string::npos) b.ffn_down = mkweight(ptr,it->second);
            else if (key.find("ffn_down.b") != string::npos) b.ffn_down_bias = (float*)ptr;
            else if (key.find("layer_output_norm.w") != string::npos) b.ffn_norm_w = (float*)ptr;
            else if (key.find("layer_output_norm.b") != string::npos) b.ffn_norm_b = (float*)ptr;
        }
        else if (key == TOKENS_EMBED_KEY) bert_tok_embd = mkweight(ptr,it->second);
        else if (key == BERT_POS_EMBED_KEY) bert_pos_embd = mkweight(ptr,it->second);
        else if (key == BERT_TYPE_EMBED_KEY) bert_tok_types = mkweight(ptr,it->second);
        else if (key == BERT_EMB_NORM_W_KEY) bert_emb_norm_w = (float*)ptr;
        else if (key == BERT_EMB_NORM_B_KEY) bert_emb_norm_b = (float*)ptr;
    }

    assert(supported_rowweight(bert_tok_embd.type));
    assert(supported_rowweight(bert_pos_embd.type));
    assert(supported_rowweight(bert_tok_types.type));
    return true;
}
