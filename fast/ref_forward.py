"""Independent reference forward pass for LiGGUF cross-checking.

Parses a GGUF directly, dequantizes to float32, and runs the LLaMA graph in
plain numpy. Only used as a correctness oracle - no code is shared with the C++
engine except the GGUF layout.
"""
import struct, sys, os
import numpy as np

def read_gguf(path):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == b"GGUF"
    p = 4
    ver, = struct.unpack_from("<I", data, p); p += 4
    nten, = struct.unpack_from("<Q", data, p); p += 8
    nkv,  = struct.unpack_from("<Q", data, p); p += 8

    def rd_str(p):
        n, = struct.unpack_from("<Q", data, p); p += 8
        return data[p:p+n].decode("utf-8", "replace"), p+n

    def rd_val(p, t):
        if t == 0:  v, = struct.unpack_from("<B", data, p); return v, p+1
        if t == 1:  v, = struct.unpack_from("<b", data, p); return v, p+1
        if t == 2:  v, = struct.unpack_from("<H", data, p); return v, p+2
        if t == 3:  v, = struct.unpack_from("<h", data, p); return v, p+2
        if t == 4:  v, = struct.unpack_from("<I", data, p); return v, p+4
        if t == 5:  v, = struct.unpack_from("<i", data, p); return v, p+4
        if t == 6:  v, = struct.unpack_from("<f", data, p); return v, p+4
        if t == 7:  v, = struct.unpack_from("<B", data, p); return bool(v) if False else bool(v), p+1
        if t == 8:
            return rd_str(p)
        if t == 9:
            et, = struct.unpack_from("<I", data, p); p += 4
            n,  = struct.unpack_from("<Q", data, p); p += 8
            arr = []
            for _ in range(n):
                v, p = rd_val(p, et)
                arr.append(v)
            return arr, p
        if t == 10: v, = struct.unpack_from("<Q", data, p); return v, p+8
        if t == 11: v, = struct.unpack_from("<q", data, p); return v, p+8
        if t == 12: v, = struct.unpack_from("<d", data, p); return v, p+8
        raise ValueError("val type %d" % t)

    kv = {}
    for _ in range(nkv):
        k, p = rd_str(p)
        t, = struct.unpack_from("<I", data, p); p += 4
        v, p = rd_val(p, t)
        kv[k] = v

    tensors = {}
    for _ in range(nten):
        name, p = rd_str(p)
        nd, = struct.unpack_from("<I", data, p); p += 4
        dims = []
        for _ in range(nd):
            d, = struct.unpack_from("<Q", data, p); p += 8
            dims.append(d)
        t, = struct.unpack_from("<I", data, p); p += 4
        off, = struct.unpack_from("<Q", data, p); p += 8
        tensors[name] = (dims, t, off)

    align = kv.get("general.alignment", 32)
    base = (p + align - 1) // align * align
    return kv, tensors, base, data


QK8_0 = 32

def deq(data, base, dims, t, off, name):
    """Return float32 numpy array shaped [dims[1], dims[0]] (row-major rows=ne1)."""
    ne0 = dims[0]
    nrow = int(np.prod(dims[1:])) if len(dims) > 1 else 1
    ptr = base + off
    if t == 0:  # F32
        a = np.frombuffer(data, dtype="<f4", count=ne0*nrow, offset=ptr)
        return a.reshape(nrow, ne0).copy()
    if t == 1:  # F16
        a = np.frombuffer(data, dtype="<f2", count=ne0*nrow, offset=ptr)
        return a.astype(np.float32).reshape(nrow, ne0)
    if t == 8:  # Q8_0
        nb = ne0 // QK8_0
        raw = np.frombuffer(data, dtype=np.uint8, count=nb*34*nrow, offset=ptr).reshape(nrow, nb, 34)
        d = raw[:, :, 0:2].copy().view("<f2").astype(np.float32).reshape(nrow, nb, 1)
        q = raw[:, :, 2:34].view(np.int8).astype(np.float32)
        return (q * d).reshape(nrow, ne0)
    raise ValueError("unhandled type %d for %s" % (t, name))


def rmsnorm(x, w, eps):
    ss = np.mean(x.astype(np.float32)**2) + eps
    return (x / np.sqrt(ss)) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def main():
    path = sys.argv[1]
    kv, tensors, base, data = read_gguf(path)
    arch = kv.get("general.architecture")
    print("arch", arch)
    p = arch + "."
    n_layers = kv[p+"block_count"]
    n_embed  = kv[p+"embedding_length"]
    n_ff     = kv[p+"feed_forward_length"]
    n_heads  = kv[p+"attention.head_count"]
    n_kv     = kv[p+"attention.head_count_kv"]
    eps      = kv[p+"attention.layer_norm_rms_epsilon"]
    head_dim = kv.get(p+"attention.key_length", n_embed // n_heads)
    vdim     = kv.get(p+"attention.value_length", head_dim)
    kv_dim   = head_dim * n_kv
    vocab    = kv["tokenizer.ggml.tokens"]
    print("layers=%d embed=%d ff=%d heads=%d kvheads=%d head_dim=%d" % (
        n_layers, n_embed, n_ff, n_heads, n_kv, head_dim))
    # tokens: comma separated list from argv[2]
    if len(sys.argv) > 2:
        toks = [int(t) for t in sys.argv[2].split(",")]
    else:
        toks = [kv.get("tokenizer.ggml.bos_token_id", 1)]
    print("tokens", toks, [vocab[t] for t in toks])

    W = {}
    def get(name):
        return deq(data, base, tensors[name][0], tensors[name][1], tensors[name][2], name)
    def get1(name):
        return np.ravel(get(name))

    W["emb"]  = get("token_embd.weight")
    W["onorm"]= get1("output_norm.weight")
    out_name = "output.weight" if "output.weight" in tensors else "token_embd.weight"
    W["out"]  = get(out_name)
    for l in range(n_layers):
        for s in ("attn_norm","attn_q","attn_k","attn_v","attn_output","ffn_norm","ffn_gate","ffn_up","ffn_down"):
            W[(l,s)] = get("blk.%d.%s.weight" % (l, s))
        W[(l,"attn_norm")] = np.ravel(W[(l,"attn_norm")])
        W[(l,"ffn_norm")]  = np.ravel(W[(l,"ffn_norm")])
        for s in ("attn_q","attn_k","attn_v"):
            nm = "blk.%d.%s.bias" % (l, s)
            W[(l,s+"_bias")] = get1(nm) if nm in tensors else None

    # ---- forward over the token list with causal attention + RoPE ----
    rope_base = kv[p+"rope.freq_base"]
    rope_dim  = kv.get(p+"rope.dimension_count", head_dim)
    if rope_dim > head_dim: rope_dim = head_dim
    freq = np.array([rope_base ** (-2.0*m/rope_dim) for m in range(rope_dim//2)], dtype=np.float32)

    def rope(x, pos):
        # LLaMA/ggml half-split rope: pair (m, m + rope_dim/2)
        y = x.copy()
        half = rope_dim // 2
        for hh in range(x.shape[0]):
            for m in range(half):
                ang = pos * freq[m]
                c, s = np.cos(ang), np.sin(ang)
                a0, a1 = x[hh, m], x[hh, m + half]
                y[hh, m]        = a0*c - a1*s
                y[hh, m + half] = a0*s + a1*c
        return y

    rep = n_heads // n_kv

    def forward_last(toks, verbose=False):
        Kc = [np.zeros((0, n_kv, head_dim), dtype=np.float32) for _ in range(n_layers)]
        Vc = [np.zeros((0, n_kv, head_dim), dtype=np.float32) for _ in range(n_layers)]
        last = None
        for pos, tok in enumerate(toks):
            x = W["emb"][tok].astype(np.float32).copy()
            for l in range(n_layers):
                h = rmsnorm(x, W[(l,"attn_norm")], eps)
                q = (W[(l,"attn_q")] @ h).reshape(n_heads, head_dim)
                k = (W[(l,"attn_k")] @ h).reshape(n_kv, head_dim)
                v = (W[(l,"attn_v")] @ h).reshape(n_kv, head_dim)
                if W[(l,"attn_q_bias")] is not None: q = q + W[(l,"attn_q_bias")].reshape(n_heads, head_dim)
                if W[(l,"attn_k_bias")] is not None: k = k + W[(l,"attn_k_bias")].reshape(n_kv, head_dim)
                if W[(l,"attn_v_bias")] is not None: v = v + W[(l,"attn_v_bias")].reshape(n_kv, head_dim)
                q = rope(q, pos); k = rope(k, pos)
                Kc[l] = np.concatenate([Kc[l], k[None]], axis=0)
                Vc[l] = np.concatenate([Vc[l], v[None]], axis=0)
                ctx = np.zeros((n_heads, head_dim), dtype=np.float32)
                for hh in range(n_heads):
                    kh = hh // rep
                    scores = (Kc[l][:, kh, :] @ q[hh]) / np.sqrt(head_dim)
                    wts = np.exp(scores - scores.max()); wts /= wts.sum()
                    ctx[hh] = wts @ Vc[l][:, kh, :]
                x = x + W[(l,"attn_output")] @ ctx.reshape(-1)
                h2 = rmsnorm(x, W[(l,"ffn_norm")], eps)
                x = x + W[(l,"ffn_down")] @ (silu(W[(l,"ffn_gate")] @ h2) * (W[(l,"ffn_up")] @ h2))
            x = rmsnorm(x, W["onorm"], eps)
            last = W["out"] @ x
            if verbose:
                top = np.argsort(-last)[:3]
                print("  ref pos %d tok %d %r -> %s" % (
                    pos, tok, vocab[tok], ["%d(%r)=%.3f" % (i, vocab[i], last[i]) for i in top]))
        return last

    if len(sys.argv) > 3 and sys.argv[3] == "gen":
        n_gen = int(sys.argv[4]) if len(sys.argv) > 4 else 10
        cur = list(toks)
        out_ids = []
        for _ in range(n_gen):
            lg = forward_last(cur)
            nxt = int(np.argmax(lg))
            out_ids.append(nxt); cur.append(nxt)
        print("greedy continuation for", toks, ":")
        print("ids :", out_ids)
        print("text:", "".join(vocab[i] for i in out_ids))
        return

    logits = forward_last(toks, verbose=True)
    order = np.argsort(-logits)[:10]
    print("top-10 logits after %d tokens (last=%d %r):" % (len(toks), toks[-1], vocab[toks[-1]]))
    for i in order:
        print("   %8d  %9.4f  %r" % (i, logits[i], vocab[i]))


main()
