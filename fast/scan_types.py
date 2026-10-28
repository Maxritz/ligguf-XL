import struct, sys, os, glob

TN = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",9:"Q8_1",
 10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",15:"Q8_K",16:"IQ2_XXS",17:"IQ2_XS",
 18:"IQ3_XXS",19:"IQ1_S",20:"IQ4_NL",21:"IQ3_S",22:"IQ2_S",23:"IQ4_XS",24:"I8",25:"I16",
 26:"I32",27:"I64",28:"F64",29:"IQ1_M",30:"BF16",34:"TQ1_0",35:"TQ2_0",39:"MXFP4",
 40:"NVFP4",41:"Q1_0",42:"Q2_0",142:"PQ2_0"}

def rdstr(p, b):
    (n,) = struct.unpack_from("<Q", b, p)
    p += 8
    s = b[p:p+n].decode("utf-8", "replace")
    return s, p + n

def skip(p, b, t):
    # gguf_val_type: 0=UINT8 1=INT8 2=UINT16 3=INT16 4=UINT32 5=INT32
    #                6=FLOAT32 7=BOOL 8=STRING 9=ARRAY 10=UINT64 11=INT64 12=FLOAT64
    if t in (0, 1, 7):
        return p + 1
    if t in (2, 3):
        return p + 2
    if t in (4, 5, 6):
        return p + 4
    if t in (10, 11, 12):
        return p + 8
    if t == 8:
        _, p = rdstr(p, b)
        return p
    if t == 9:
        (et,) = struct.unpack_from("<I", b, p); p += 4
        (n,) = struct.unpack_from("<Q", b, p); p += 8
        for _ in range(n):
            p = skip(p, b, et)
        return p
    raise ValueError("tag %d" % t)

def scan(path):
    with open(path, "rb") as f:
        head = f.read(1 << 20)
        f.seek(0, 2); sz = f.tell()
    p = 0
    if head[:4] != b"GGUF":
        print(path, "not gguf"); return
    p = 4
    ver, = struct.unpack_from("<I", head, p); p += 4
    nten, = struct.unpack_from("<Q", head, p); p += 8
    nkv, = struct.unpack_from("<Q", head, p); p += 8
    arch = None
    kvpairs = {}
    # read kv with lazy mmap-ish approach: just read whole file header region in chunks
    with open(path, "rb") as f:
        f.seek(0); buf = f.read(min(sz, 64 << 20))
    p = 4 + 4 + 8 + 8
    for _ in range(nkv):
        k, p = rdstr(p, buf)
        (t,) = struct.unpack_from("<I", buf, p); p += 4
        if k == "general.architecture":
            v, p = rdstr(p, buf); arch = v
        elif k == "qwen35.full_attention_interval":
            (v,) = struct.unpack_from("<I", buf, p); p += 4; kvpairs[k] = v
        elif k in ("llama.block_count","qwen35.block_count","qwen3.block_count"):
            (v,) = struct.unpack_from("<I", buf, p); p += 4; kvpairs[k] = v
        elif k in ("llama.embedding_length","qwen35.embedding_length","qwen3.embedding_length"):
            (v,) = struct.unpack_from("<I", buf, p); p += 4; kvpairs[k] = v
        elif k in ("llama.attention.head_count","qwen35.attention.head_count"):
            (v,) = struct.unpack_from("<I", buf, p); p += 4; kvpairs[k] = v
        elif k in ("llama.attention.head_count_kv","qwen35.attention.head_count_kv"):
            (v,) = struct.unpack_from("<I", buf, p); p += 4; kvpairs[k] = v
        elif k in ("llama.attention.key_length","qwen35.attention.key_length"):
            (v,) = struct.unpack_from("<I", buf, p); p += 4; kvpairs[k] = v
        elif k in ("llama.attention.value_length","qwen35.attention.value_length"):
            (v,) = struct.unpack_from("<I", buf, p); p += 4; kvpairs[k] = v
        elif k in ("qwen35.ssm.conv_kernel","qwen35.ssm.inner_size","qwen35.ssm.state_size",
                   "qwen35.ssm.time_step_rank","qwen35.ssm.group_count",
                   "llama.expert_count","llama.expert_used_count","llama.feed_forward_length",
                   "llama.context_length"):
            (v,) = struct.unpack_from("<I", buf, p); p += 4; kvpairs[k] = v
        elif k in ("llama.attention.layer_norm_rms_epsilon","qwen35.attention.layer_norm_rms_epsilon"):
            (v,) = struct.unpack_from("<f", buf, p); p += 4; kvpairs[k] = round(v, 8)
        else:
            p = skip(p, buf, t)
    types = {}
    names = {}
    for _ in range(nten):
        k, p = rdstr(p, buf)
        nd, = struct.unpack_from("<I", buf, p); p += 4
        dims = []
        for _ in range(nd):
            (d,) = struct.unpack_from("<Q", buf, p); p += 8; dims.append(d)
        (t,) = struct.unpack_from("<I", buf, p); p += 4
        (off,) = struct.unpack_from("<Q", buf, p); p += 8
        types[t] = types.get(t, 0) + 1
        names.setdefault(k, []).append((dims, t))
    print("=" * 70)
    print(os.path.basename(path), "| ver", ver, "| arch", arch)
    print("  tensors", nten, "kv", nkv)
    for k, v in kvpairs.items():
        print("   ", k, "=", v)
    print("  types:", ", ".join("%s(%d)x%d" % (TN.get(t, "?"), t, c) for t, c in sorted(types.items())))
    return names

if __name__ == "__main__":
    files = []
    for a in sys.argv[1:]:
        if os.path.isdir(a):
            files += sorted(glob.glob(os.path.join(a, "*.gguf")))
        else:
            files.append(a)
    for f in files:
        try:
            scan(f)
        except Exception as e:
            print(f, "ERROR", e)
