import struct, sys, os

def rdstr(p, b):
    (n,) = struct.unpack_from("<Q", b, p); p += 8
    return b[p:p+n].decode("utf-8", "replace"), p + n

def skip(p, b, t):
    if t in (0, 1, 7): return p + 1
    if t in (2, 3): return p + 2
    if t in (4, 5, 6): return p + 4
    if t in (10, 11, 12): return p + 8
    if t == 8:
        _, p = rdstr(p, b); return p
    if t == 9:
        (et,) = struct.unpack_from("<I", b, p); p += 4
        (n,) = struct.unpack_from("<Q", b, p); p += 8
        for _ in range(n): p = skip(p, b, et)
        return p
    raise ValueError(t)

path = sys.argv[1]
want = sys.argv[2] if len(sys.argv) > 2 else "blk.0."
sz = os.path.getsize(path)
with open(path, "rb") as f:
    buf = f.read(min(sz, 512 << 20))
p = 24
nten, nkv = struct.unpack_from("<QQ", buf, 8)
kvs = []
for _ in range(nkv):
    k, p = rdstr(p, buf)
    (t,) = struct.unpack_from("<I", buf, p); p += 4
    if t == 8:
        v, p = rdstr(p, buf)
    elif t == 9:
        (et,) = struct.unpack_from("<I", buf, p); p += 4
        (n,) = struct.unpack_from("<Q", buf, p); p += 8
        arr = []
        for _ in range(n):
            if et == 8:
                s, p = rdstr(p, buf); arr.append(s)
            elif et in (4, 5):
                (x,) = struct.unpack_from("<i", buf, p); p += 4; arr.append(str(x))
            elif et == 6:
                (x,) = struct.unpack_from("<f", buf, p); p += 4; arr.append(str(x))
            else:
                p = skip(p, buf, et)
        v = "[" + ",".join(arr) + "]"
    elif t in (4, 5):
        (x,) = struct.unpack_from("<i", buf, p); p += 4; v = str(x)
    elif t == 6:
        (x,) = struct.unpack_from("<f", buf, p); p += 4; v = repr(x)
    elif t == 7:
        (x,) = struct.unpack_from("<B", buf, p); p += 1; v = str(bool(x))
    else:
        (x,) = struct.unpack_from("<q", buf, p); p += 8; v = str(x)
    kvs.append((k, v))

print("== KV (%d) ==" % len(kvs))
for k, v in kvs:
    print("  %-58s %s" % (k, v[:150]))

print("== tensors matching '%s' ==" % want)
seen = set()
for _ in range(nten):
    k, p = rdstr(p, buf)
    nd, = struct.unpack_from("<I", buf, p); p += 4
    dims = []
    for _ in range(nd):
        (d,) = struct.unpack_from("<Q", buf, p); p += 8; dims.append(d)
    (t,) = struct.unpack_from("<I", buf, p); p += 4
    (off,) = struct.unpack_from("<Q", buf, p); p += 8
    if want in k:
        key = (k.split(".", 2)[-1] if k.startswith("blk.") else k)
        if key not in seen or want != "blk.":
            seen.add(key)
            print("  %-50s %-16s type=%d" % (k, "x".join(str(d) for d in dims), t))
