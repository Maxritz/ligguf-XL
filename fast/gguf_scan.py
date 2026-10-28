#!/usr/bin/env python3
"""Authoritative GGUF header/tensor scanner.

dump_gguf.exe bails on some files (gemma4) and its text confuses array value
types with tensor types, so this reads the container directly.

Usage: python gguf_scan.py <model.gguf> [--tensors N] [--arch]
"""
import struct, sys, collections

GGUF_MAGIC = 0x46554747

# ggml_type ids, llama.cpp/include/ggml.h (the ABI we must match)
GGML_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 4: "Q4_2(removed)", 5: "Q4_3(removed)",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
    13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8",
    25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 31: "Q4_0_4_4",
    32: "Q4_0_4_8", 33: "Q4_0_8_8", 34: "TQ1_0", 35: "TQ2_0", 36: "IQ4_NL_4_4",
    37: "IQ4_NL_4_8", 38: "IQ4_NL_8_8", 39: "MXFP4", 40: "COUNT",
}
KV_TYPE_NAMES = {0: "u8", 1: "i8", 2: "u16", 3: "i16", 4: "u32", 5: "i32",
                 6: "f32", 7: "bool", 8: "str", 9: "array", 10: "u64", 11: "i64", 12: "f64"}
FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


class R:
    def __init__(self, f):
        self.f = f

    def raw(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("short read")
        return b

    def u32(self):
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.raw(8))[0]

    def s(self):
        return self.raw(self.u64()).decode("utf-8", "replace")

    def val(self, t):
        if t in FIXED:
            return struct.unpack("<" + {1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
                                        7: "?", 10: "Q", 11: "q", 12: "d"}[t], self.raw(FIXED[t]))[0]
        if t == 8:
            return self.s()
        if t == 9:
            et = self.u32()
            n = self.u64()
            return [self.val(et) for _ in range(n)]
        raise ValueError("kv type %d" % t)


def main():
    path = sys.argv[1]
    show = 0
    if "--tensors" in sys.argv:
        show = int(sys.argv[sys.argv.index("--tensors") + 1])
    only = "--arch" in sys.argv
    with open(path, "rb") as f:
        r = R(f)
        magic = r.u32()
        if magic != GGUF_MAGIC:
            print("not a gguf (magic %08x)" % magic)
            return 1
        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        kv = {}
        for _ in range(n_kv):
            k = r.s()
            t = r.u32()
            kv[k] = r.val(t)
        arch = kv.get("general.architecture", "?")
        if only:
            print("%s\t%s\tv%d\t%d tensors\t%d kv" % (path.split("\\")[-1], arch, version, n_tensors, n_kv))
            for k in sorted(kv):
                if k.startswith(arch + ".") or k.startswith("general."):
                    print("  %s = %r" % (k, kv[k]))
            return 0
        tensors = []
        for _ in range(n_tensors):
            name = r.s()
            nd = r.u32()
            dims = [r.u64() for _ in range(nd)]
            ty = r.u32()
            off = r.u64()
            tensors.append((name, dims, ty, off))
        print("%s" % path.split("\\")[-1])
        print("  version=%d arch=%s tensors=%d kv=%d" % (version, arch, n_tensors, n_kv))
        tl = collections.Counter()
        for _, _, ty, _ in tensors:
            tl[ty] += 1
        print("  types: " + ", ".join("%s(%d)x%d" % (GGML_TYPES.get(t, "?" + str(t)), t, n)
                                      for t, n in sorted(tl.items())))
        # block-0 tensor names, plus any name not matching blk.*
        names = [t[0] for t in tensors]
        b0 = [t for t in tensors if t[0].startswith("blk.0.") or not t[0].startswith("blk.")]
        for name, dims, ty, off in b0:
            print("  %-46s %-10s %s" % (name, GGML_TYPES.get(ty, "?" + str(ty)), dims))
        if show:
            for name, dims, ty, off in tensors[:show]:
                print("  %-46s %-10s %s" % (name, GGML_TYPES.get(ty, "?" + str(ty)), dims))
    return 0


if __name__ == "__main__":
    sys.exit(main())
