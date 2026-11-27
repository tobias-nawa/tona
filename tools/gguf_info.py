#!/usr/bin/env python3
"""Reads the header and tensor directory of a GGUF file without loading the weights. Standard library only.

Usage: tools/gguf_info.py FILE.gguf [--json] [--tensors]
"""

import argparse
import json
import struct
import sys
from collections import defaultdict

# ggml_type -> (name, elements per block, bytes per block), as of ggml (llama.cpp b11822, ggml/src/ggml.c)
GGML_TYPES = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17),
    40: ("NVFP4", 64, 36),
    41: ("Q1_0", 128, 18),
    42: ("Q2_0", 64, 18),
}

# GGUF value types
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
SCALAR = {
    U8: "<B",
    I8: "<b",
    U16: "<H",
    I16: "<h",
    U32: "<I",
    I32: "<i",
    F32: "<f",
    BOOL: "<?",
    U64: "<Q",
    I64: "<q",
    F64: "<d",
}


class Reader:
    def __init__(self, f):
        self.f = f

    def read(self, fmt):
        n = struct.calcsize(fmt)
        return struct.unpack(fmt, self.f.read(n))[0]

    def string(self):
        n = self.read("<Q")
        return self.f.read(n).decode("utf-8", "replace")

    def value(self, t, max_array=64):
        if t in SCALAR:
            return self.read(SCALAR[t])
        if t == STR:
            return self.string()
        if t == ARR:
            et, n = self.read("<I"), self.read("<Q")
            if et in SCALAR and n > max_array:
                # skip long number arrays (tokenizer scores and the like)
                self.f.seek(n * struct.calcsize(SCALAR[et]), 1)
                return {"array": "skipped", "type": et, "len": n}
            items = [self.value(et, max_array) for _ in range(n)]
            return items if n <= max_array else {"array": "truncated", "len": n, "head": items[:8]}
        raise ValueError(f"unknown GGUF value type {t}")


def read_gguf(path: str) -> dict:
    with open(path, "rb") as f:
        r = Reader(f)
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"{path}: not a GGUF file (magic {magic!r})")
        version, n_tensors, n_kv = r.read("<I"), r.read("<Q"), r.read("<Q")
        kv = {}
        for _ in range(n_kv):
            key = r.string()
            kv[key] = r.value(r.read("<I"))
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            dims = [r.read("<Q") for _ in range(r.read("<I"))]
            ttype, offset = r.read("<I"), r.read("<Q")
            tensors.append({"name": name, "dims": dims, "type": ttype, "offset": offset})
    return {"version": version, "kv": kv, "tensors": tensors}


def tensor_bytes(t: dict) -> int | None:
    info = GGML_TYPES.get(t["type"])
    if not info:
        return None
    n = 1
    for d in t["dims"]:
        n *= d
    return n // info[1] * info[2]


def summary(g: dict) -> dict:
    by_type = defaultdict(lambda: [0, 0])
    unknown = set()
    for t in g["tensors"]:
        b = tensor_bytes(t)
        name = GGML_TYPES.get(t["type"], (f"unknown({t['type']})",))[0]
        if b is None:
            unknown.add(t["type"])
            b = 0
        by_type[name][0] += 1
        by_type[name][1] += b
    return {
        "tensors": len(g["tensors"]),
        "by_type": {k: {"count": c, "gib": round(b / 2**30, 2)} for k, (c, b) in sorted(by_type.items())},
        "unknown_types": sorted(unknown),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--tensors", action="store_true", help="list all tensors")
    a = ap.parse_args()
    g = read_gguf(a.path)
    if a.json:
        print(
            json.dumps(
                {
                    "version": g["version"],
                    "kv": g["kv"],
                    "summary": summary(g),
                    "tensors": g["tensors"] if a.tensors else None,
                },
                indent=1,
                ensure_ascii=False,
            )
        )
        return 0
    print(f"GGUF v{g['version']}, {len(g['kv'])} keys, {len(g['tensors'])} tensors")
    for k, v in g["kv"].items():
        s = json.dumps(v, ensure_ascii=False)
        print(f"  {k} = {s[:160]}")
    print(json.dumps(summary(g), indent=1, ensure_ascii=False))
    if a.tensors:
        for t in g["tensors"]:
            tn = GGML_TYPES.get(t["type"], (str(t["type"]),))[0]
            print(f"  {t['name']:48s} {tn:8s} {t['dims']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
