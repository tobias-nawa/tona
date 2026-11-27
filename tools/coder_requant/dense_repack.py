"""Dense Q3_K, Q4_0 and Q5_0 tensors to the types Tona has kernels for (Q6_K, Q8_0), without loss.

The 512-expert Flash-Next files mix these three types into the dense part (embedding, attention, shared experts). The
engine has no kernels for them, and every one fits a type it has:

- Q3_K -> Q6_K: the 16 scales are 6 bits minus 32 and fit the int8 scales of Q6_K (scale8 = scale6 - 32); a Q3_K weight
  is a 3-bit value minus 4 (range -4 .. 3), Q6_K stores a 6-bit value minus 32, so q6 = q3 + 28 gives
  d * scale * (q6 - 32) = d * (scale6 - 32) * (q3 - 4); both formulas use the same fp16 d.
- Q4_0 -> Q8_0 and Q5_0 -> Q8_0: weight = d * (nibble - 8) or d * (5-bit value - 16), Q8_0 stores d * int8.

The sizes grow (0.72 GiB of Q3_K becomes 1.4 GiB), so the output is a new file, not an in-place patch. `verify`
dequantizes the source and the repacked tensors with independent formulas and compares every block bit for bit.
"""

import sys
from dataclasses import dataclass

import numpy as np

from coder_requant import slicing
from coder_requant.gguf_layout import Layout, Tensor, parse, tensor_bytes

Q4_0, Q5_0, Q8_0, Q3_K, Q6_K = 2, 6, 8, 11, 14
TARGET = {Q3_K: Q6_K, Q4_0: Q8_0, Q5_0: Q8_0}
HEADER_READ = 64 << 20  # enough for the header of the 512-expert files (~25 MB)
BLOCKS_PER_PIECE = 1 << 18  # ~29 MB of Q3_K per piece; bounds the memory of the numpy temporaries (~1 GB)
BLK = {Q4_0: (32, 18), Q5_0: (32, 22), Q8_0: (32, 34), Q3_K: (256, 110), Q6_K: (256, 210)}  # (weights, bytes)

E = np.arange(256)  # element index inside a super-block
_N, _J, _H, _L = E // 128, (E % 128) // 32, (E % 32) // 16, E % 16  # ggml's loop nest of dequantize_row_q3_K
_SC = np.arange(16)


def _f16(raw: np.ndarray) -> np.ndarray:
    """fp16 as the two little-endian bytes at the end of blocks [n, 2] -> float32 [n]."""
    return np.ascontiguousarray(raw).view("<f2").reshape(-1).astype(np.float32)


# Q3_K


def q3k_scales(blocks: np.ndarray) -> np.ndarray:
    """The 16 scales of Q3_K blocks [n, 110] as int8 values (6 bits minus 32), as ggml's kmask unpacking."""
    sc, g = blocks[:, 96:108], _SC // 4
    low = (sc[:, (_SC % 4) + 4 * (g % 2)] >> (4 * (g // 2))) & 0xF
    high = (sc[:, 8 + _SC % 4] >> (2 * g)) & 3
    return (low | (high << 4)).astype(np.int16) - 32


def q3k_values(blocks: np.ndarray) -> np.ndarray:
    """The weights of Q3_K blocks [n, 110] as integers -4 .. 3, [n, 256] in element order."""
    qs = blocks[:, 32 + _N * 32 + _H * 16 + _L]
    hm = blocks[:, _H * 16 + _L]
    low = (qs >> (2 * _J)) & 3
    high = (hm >> (4 * _N + _J)) & 1
    return low.astype(np.int16) - np.where(high == 1, 0, 4)


def dequant_q3k(blocks: np.ndarray) -> np.ndarray:
    d = _f16(blocks[:, 108:110])[:, None]
    dl = d * q3k_scales(blocks)[:, E // 16].astype(np.float32)
    return dl * q3k_values(blocks).astype(np.float32)


# Q6_K


def dequant_q6k(blocks: np.ndarray) -> np.ndarray:
    """ggml's dequantize_row_q6_K: d * sc * (q6 - 32), the 4 sextets of a 128-weight half from ql and qh."""
    n, k, ln = E // 128, (E % 128) // 32, E % 32
    ql = blocks[:, 64 * n + 32 * (k & 1) + ln]
    qh = blocks[:, 128 + 32 * n + ln]
    q = ((ql >> (4 * (k >= 2))) & 0xF) | (((qh >> (2 * k)) & 3) << 4)
    d = _f16(blocks[:, 208:210])[:, None]
    sc = blocks[:, 192 + E // 16].view(np.int8).astype(np.float32)
    return (d * sc) * (q.astype(np.int16) - 32).astype(np.float32)


def q3k_to_q6k(data: bytes) -> bytes:
    blocks = np.frombuffer(data, np.uint8).reshape(-1, 110)
    q6 = (q3k_values(blocks) + 32).astype(np.uint8)  # 28 .. 35
    out = np.zeros((len(blocks), 210), np.uint8)
    n, k, ln = E // 128, (E % 128) // 32, E % 32
    for j, (nn, kk, ll) in enumerate(zip(n, k, ln, strict=True)):  # 256 element positions, vectorized over the blocks
        v = q6[:, j]
        out[:, 64 * nn + 32 * (kk & 1) + ll] |= ((v & 0xF) << (4 * (kk >= 2))).astype(np.uint8)
        out[:, 128 + 32 * nn + ll] |= ((v >> 4) << (2 * kk)).astype(np.uint8)
    out[:, 192:208] = q3k_scales(blocks).astype(np.int8).view(np.uint8)
    out[:, 208:210] = blocks[:, 108:110]
    return out.tobytes()


# Q4_0, Q5_0, Q8_0


def dequant_q8_0(blocks: np.ndarray) -> np.ndarray:
    return _f16(blocks[:, 0:2])[:, None] * blocks[:, 2:34].view(np.int8).astype(np.float32)


def q4_0_values(blocks: np.ndarray) -> np.ndarray:
    qs = blocks[:, 2:18]
    return np.concatenate([qs & 0xF, qs >> 4], axis=1).astype(np.int16) - 8  # low nibbles are weights 0..15


def q5_0_values(blocks: np.ndarray) -> np.ndarray:
    qs, qh = blocks[:, 6:22], np.ascontiguousarray(blocks[:, 2:6]).view("<u4").reshape(-1, 1)
    j = np.arange(16)
    lo = (qs & 0xF).astype(np.int16) | (((qh >> j) & 1) << 4).astype(np.int16)
    hi = (qs >> 4).astype(np.int16) | (((qh >> (j + 16)) & 1) << 4).astype(np.int16)
    return np.concatenate([lo, hi], axis=1) - 16


def dequant_q4_0(blocks: np.ndarray) -> np.ndarray:
    return q4_0_values(blocks).astype(np.float32) * _f16(blocks[:, 0:2])[:, None]


def dequant_q5_0(blocks: np.ndarray) -> np.ndarray:
    return q5_0_values(blocks).astype(np.float32) * _f16(blocks[:, 0:2])[:, None]


def to_q8_0(blocks: np.ndarray, values: np.ndarray) -> bytes:
    out = np.empty((len(blocks), 34), np.uint8)
    out[:, 0:2] = blocks[:, 0:2]
    out[:, 2:] = values.astype(np.int8).view(np.uint8)
    return out.tobytes()


def q4_0_to_q8_0(data: bytes) -> bytes:
    blocks = np.frombuffer(data, np.uint8).reshape(-1, 18)
    return to_q8_0(blocks, q4_0_values(blocks))


def q5_0_to_q8_0(data: bytes) -> bytes:
    blocks = np.frombuffer(data, np.uint8).reshape(-1, 22)
    return to_q8_0(blocks, q5_0_values(blocks))


CONVERT = {Q3_K: q3k_to_q6k, Q4_0: q4_0_to_q8_0, Q5_0: q5_0_to_q8_0}
DEQUANT = {Q3_K: dequant_q3k, Q6_K: dequant_q6k, Q4_0: dequant_q4_0, Q5_0: dequant_q5_0, Q8_0: dequant_q8_0}


# the file


def out_type(t: Tensor) -> int:
    return TARGET.get(t.type, t.type)


def plan(layout: Layout, src: str) -> list[slicing.OutTensor]:
    """Every tensor as a byte range of the source with its new type; a converted tensor has its new size, its chunk
    is a placeholder for the header (the data is written by convert_tensor)."""
    out = []
    for t in layout.tensors:
        nbytes = tensor_bytes(out_type(t), t.dims)
        out.append(slicing.OutTensor(t.name, list(t.dims), out_type(t), [slicing.Chunk(src, t.offset, nbytes)]))
    return out


@dataclass
class Counts:
    converted: dict[int, int]  # source type -> tensors


def pieces(f, start: int, nbytes: int, block_bytes: int):
    """The bytes of a tensor in pieces of whole blocks."""
    step = BLOCKS_PER_PIECE * block_bytes
    f.seek(start)
    for at in range(0, nbytes, step):
        yield f.read(min(step, nbytes - at))


def write_file(src: str, dst: str) -> Counts:
    """A copy of src (the header of a GGUF shard, all its tensors) with Q3_K as Q6_K and Q4_0, Q5_0 as Q8_0."""
    with open(src, "rb") as f:
        layout = parse(f.read(HEADER_READ))
    tensors, align = plan(layout, "src"), layout.alignment
    counts = Counts({})
    with open(src, "rb") as f, open(dst, "xb") as out:
        out.write(slicing.header(layout.kvs, tensors, align))
        for t in layout.tensors:
            start = layout.data_start + t.offset
            if t.type in CONVERT:
                counts.converted[t.type] = counts.converted.get(t.type, 0) + 1
                for piece in pieces(f, start, t.nbytes, BLK[t.type][1]):
                    out.write(CONVERT[t.type](piece))
                written = tensor_bytes(out_type(t), t.dims)
            else:
                for piece in pieces(f, start, t.nbytes, 1 << 20):
                    out.write(piece)
                written = t.nbytes
            out.write(bytes(-written % align))
    return counts


def verify(src: str, dst: str) -> dict[str, int]:
    """Dequantizes every converted tensor of src and dst and compares the float32 bit patterns of all weights.
    Returns the number of blocks compared per source type name; raises ValueError at the first difference."""
    names = {Q3_K: "Q3_K", Q4_0: "Q4_0", Q5_0: "Q5_0"}
    with open(src, "rb") as fs, open(dst, "rb") as fd:
        ls, ld = parse(fs.read(HEADER_READ)), parse(fd.read(HEADER_READ))
        if [t.name for t in ls.tensors] != [t.name for t in ld.tensors]:
            raise ValueError("the tensor lists differ")
        done = {n: 0 for n in names.values()}
        for a, b in zip(ls.tensors, ld.tensors, strict=True):
            if b.type != out_type(a) or a.dims != b.dims:
                raise ValueError(f"{a.name}: type or shape differs ({a.type} -> {b.type})")
            if a.type not in CONVERT:
                if not same_bytes(fs, ls.data_start + a.offset, fd, ld.data_start + b.offset, a.nbytes):
                    raise ValueError(f"{a.name}: bytes of an unconverted tensor differ")
                continue
            done[names[a.type]] += compare(fs, ls.data_start + a.offset, fd, ld.data_start + b.offset, a, b)
    return done


def same_bytes(fa, pa: int, fb, pb: int, n: int) -> bool:
    for at in range(0, n, 64 << 20):
        fa.seek(pa + at), fb.seek(pb + at)
        m = min(64 << 20, n - at)
        if fa.read(m) != fb.read(m):
            return False
    return True


def compare(fs, ps: int, fd, pd: int, a: Tensor, b: Tensor) -> int:
    """Bit-for-bit comparison of the dequantized weights of one converted tensor; returns the blocks compared."""
    nblk, bs, bd = a.nbytes // BLK[a.type][1], BLK[a.type][1], BLK[b.type][1]
    for first in range(0, nblk, BLOCKS_PER_PIECE):
        n = min(BLOCKS_PER_PIECE, nblk - first)
        fs.seek(ps + first * bs), fd.seek(pd + first * bd)
        sa = np.frombuffer(fs.read(n * bs), np.uint8).reshape(n, bs)
        sb = np.frombuffer(fd.read(n * bd), np.uint8).reshape(n, bd)
        wa = DEQUANT[a.type](sa).reshape(-1, 256 if a.type == Q3_K else 32)
        wb = DEQUANT[b.type](sb).reshape(-1, 256 if b.type == Q6_K else 32)
        if not np.array_equal(wa.view(np.uint32), wb.view(np.uint32)):
            bad = int(np.argmax((wa.view(np.uint32) != wb.view(np.uint32)).any(axis=1)))
            raise ValueError(f"{a.name}: block {first + bad} dequantizes differently")
    return nblk


def main(argv: list[str]) -> None:
    if len(argv) != 3 or argv[0] not in ("write", "verify"):
        sys.exit("usage: dense_repack.py write|verify SOURCE.gguf OUTPUT.gguf")
    cmd, src, dst = argv
    if cmd == "write":
        counts = write_file(src, dst)
        print(f"{dst}: converted tensors by source type {counts.converted}")
    else:
        print("blocks compared bit for bit:", verify(src, dst))


if __name__ == "__main__":
    main(sys.argv[1:])
