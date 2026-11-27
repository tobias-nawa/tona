"""tools/coder_requant/dense_repack.py: Q3_K to Q6_K, Q4_0 and Q5_0 to Q8_0 without loss."""

import struct

import numpy as np
import pytest
from coder_requant import dense_repack as dr
from coder_requant import gguf_layout, slicing
from mini_model.gguf_io import KV, STR

rng = np.random.default_rng(146)


def f16_bytes(n: int) -> np.ndarray:
    """n finite fp16 scales as [n, 2] bytes, from tiny to large, both signs."""
    v = (rng.standard_normal(n) * 10.0 ** rng.integers(-4, 3, n)).astype("<f2")
    return v.view(np.uint8).reshape(n, 2)


def random_blocks(kind: int, n: int) -> np.ndarray:
    nbytes = dr.BLK[kind][1]
    b = rng.integers(0, 256, (n, nbytes), dtype=np.uint8)
    d_at = {dr.Q3_K: 108, dr.Q6_K: 208, dr.Q4_0: 0, dr.Q5_0: 0, dr.Q8_0: 0}[kind]
    b[:, d_at : d_at + 2] = f16_bytes(n)
    return b


# Scalar transcriptions of ggml's dequantize_row_* (ggml-quants.c), one block, the loop order of the C code


def ggml_q3k(b: bytes) -> list[float]:
    hm, qs, sc = b[0:32], b[32:96], b[96:108]
    d = float(np.frombuffer(b[108:110], "<f2")[0])
    a = list(struct.unpack("<3I", sc))
    tmp, k1, k2 = struct.unpack("<I", sc[8:12])[0], 0x03030303, 0x0F0F0F0F
    w = struct.unpack("<4I", sc[:8] + sc[8:12] + b"\0\0\0\0")
    w = [w[0], w[1], w[2], 0]
    w[2] = ((a[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4)
    w[3] = ((a[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4)
    w[0] = (a[0] & k2) | (((tmp >> 0) & k1) << 4)
    w[1] = (a[1] & k2) | (((tmp >> 2) & k1) << 4)
    scales = list(struct.pack("<4I", *w))
    out, is_, m, q = [], 0, 1, 0
    for _ in range(2):
        shift = 0
        for _ in range(4):
            for half in range(2):
                dl = np.float32(d) * np.float32(scales[is_] - 32)
                is_ += 1
                for ln in range(16):
                    v = ((qs[q + 16 * half + ln] >> shift) & 3) - (0 if hm[16 * half + ln] & m else 4)
                    out.append(float(np.float32(dl) * np.float32(v)))
            shift += 2
            m <<= 1
        q += 32
    return out


def ggml_q6k(b: bytes) -> list[float]:
    ql, qh = b[0:128], b[128:192]
    sc = np.frombuffer(b[192:208], np.int8)
    d = np.float32(np.frombuffer(b[208:210], "<f2")[0])
    out = [0.0] * 256
    for n in range(2):
        for ln in range(32):
            i = ln // 16
            o, qlb, qhb, s = 128 * n, 64 * n, 32 * n, 8 * n
            q = [
                ((ql[qlb + ln] & 0xF) | (((qh[qhb + ln] >> 0) & 3) << 4)) - 32,
                ((ql[qlb + ln + 32] & 0xF) | (((qh[qhb + ln] >> 2) & 3) << 4)) - 32,
                ((ql[qlb + ln] >> 4) | (((qh[qhb + ln] >> 4) & 3) << 4)) - 32,
                ((ql[qlb + ln + 32] >> 4) | (((qh[qhb + ln] >> 6) & 3) << 4)) - 32,
            ]
            for k in range(4):
                out[o + ln + 32 * k] = float(d * np.float32(sc[s + i + 2 * k]) * np.float32(q[k]))
    return out


def ggml_q5_0(b: bytes) -> list[float]:
    d, qh, qs = np.float32(np.frombuffer(b[0:2], "<f2")[0]), struct.unpack("<I", b[2:6])[0], b[6:22]
    out = [0.0] * 32
    for j in range(16):
        x0 = ((qs[j] & 0x0F) | (((qh >> (j + 0)) << 4) & 0x10)) - 16
        x1 = ((qs[j] >> 4) | ((qh >> (j + 12)) & 0x10)) - 16
        out[j], out[j + 16] = float(np.float32(x0) * d), float(np.float32(x1) * d)
    return out


def ggml_q4_0(b: bytes) -> list[float]:
    d, qs = np.float32(np.frombuffer(b[0:2], "<f2")[0]), b[2:18]
    out = [0.0] * 32
    for j in range(16):
        out[j], out[j + 16] = float(np.float32((qs[j] & 0xF) - 8) * d), float(np.float32((qs[j] >> 4) - 8) * d)
    return out


@pytest.mark.parametrize(
    "kind,ref",
    [(dr.Q3_K, ggml_q3k), (dr.Q6_K, ggml_q6k), (dr.Q5_0, ggml_q5_0), (dr.Q4_0, ggml_q4_0)],
)
def test_vectorized_dequant_equals_the_ggml_loops(kind, ref):
    blocks = random_blocks(kind, 64)
    got = dr.DEQUANT[kind](blocks)
    for i in range(len(blocks)):
        want = np.array(ref(blocks[i].tobytes()), np.float32)
        assert np.array_equal(got[i].view(np.uint32), want.view(np.uint32)), (kind, i)


@pytest.mark.parametrize(
    "src,dst,fn",
    [(dr.Q3_K, dr.Q6_K, dr.q3k_to_q6k), (dr.Q4_0, dr.Q8_0, dr.q4_0_to_q8_0), (dr.Q5_0, dr.Q8_0, dr.q5_0_to_q8_0)],
)
def test_repack_is_lossless_on_random_blocks(src, dst, fn):
    blocks = random_blocks(src, 2000)
    out = np.frombuffer(fn(blocks.tobytes()), np.uint8).reshape(len(blocks), dr.BLK[dst][1])
    a, b = dr.DEQUANT[src](blocks), dr.DEQUANT[dst](out)
    assert np.array_equal(a.reshape(-1).view(np.uint32), b.reshape(-1).view(np.uint32))


def test_q3k_extremes_map_inside_q6k():
    blocks = np.zeros((1, 110), np.uint8)  # all low bits 0, hmask 0: weight -4 everywhere
    blocks[:, 108:110] = np.array([0x00, 0x3C], np.uint8)  # d = 1.0
    out = np.frombuffer(dr.q3k_to_q6k(blocks.tobytes()), np.uint8).reshape(1, 210)
    assert np.array_equal(dr.dequant_q6k(out), dr.dequant_q3k(blocks))
    blocks[:, 0:96] = 255  # weight 3 everywhere
    out = np.frombuffer(dr.q3k_to_q6k(blocks.tobytes()), np.uint8).reshape(1, 210)
    assert np.array_equal(dr.dequant_q6k(out), dr.dequant_q3k(blocks))


def write_model(path, tensors: dict[str, tuple[int, list[int], np.ndarray]]) -> None:
    arch = b"mini"
    kvs = [KV("general.architecture", STR, -1, struct.pack("<Q", len(arch)) + arch)]
    outs = []
    for name, (ttype, dims, blocks) in tensors.items():
        (path.parent / f"src-{name}").write_bytes(blocks.tobytes())
        outs.append(slicing.OutTensor(name, dims, ttype, [slicing.Chunk(name, 0, blocks.size)]))
    slicing.write(str(path), kvs, outs, slicing.Hooks(path=lambda n: str(path.parent / f"src-{n}")))


def test_write_and_verify_a_file(tmp_path):
    src, out = tmp_path / "in.gguf", tmp_path / "out.gguf"
    norm = np.arange(16, dtype=np.uint8)
    write_model(src, {
        "token_embd.weight": (dr.Q3_K, [512, 3], random_blocks(dr.Q3_K, 6)),
        "blk.0.a.weight": (dr.Q4_0, [64, 2], random_blocks(dr.Q4_0, 4)),
        "blk.0.b.weight": (dr.Q5_0, [64, 2], random_blocks(dr.Q5_0, 4)),
        "blk.0.norm.weight": (0, [4], norm),
    })  # fmt: skip
    counts = dr.write_file(str(src), str(out))
    assert counts.converted == {dr.Q3_K: 1, dr.Q4_0: 1, dr.Q5_0: 1}
    lay = gguf_layout.parse(out.read_bytes())
    assert [t.type for t in lay.tensors] == [dr.Q6_K, dr.Q8_0, dr.Q8_0, 0]
    assert dr.verify(str(src), str(out)) == {"Q3_K": 6, "Q4_0": 4, "Q5_0": 4}
    with pytest.raises(FileExistsError):  # never overwrites
        dr.write_file(str(src), str(out))


def test_verify_finds_a_changed_block(tmp_path):
    src, out = tmp_path / "in.gguf", tmp_path / "out.gguf"
    write_model(src, {"blk.0.a.weight": (dr.Q4_0, [64, 2], random_blocks(dr.Q4_0, 4))})
    dr.write_file(str(src), str(out))
    raw = bytearray(out.read_bytes())
    lay = gguf_layout.parse(bytes(raw))
    raw[lay.data_start + 34 + 5] ^= 1  # a weight in the second Q8_0 block
    out.write_bytes(raw)
    with pytest.raises(ValueError, match="block 1"):
        dr.verify(str(src), str(out))
