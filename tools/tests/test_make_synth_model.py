import math
import random
import struct

from mini_model import blocks, fast


def test_quant_sizes_and_scales():
    for ttype in (8, 12, 13, 14, 16, 18, 20, 21, 22, 23, 42):
        _, per_block, size = blocks.TYPES[ttype]
        n = per_block * 3000
        data = fast.quant_bytes(ttype, n, random.Random(1), 0.01)
        assert len(data) == blocks.nbytes(ttype, n)
        off = 208 if ttype == 14 else 0  # every block carries the same d
        d = {data[i * size + off : i * size + off + 2] for i in range(3000)}
        assert d == {blocks.fp16(0.01 / blocks.RMS_AT_UNIT_SCALE[ttype])}


def test_same_seed_same_bytes():
    a, b = (fast.quant_bytes(16, 256 * 10, random.Random(5), 1.0) for _ in range(2))
    assert a == b


def test_bf16_finite_near_sigma():
    raw = fast.bf16_bytes(random.Random(2), 1 << 16, 0.05)
    vals = [struct.unpack("<f", b"\0\0" + raw[i : i + 2])[0] for i in range(0, len(raw), 2)]
    assert all(math.isfinite(v) for v in vals)
    rms = math.sqrt(sum(v * v for v in vals) / len(vals))
    assert 0.5 * 0.05 < rms < 2 * 0.05
