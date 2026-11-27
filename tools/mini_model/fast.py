"""Random tensor data for models in full size, at C speed with the standard library only.

The per-block fields of blocks.quant_bytes (scales) are written with strided slices instead of a loop over the
blocks: fields that are the same in every block once, fields that vary from a pool of POOL random patterns, block i
taking pattern i % POOL. Large BF16 tensors take their high byte (sign and upper exponent bits) from 256 quantiles of
the normal distribution and their low byte at random: the random lowest exponent bit and mantissa put the rms at
about 1.34 sigma (measured on 65536 values), never NaN or inf. Different bytes than blocks.py for the same seed;
meant for speed measurements, where only shapes, types and sane magnitudes count.
"""

import random
import statistics
import struct

from . import blocks

POOL = 1021  # prime, so the pattern does not line up with rows of any power-of-two length
CHUNK = 1 << 26  # random.randbytes takes at most 2^31 bits per call


def _randbytes(rng: random.Random, n: int) -> bytearray:
    buf = bytearray()
    for o in range(0, n, CHUNK):
        buf += rng.randbytes(min(CHUNK, n - o))
    return buf


def _stamp(buf: bytearray, size: int, n: int, off: int, pattern: bytes) -> None:
    for j, b in enumerate(pattern):
        buf[off + j :: size] = bytes([b]) * n


def _cycle(buf: bytearray, size: int, n: int, off: int, patterns: list[bytes]) -> None:
    for j in range(len(patterns[0])):
        col = bytes(p[j] for p in patterns)
        buf[off + j :: size] = (col * (n // len(col) + 1))[:n]


def _kquant(rng: random.Random) -> bytes:
    sc = [rng.randint(8, 40) for _ in range(8)]
    return blocks._pack_k4(sc, sc)


def _q6k(rng: random.Random) -> bytes:
    return bytes(rng.choice((-1, 1)) * rng.randint(4, 12) & 255 for _ in range(16))


def _iq4xs(rng: random.Random) -> bytes:
    ls = [rng.randint(36, 44) for _ in range(8)]
    return bytes((ls[2 * i] & 15) | ((ls[2 * i + 1] & 15) << 4) for i in range(4))


def quant_bytes(ttype: int, elements: int, rng: random.Random, rms: float) -> bytes:
    """Same block contents as blocks.quant_bytes, scale patterns drawn from a pool."""
    _, per_block, size = blocks.TYPES[ttype]
    n = elements // per_block
    buf = _randbytes(rng, n * size)
    d = rms / blocks.RMS_AT_UNIT_SCALE[ttype]
    if ttype in (12, 13):
        mid = 7.5 if ttype == 12 else 15.5  # w = d * sc * (q - mid): min = mid * scale makes the block mean zero
        _stamp(buf, size, n, 0, blocks.fp16(d) + blocks.fp16(mid * d))
        _cycle(buf, size, n, 4, [_kquant(rng) for _ in range(POOL)])
    elif ttype == 14:
        _cycle(buf, size, n, 192, [_q6k(rng) for _ in range(POOL)])
        _stamp(buf, size, n, 208, blocks.fp16(d))
    elif ttype == 23:
        _stamp(buf, size, n, 0, blocks.fp16(d) + struct.pack("<H", 0xAAAA))
        _cycle(buf, size, n, 4, [_iq4xs(rng) for _ in range(POOL)])
    else:
        _stamp(buf, size, n, 0, blocks.fp16(d))
    return bytes(buf)


def bf16_bytes(rng: random.Random, n: int, sigma: float) -> bytes:
    dist = statistics.NormalDist(0.0, sigma)
    high = bytes(blocks.float_bytes(blocks.BF16, [dist.inv_cdf((i + 0.5) / 256)])[1] for i in range(256))
    buf = _randbytes(rng, 2 * n)
    buf[1::2] = bytes(buf[1::2]).translate(high)
    return bytes(buf)
