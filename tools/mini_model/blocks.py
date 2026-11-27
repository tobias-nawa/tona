"""Random tensor data in the quantized layouts of the engine (engine/model/quant.cpp, tools/gguf_info.py).

Quant payloads are random bytes; only the scale fields are set, so that the weights have a known size and no
block carries a constant offset (random bytes in a Q4_K block would give all-positive weights).
"""

import math
import random
import struct

# type -> (name, elements per block, bytes per block)
TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 8: ("Q8_0", 32, 34), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 16: ("IQ2_XXS", 256, 66), 18: ("IQ3_XXS", 256, 98), 20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 30: ("BF16", 1, 2),
    40: ("NVFP4", 64, 36), 42: ("Q2_0", 64, 18),
}  # fmt: skip
F32, F16, BF16 = 0, 1, 30

# Root mean square of one dequantized weight at scale d = 1, measured with the engine's dequantizer on 512 blocks
# made by this module (research/measurements/2026-11-02-mini-model.md). The generator divides the wanted rms by it.
RMS_AT_UNIT_SCALE = {
    8: 74.2,
    42: 1.22,
    12: 118.0,
    13: 239.0,
    14: 156.0,
    16: 53.4,  # computed from the grid: sqrt(E[g^2] = 534.4 * E[((0.5 + ls) / 4)^2] = 5.33, ls uniform 0..15)
    18: 143.0,
    20: 67.7,
    21: 142.0,
    22: 56.9,
    23: 566.0,
}


# NVFP4: the E2M1 codes doubled (0, 1, 2, 3, 4, 6, 8, 12, signed) have an rms of sqrt(34.25) with random nibbles;
# a weight is code * ue4m3_half(scale byte), and the scale byte is the same in all four sub-blocks of a block
NVFP4_CODE_RMS = math.sqrt(34.25)


def ue4m3_half(x: int) -> float:
    """The scale byte as engine/model/quant.cpp reads it (UE4M3, halved): 0 and 0x7F give 0."""
    if x in (0, 0x7F):
        return 0.0
    exp, man = (x >> 3) & 0xF, x & 7
    return (math.ldexp(man, -9) if exp == 0 else math.ldexp(1.0 + man / 8.0, exp - 7)) * 0.5


def nvfp4_scale_byte(rms: float) -> int:
    """The scale byte whose weights come closest to the wanted rms."""
    want = rms / NVFP4_CODE_RMS
    return min(range(1, 0x7F), key=lambda x: abs(ue4m3_half(x) - want))


def nbytes(ttype: int, elements: int) -> int:
    _, per_block, block_bytes = TYPES[ttype]
    return elements // per_block * block_bytes


def fp16(x: float) -> bytes:
    return struct.pack("<e", x)


def seeded(seed: int, name: str) -> random.Random:
    """One stream per tensor, so the bytes do not depend on the order of the tensors."""
    return random.Random(f"{seed}/{name}")


def _pack_k4(sc: list[int], mn: list[int]) -> bytes:
    """Inverse of get_scale_min_k4: eight 6-bit scales and eight 6-bit mins in 12 bytes."""
    q = bytearray(12)
    for j in range(4):
        q[j] = sc[j] | ((sc[j + 4] >> 4) << 6)
        q[j + 4] = mn[j] | ((mn[j + 4] >> 4) << 6)
        q[j + 8] = (sc[j + 4] & 15) | ((mn[j + 4] & 15) << 4)
    return bytes(q)


def _fix_kquant(buf: bytearray, o: int, rng: random.Random, d: float, mid: float) -> None:
    # w = d * sc * (q - mid): min = mid * scale makes the block mean zero
    sc = [rng.randint(8, 40) for _ in range(8)]
    buf[o : o + 4] = fp16(d) + fp16(mid * d)
    buf[o + 4 : o + 16] = _pack_k4(sc, sc)


def _fix_q6k(buf: bytearray, o: int, rng: random.Random, d: float) -> None:
    buf[o + 192 : o + 208] = bytes(rng.choice((-1, 1)) * rng.randint(4, 12) & 255 for _ in range(16))
    buf[o + 208 : o + 210] = fp16(d)


def _fix_iq4xs(buf: bytearray, o: int, rng: random.Random, d: float) -> None:
    # scale (ls - 32) in 4..12: ls = 36..44 has the high bits 0b10
    ls = [rng.randint(36, 44) for _ in range(8)]
    buf[o : o + 2] = fp16(d)
    buf[o + 2 : o + 4] = struct.pack("<H", 0xAAAA)
    for i in range(4):
        buf[o + 4 + i] = (ls[2 * i] & 15) | ((ls[2 * i + 1] & 15) << 4)


def quant_bytes(ttype: int, elements: int, rng: random.Random, rms: float) -> bytes:
    """Blocks with random quants whose dequantized values have about the given rms."""
    _, per_block, size = TYPES[ttype]
    n = elements // per_block
    buf = bytearray(rng.randbytes(n * size))
    d = rms / RMS_AT_UNIT_SCALE[ttype] if ttype != 40 else 0.0
    for i in range(n):
        o = i * size
        if ttype == 12:
            _fix_kquant(buf, o, rng, d, 7.5)
        elif ttype == 13:
            _fix_kquant(buf, o, rng, d, 15.5)
        elif ttype == 14:
            _fix_q6k(buf, o, rng, d)
        elif ttype == 23:
            _fix_iq4xs(buf, o, rng, d)
        elif ttype == 40:
            buf[o : o + 4] = bytes([nvfp4_scale_byte(rms)]) * 4
        else:
            buf[o : o + 2] = fp16(d)
    return bytes(buf)


def _bf16(x: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", x))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)  # round to nearest even
    return struct.pack("<H", (bits >> 16) & 0xFFFF)


def float_bytes(ttype: int, values: list[float]) -> bytes:
    if ttype == F32:
        return struct.pack(f"<{len(values)}f", *values)
    if ttype == F16:
        return struct.pack(f"<{len(values)}e", *values)
    return b"".join(_bf16(v) for v in values)


def gaussian(rng: random.Random, n: int, sigma: float) -> list[float]:
    return [rng.gauss(0.0, sigma) for _ in range(n)]


def uniform(rng: random.Random, n: int, lo: float, hi: float) -> list[float]:
    return [lo + (hi - lo) * rng.random() for _ in range(n)]


def sigma_of(fan_in: int, gain: float) -> float:
    return gain / math.sqrt(fan_in)
