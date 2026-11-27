"""Which base experts the Coder kept: match each Coder router row (BF16) to a base router row (F32).

The Coder is an expert selection of the base model with the router rows of the kept experts, so each Coder row
should equal one base row rounded to BF16. A row with no match or with more than one aborts: the slice would build
the wrong experts silently.
"""

import struct
from array import array

KEY_COLS = 16  # leading columns used as the lookup key before the full-row comparison


def bf16_round(f32_bits: int) -> int:
    """Round-to-nearest-even F32 -> BF16, as ggml_compute_fp32_to_bf16 (NaN kept quiet)."""
    if (f32_bits & 0x7FFFFFFF) > 0x7F800000:
        return (f32_bits >> 16) | 0x40
    return (f32_bits + 0x7FFF + ((f32_bits >> 16) & 1)) >> 16


def rows_f32_as_bf16(data: bytes, n_rows: int, n_cols: int, cols: int | None = None) -> list[tuple[int, ...]]:
    """BF16 bit patterns of each F32 row (only the first `cols` columns if given)."""
    bits = array("I")
    bits.frombytes(data)
    take = n_cols if cols is None else cols
    return [tuple(bf16_round(b) for b in bits[r * n_cols : r * n_cols + take]) for r in range(n_rows)]


def rows_bf16(data: bytes, n_rows: int, n_cols: int) -> list[tuple[int, ...]]:
    vals = struct.unpack(f"<{n_rows * n_cols}H", data)
    return [vals[r * n_cols : (r + 1) * n_cols] for r in range(n_rows)]


def match_layer(coder_bf16: bytes, base_f32: bytes, n_cols: int) -> list[int]:
    """For each Coder row j the base row index; raises ValueError on a missing or ambiguous match."""
    n_coder = len(coder_bf16) // (2 * n_cols)
    n_base = len(base_f32) // (4 * n_cols)
    coder = rows_bf16(coder_bf16, n_coder, n_cols)
    keys: dict[tuple[int, ...], list[int]] = {}
    for i, key in enumerate(rows_f32_as_bf16(base_f32, n_base, n_cols, KEY_COLS)):
        keys.setdefault(key, []).append(i)
    full: dict[int, tuple[int, ...]] = {}
    out = []
    for j, row in enumerate(coder):
        hits = []
        for i in keys.get(row[:KEY_COLS], []):
            if i not in full:
                full[i] = rows_f32_as_bf16(base_f32[i * 4 * n_cols : (i + 1) * 4 * n_cols], 1, n_cols)[0]
            if full[i] == row:
                hits.append(i)
        if len(hits) != 1:
            raise ValueError(f"Coder expert {j}: {len(hits)} matching base rows {hits[:4]}")
        out.append(hits[0])
    if len(set(out)) != len(out):
        raise ValueError("two Coder experts map to the same base expert")
    return out
