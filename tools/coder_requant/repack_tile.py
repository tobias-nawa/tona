"""Q6_K layer matrices to TONA_Q6_K_T (docs/use/tona-types.md), in place on a copy of a GGUF file.

The tile type interleaves 8 rows in simdgroup MMA fragment order, so a Metal lane reads its 8 weights as one 32-bit
word, one 16-bit word and one scale byte (research/measurements/2026-11-12-mac-dense-mma-probes.md). Sizes and offsets
stay; only the data of the chosen tensors and their type fields change. Rows are no longer contiguous, so the type is
refused for every tensor read by row: only 2-D matrices of a layer (blk.*) qualify.
"""

import re

import numpy as np

from coder_requant.gguf_layout import Layout, Tensor, parse
from coder_requant.repack import HEADER_READ

Q6_K, Q6_K_T = 14, 1019
LAYER_MATRICES = r"^blk\.\d+\.[a-z_]+\.weight$"  # not the MTP layer's nextn.* names, not token_embd/output/PLE
TILE_ROWS = 8

LANE = np.arange(32)
LANE_ROW = ((LANE >> 2) & 4) | ((LANE >> 1) & 3)  # mma_fm of the Metal kernels
LANE_GROUP = ((LANE >> 2) & 2) | (LANE & 1)  # mma_fn / 2


def q6k_codes(blocks: np.ndarray) -> np.ndarray:
    """6-bit codes of Q6_K blocks [..., 210] as [..., 256] (ggml's ql/qh layout)."""
    j = np.arange(256)
    n, low, quarter = j // 128, j % 32, (j % 128) // 32
    ql = blocks[..., 64 * n + low + 32 * (quarter & 1)]
    qh = blocks[..., 128 + 32 * n + low]
    return ((ql >> (4 * (quarter >= 2))) & 0xF) | (((qh >> (2 * quarter)) & 3) << 4)


def tile_rows(data: bytes, rows: int, k: int) -> bytes:
    """rows (a multiple of 8) Q6_K rows of k weights as TONA_Q6_K_T tiles."""
    sb = k // 256
    q6 = np.frombuffer(data, np.uint8).reshape(rows, sb, 210)
    t, c = rows // TILE_ROWS, k // 32
    codes = q6k_codes(q6).reshape(t, TILE_ROWS, c, 4, 8).astype(np.uint32)
    entry = codes[:, LANE_ROW, :, LANE_GROUP, :].transpose(1, 2, 0, 3)  # [tile, chunk, lane, weight]
    shift = np.arange(8, dtype=np.uint32)
    ql = ((entry & 0xF) << (4 * shift)).sum(axis=3, dtype=np.uint32).astype("<u4")
    qh = (((entry >> 4) & 3) << (2 * shift)).sum(axis=3, dtype=np.uint32).astype("<u2")
    scales = q6[..., 192:208].reshape(t, TILE_ROWS, c, 2)
    sc = scales[:, LANE_ROW[::2], :, LANE_GROUP[::2] // 2].transpose(1, 2, 0)  # entry pairs e, e + 1
    d = q6[..., 208:210].reshape(t, TILE_ROWS, sb, 2).transpose(0, 2, 1, 3)
    parts = [np.ascontiguousarray(p).view(np.uint8) for p in (ql, qh, sc, d)]
    return np.concatenate([p.reshape(t, -1) for p in parts], axis=1).tobytes()


def chosen(layout: Layout, pattern: str = LAYER_MATRICES) -> list[Tensor]:
    """Q6_K tensors whose name matches; a match the type cannot hold is an error, not a silent skip."""
    out = []
    for t in layout.tensors:
        if t.type != Q6_K or not re.search(pattern, t.name):
            continue
        if not t.name.startswith("blk.") or len(t.dims) != 2 or t.dims[1] % TILE_ROWS or t.dims[0] % 256:
            raise ValueError(
                f"{t.name}: TONA_Q6_K_T only for 2-D layer matrices with rows a multiple of 8 and 256 weights; "
                "token_embd, output, per_layer_token_embd, the MTP tables and experts are read by row "
                "(docs/use/tona-types.md)"
            )
        out.append(t)
    return out


def tile_file(path: str, pattern: str = LAYER_MATRICES, tiles_per_chunk: int = 512) -> list[str]:
    """Tiles the chosen tensors of path in place; returns their names. Data first, then the type fields."""
    with open(path, "r+b") as f:
        layout = parse(f.read(HEADER_READ))
        tensors = chosen(layout, pattern)
        for t in tensors:
            k, rows = t.dims[0], t.dims[1]
            row_bytes = k // 256 * 210
            for r0 in range(0, rows, TILE_ROWS * tiles_per_chunk):
                n = min(TILE_ROWS * tiles_per_chunk, rows - r0)
                pos = layout.data_start + t.offset + r0 * row_bytes
                f.seek(pos)
                data = f.read(n * row_bytes)
                f.seek(pos)
                f.write(tile_rows(data, n, k))
        for t in tensors:
            f.seek(t.type_pos)
            f.write(Q6_K_T.to_bytes(4, "little"))
    return [t.name for t in tensors]
