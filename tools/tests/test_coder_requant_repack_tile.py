"""tools/coder_requant/repack_tile.py: Q6_K layer matrices to TONA_Q6_K_T, and the tensors it must refuse."""

import random
import struct

import pytest
from coder_requant import gguf_layout, pipeline, repack_tile, slicing
from mini_model.gguf_io import KV, STR

ATTN, OUT, NORM = "blk.0.attn_q.weight", "output.weight", "blk.0.attn_norm.weight"


def q6k_rows(rows: int, k: int, seed: int) -> bytes:
    rng = random.Random(seed)
    out = bytearray()
    for _ in range(rows * k // 256):
        out += bytes(rng.randrange(256) for _ in range(208)) + struct.pack("<e", rng.choice([0.5, 0.25, 2.0]))
    return bytes(out)


def ggml_weight(row: bytes, w: int) -> tuple[int, int, bytes]:
    """Code, scale byte and d of weight w of a Q6_K row, read as ggml's dequantize_row_q6_K does."""
    b = row[(w // 256) * 210 :][:210]
    j = w % 256
    n, low, quarter = j // 128, j % 32, (j % 128) // 32
    ql, qh = b[64 * n + low + 32 * (quarter & 1)], b[128 + 32 * n + low]
    code = ((ql >> 4) if quarter >= 2 else (ql & 0xF)) | (((qh >> (2 * quarter)) & 3) << 4)
    return code, b[192 + j // 16], b[208:210]


def tile_weight(tile: bytes, k: int, r8: int, w: int) -> tuple[int, int, bytes]:
    """The same from a tile, following docs/use/tona-types.md literally."""
    c, q, i = w // 32, (w % 32) // 8, w % 8
    lane = next(n for n in range(32) if 4 * (n >> 4 & 1) + (n >> 1 & 3) == r8 and 2 * (n >> 3 & 1) + (n & 1) == q)
    e = 32 * c + lane
    ql = int.from_bytes(tile[4 * e :][:4], "little")
    qh = int.from_bytes(tile[4 * k + 2 * e :][:2], "little")
    d_at = 6 * k + k // 2 + 2 * (8 * (w // 256) + r8)
    return (ql >> 4 * i & 0xF) | ((qh >> 2 * i & 3) << 4), tile[6 * k + e // 2], tile[d_at : d_at + 2]


def test_tiles_hold_every_weight_of_the_q6k_rows():
    rows, k = 16, 512
    data = q6k_rows(rows, k, 3)
    tiles = repack_tile.tile_rows(data, rows, k)
    assert len(tiles) == len(data)
    rb = k // 256 * 210
    for r in range(rows):
        tile = tiles[(r // 8) * 8 * rb :][: 8 * rb]
        for w in range(k):
            assert tile_weight(tile, k, r % 8, w) == ggml_weight(data[r * rb :][:rb], w), (r, w)


def write_model(path, shapes: dict) -> dict[str, bytes]:
    raw = {n: (q6k_rows(s[1], s[0], i) if t == 14 else bytes(16)) for i, (n, (s, t)) in enumerate(shapes.items())}
    arch = b"mini"
    kvs = [KV("general.architecture", STR, -1, struct.pack("<Q", len(arch)) + arch)]
    tensors = [slicing.OutTensor(n, s, t, [slicing.Chunk(n, 0, len(raw[n]))]) for n, (s, t) in shapes.items()]
    for n, r in raw.items():
        (path.parent / f"src-{n}").write_bytes(r)
    slicing.write(str(path), kvs, tensors, slicing.Hooks(path=lambda n: str(path.parent / f"src-{n}")))
    return raw


def data_of(path, name):
    lay = gguf_layout.parse(path.read_bytes())
    t = lay.by_name[name]
    return t.type, path.read_bytes()[lay.data_start + t.offset :][: t.nbytes]


def test_tile_command_changes_only_layer_matrices(tmp_path):
    src, out = tmp_path / "in.gguf", tmp_path / "out.gguf"
    raw = write_model(src, {ATTN: ([512, 8], 14), OUT: ([256, 8], 14), NORM: ([4], 0)})
    pipeline.main(["repack", "--tile", "--input", str(src), "--out", str(out)])
    assert data_of(out, ATTN) == (1019, repack_tile.tile_rows(raw[ATTN], 8, 512))
    assert data_of(out, OUT) == (14, raw[OUT]) and data_of(out, NORM) == (0, raw[NORM])
    assert data_of(src, ATTN) == (14, raw[ATTN])


@pytest.mark.parametrize(
    "name,shape",
    [("output.weight", [256, 8]), ("token_embd.weight", [256, 8]), ("per_layer_token_embd.weight", [256, 8]),
     (ATTN, [256, 4]), ("blk.0.ffn_down_exps.weight", [256, 8, 2])],
)  # fmt: skip
def test_tensors_read_by_row_are_refused(tmp_path, name, shape):
    src = tmp_path / "in.gguf"
    write_model(src, {name: (shape, 14)})
    with pytest.raises(ValueError, match="read by row"):
        repack_tile.tile_file(str(src), pattern=".")
