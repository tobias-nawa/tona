"""tools/coder_requant/repack.py: IQ3_XXS experts to TONA_IQ3_XXS_R on a copy of the file."""

import struct

import pytest
from coder_requant import gguf_layout, pipeline, repack, slicing
from mini_model.gguf_io import KV, STR

GATE, NORM, ATTN = "blk.0.ffn_gate_exps.weight", "blk.0.attn_norm.weight", "blk.0.attn_q.weight"


def block(seed: int) -> bytes:
    return bytes((seed * 31 + i * 7) % 256 for i in range(repack.BLOCK))


def write_model(path, row: int = 512) -> dict[str, bytes]:
    """Two experts of one IQ3_XXS row each, an IQ3_XXS attention matrix and an F32 norm."""
    n_blocks = row // 256
    raw = {
        GATE: b"".join(block(e * 10 + b) for e in range(2) for b in range(n_blocks)),
        ATTN: b"".join(block(90 + b) for b in range(n_blocks)),
        NORM: bytes(range(16)),
    }
    shapes = {GATE: ([row, 1, 2], 18), ATTN: ([row, 1], 18), NORM: ([4], 0)}
    arch = b"mini"
    kvs = [KV("general.architecture", STR, -1, struct.pack("<Q", len(arch)) + arch)]
    tensors = [slicing.OutTensor(n, *shapes[n], [slicing.Chunk(n, 0, len(r))]) for n, r in raw.items()]
    for n, r in raw.items():
        (path.parent / f"src-{n}").write_bytes(r)
    slicing.write(str(path), kvs, tensors, slicing.Hooks(path=lambda n: str(path.parent / f"src-{n}")))
    return raw


def data_of(path, name):
    lay = gguf_layout.parse(path.read_bytes())
    t = lay.by_name[name]
    return t.type, path.read_bytes()[lay.data_start + t.offset :][: t.nbytes]


def test_repack_blocks_moves_fields():
    a, b = block(1), block(2)
    sb = repack.repack_blocks(a + b)
    assert sb == a[2:66] + b[2:66] + a[66:98] + b[66:98] + a[0:2] + b[0:2]
    with pytest.raises(ValueError):
        repack.repack_blocks(a)


def test_repack_command_changes_only_the_experts(tmp_path):
    src, out = tmp_path / "in.gguf", tmp_path / "out.gguf"
    raw = write_model(src)
    pipeline.main(["repack", "--input", str(src), "--out", str(out)])
    assert data_of(out, GATE) == (1018, repack.repack_blocks(raw[GATE]))
    assert data_of(out, ATTN) == (18, raw[ATTN]) and data_of(out, NORM) == (0, raw[NORM])
    assert data_of(src, GATE) == (18, raw[GATE])  # the input stays as it was
    assert out.stat().st_size == src.stat().st_size
    with pytest.raises(SystemExit):  # never overwrites
        pipeline.main(["repack", "--input", str(src), "--out", str(out)])


def test_rows_without_whole_super_blocks_are_refused(tmp_path):
    src = tmp_path / "in.gguf"
    write_model(src, row=256)
    with pytest.raises(ValueError, match="super-blocks"):
        repack.repack_file(str(src))
