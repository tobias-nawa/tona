import collections
import hashlib
from pathlib import Path

import gguf_info
import make_mini_model
import pytest
from mini_model import gguf_io, nemotron_h, qwen35moe, shapes, tensors

ROOT = Path(__file__).resolve().parents[2]
REAL = make_mini_model.DEFAULT_SOURCE
SMALL = shapes.Shape(vocab=64, n_layer=4, n_expert=4, expert_ff=64)


def fake_source(path: Path, mtp: bool) -> str:
    """A source file with the key-value types of the real one and every tensor role, without data."""
    s = SMALL
    types = collections.defaultdict(lambda: [0])
    planned = tensors.mtp_tensors(s, types, 1, 8) if mtp else tensors.target_tensors(s, types, 1)
    kvs = [gguf_io.KV("general.architecture", gguf_io.STR, -1, gguf_io.encode_scalar(gguf_io.STR, "qwen4exp"))]
    kvs.append(gguf_io.KV("tokenizer.ggml.model", gguf_io.STR, -1, gguf_io.encode_scalar(gguf_io.STR, "gpt2")))
    for key, value in shapes.overrides(s, s.n_layer + mtp, mtp).items():
        if isinstance(value, list):
            raw = gguf_io.encode_value(gguf_io.KV(key, gguf_io.ARR, gguf_io.I32, b""), value)
            kvs.append(gguf_io.KV(key, gguf_io.ARR, gguf_io.I32, raw))
        elif isinstance(value, str):
            kvs.append(gguf_io.KV(key, gguf_io.STR, -1, gguf_io.encode_scalar(gguf_io.STR, value)))
        else:
            kvs.append(gguf_io.KV(key, gguf_io.U32, -1, gguf_io.encode_scalar(gguf_io.U32, value)))
    empty = [gguf_io.OutTensor(t.name, t.dims, 0, 0, lambda: b"") for t in planned]
    gguf_io.write_gguf(str(path), kvs, empty)
    return str(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("mtp", [False, True])
def test_same_seed_same_bytes(tmp_path, mtp):
    src = fake_source(tmp_path / "src.gguf", mtp)
    a, b, c = (tmp_path / n for n in ("a.gguf", "b.gguf", "c.gguf"))
    make_mini_model.write_model(str(a), src, mtp, SMALL, 1)
    make_mini_model.write_model(str(b), src, mtp, SMALL, 1)
    make_mini_model.write_model(str(c), src, mtp, SMALL, 2)
    assert digest(a) == digest(b)
    assert digest(a) != digest(c)


def test_header_is_readable(tmp_path):
    src = fake_source(tmp_path / "src.gguf", False)
    out = tmp_path / "mini.gguf"
    make_mini_model.write_model(str(out), src, False, SMALL, 1)
    g = gguf_info.read_gguf(str(out))
    assert g["kv"]["qwen4exp.block_count"] == 4
    assert g["kv"]["qwen4exp.embedding_length"] == 256
    assert g["kv"]["qwen4exp.attention.compress_ratios"] == [0, 0, 0, 4]
    by_name = {t["name"]: t for t in g["tensors"]}
    assert by_name["blk.0.ffn_gate_exps.weight"]["dims"] == [256, 64, 4]
    assert by_name["blk.3.attn_q.weight"]["dims"] == [256, 2 * 24 * 256]
    assert "blk.1.ple_key.weight" in by_name and "blk.2.ple_key.weight" not in by_name


def test_mtp_has_one_block_after_the_target(tmp_path):
    src = fake_source(tmp_path / "src.gguf", True)
    out = tmp_path / "mtp.gguf"
    make_mini_model.write_model(str(out), src, True, SMALL, 1)
    g = gguf_info.read_gguf(str(out))
    assert g["kv"]["qwen4exp.block_count"] == 5
    assert all(t["name"].startswith("blk.4.") for t in g["tensors"])


def test_float_and_quant_data_sizes():
    from mini_model import blocks

    rng = blocks.seeded(1, "x")
    for ttype in (8, 12, 13, 14, 18, 20, 21, 22, 23, 42):
        n = blocks.TYPES[ttype][1] * 3
        assert len(blocks.quant_bytes(ttype, n, rng, 0.05)) == blocks.nbytes(ttype, n)


@pytest.mark.skipif(not Path(REAL).exists(), reason="needs the real model's header")
def test_every_type_of_every_role_appears(tmp_path):
    out = tmp_path / "mini.gguf"
    full = shapes.Shape(vocab=64, n_expert=4, expert_ff=64)  # all 12 layers, small experts
    make_mini_model.write_model(str(out), REAL, False, full, 1)
    real = tensors.role_types(gguf_io.read_header(REAL)[1])
    mini = tensors.role_types(gguf_io.read_header(str(out))[1])
    for role in ("ffn_gate_exps.weight", "ffn_gate_shexp.weight", "attn_gate.weight"):
        assert set(mini[role]) == set(real[role]), role


ORNITH = qwen35moe.Shape(vocab=64, n_layer=4, n_expert=8, n_used=2)


def fake_ornith_source(path: Path) -> str:
    """A source header with the keys the mini Ornith overrides; no tensors, so F32 everywhere."""
    kvs = [gguf_io.KV("general.architecture", gguf_io.STR, -1, gguf_io.encode_scalar(gguf_io.STR, "qwen35moe"))]
    for key, value in qwen35moe.overrides(ORNITH).items():
        kind = gguf_io.STR if isinstance(value, str) else gguf_io.U32
        kvs.append(gguf_io.KV(key, kind, -1, gguf_io.encode_scalar(kind, value)))
    types = collections.defaultdict(lambda: [0])
    empty = [gguf_io.OutTensor(t.name, t.dims, 0, 0, lambda: b"") for t in qwen35moe.model_tensors(ORNITH, types, 1)]
    gguf_io.write_gguf(str(path), kvs, empty)
    return str(path)


def test_mini_ornith_layout(tmp_path):
    src = fake_ornith_source(tmp_path / "src.gguf")
    out = tmp_path / "mini-ornith.gguf"
    make_mini_model.write_ornith(str(out), src, ORNITH, 1)
    g = gguf_info.read_gguf(str(out))
    assert g["kv"]["qwen35moe.block_count"] == 5  # four trunk layers and the MTP layer
    by_name = {t["name"]: t for t in g["tensors"]}
    assert "blk.0.attn_qkv.weight" in by_name and "blk.0.attn_q.weight" not in by_name
    assert by_name["blk.3.attn_q.weight"]["dims"] == [256, 2 * 16 * 256]  # [q | gate] per head
    assert by_name["blk.0.ffn_gate_exps.weight"]["dims"] == [256, 256, 8]
    assert by_name["blk.4.nextn.eh_proj.weight"]["dims"] == [512, 256]
    assert "blk.4.nextn.shared_head_norm.weight" in by_name and "blk.4.attn_q.weight" in by_name
    assert not any("hc_" in n or "ple" in n or "indexer" in n for n in by_name)  # one stream, no PLE, no indexer


def test_mini_ornith_same_seed_same_bytes(tmp_path):
    src = fake_ornith_source(tmp_path / "src.gguf")
    a, b, c = (tmp_path / n for n in ("a.gguf", "b.gguf", "c.gguf"))
    make_mini_model.write_ornith(str(a), src, ORNITH, 1)
    make_mini_model.write_ornith(str(b), src, ORNITH, 1)
    make_mini_model.write_ornith(str(c), src, ORNITH, 2)
    assert digest(a) == digest(b)
    assert digest(a) != digest(c)


BOLT = nemotron_h.Shape(vocab=64)


def fake_bolt_source(path: Path) -> str:
    """A source header with the keys the mini Bolt overrides (arrays as I32 like the real file); F32 everywhere."""
    kvs = [gguf_io.KV("general.architecture", gguf_io.STR, -1, gguf_io.encode_scalar(gguf_io.STR, "nemotron_h_moe"))]
    for key, value in nemotron_h.overrides(BOLT).items():
        if isinstance(value, list):
            raw = gguf_io.encode_value(gguf_io.KV(key, gguf_io.ARR, gguf_io.I32, b""), value)
            kvs.append(gguf_io.KV(key, gguf_io.ARR, gguf_io.I32, raw))
        else:
            kind = gguf_io.STR if isinstance(value, str) else gguf_io.U32
            kvs.append(gguf_io.KV(key, kind, -1, gguf_io.encode_scalar(kind, value)))
    types = collections.defaultdict(lambda: [0])
    empty = [gguf_io.OutTensor(t.name, t.dims, 0, 0, lambda: b"") for t in nemotron_h.model_tensors(BOLT, types, 1)]
    gguf_io.write_gguf(str(path), kvs, empty)
    return str(path)


def test_mini_bolt_layout(tmp_path):
    src = fake_bolt_source(tmp_path / "src.gguf")
    out = tmp_path / "mini-bolt.gguf"
    make_mini_model.write_bolt(str(out), src, BOLT, 1)
    g = gguf_info.read_gguf(str(out))
    assert g["kv"]["nemotron_h_moe.block_count"] == 8
    assert g["kv"]["nemotron_h_moe.attention.head_count_kv"] == [0, 0, 0, 0, 2, 0, 0, 0]
    assert g["kv"]["nemotron_h_moe.expert_used_count"] == [0, 4, 0, 3, 0, 0, 5, 0]
    by_name = {t["name"]: t for t in g["tensors"]}
    assert by_name["blk.0.ssm_in.weight"]["dims"] == [256, 1544] and by_name["blk.0.ssm_in.scale"]["dims"] == [1]
    assert by_name["blk.0.ssm_norm.weight"]["dims"] == [256, 2]
    # uneven MoE blocks: own expert count and width, latent pair, no gate tensor
    assert by_name["blk.1.ffn_up_exps.weight"]["dims"] == [256, 256, 32]
    assert by_name["blk.3.ffn_down_exps.weight"]["dims"] == [512, 256, 24]
    assert by_name["blk.6.ffn_gate_inp.weight"]["dims"] == [256, 40]
    assert "blk.1.ffn_latent_down.weight" in by_name and not any("gate_exps" in n for n in by_name)
    assert "blk.4.attn_q.weight" in by_name and "blk.4.attn_q_norm.weight" not in by_name


def test_nvfp4_scale_byte_and_size():
    from mini_model import blocks

    assert blocks.ue4m3_half(0x38) == 0.5 and blocks.ue4m3_half(0x7F) == 0.0 and blocks.ue4m3_half(0x01) == 2.0**-10
    for rms in (0.1, 0.5, 3.0):  # the nearest of 126 values on a log scale is within 7 % of the wanted weight size
        got = blocks.ue4m3_half(blocks.nvfp4_scale_byte(rms)) * blocks.NVFP4_CODE_RMS
        assert abs(got / rms - 1) < 0.07
    rng = blocks.seeded(1, "x")
    data = blocks.quant_bytes(40, 64 * 3, rng, 0.05)
    assert len(data) == blocks.nbytes(40, 64 * 3) == 3 * 36
    assert data[0:4] == bytes([data[0]]) * 4 and data[36:40] == data[0:4]
