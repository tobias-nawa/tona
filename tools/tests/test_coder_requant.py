"""Logic of tools/coder_requant: router index match, expert slicing on a mini GGUF, selection, override list."""

import json
import math
import struct

import pytest
from coder_requant import gguf_layout, mtp, pipeline, router_match, select_experts, slicing, variants
from mini_model.gguf_io import KV, STR, U32

N_COLS = 4
N_EXP = 6


def kv_str(key: str, value: str) -> KV:
    raw = value.encode()
    return KV(key, STR, -1, struct.pack("<Q", len(raw)) + raw)


def kv_u32(key: str, value: int) -> KV:
    return KV(key, U32, -1, struct.pack("<I", value))


def write_model(
    path, n_exp: int = N_EXP, router_type: int = 0, extra=(), ple: bool = False, layers: int = 2
) -> dict[str, bytes]:
    """A mini model: per layer an F32 router [N_COLS, n_exp], a Q8_0 expert tensor [32, 1, n_exp] whose expert e
    carries the byte e + 16 * layer, and an F32 norm; returns the tensor bytes by name."""
    kvs = [kv_str("general.architecture", "mini"), kv_u32("mini.expert_count", n_exp), *extra]
    tensors, data = [], {}
    for layer in range(layers):
        elem = 4 if router_type == 0 else 2
        router = bytes((layer * 64 + e * N_COLS + c) % 251 for e in range(n_exp) for c in range(N_COLS * elem))
        experts = b"".join(bytes([e + 16 * layer]) * 34 for e in range(n_exp))
        for name, dims, ttype, raw in (
            (f"blk.{layer}.ffn_gate_inp.weight", [N_COLS, n_exp], router_type, router),
            (f"blk.{layer}.ffn_gate_exps.weight", [32, 1, n_exp], 8, experts),
            (f"blk.{layer}.attn_norm.weight", [N_COLS], 0, bytes(range(layer, layer + 16))),
        ):
            tensors.append(slicing.OutTensor(name, dims, ttype, [slicing.Chunk(name, 0, len(raw))]))
            data[name] = raw

    if ple:  # the per-layer embedding table: a tensor without an expert dimension
        raw = bytes(range(34))
        tensors.append(slicing.OutTensor(pipeline.PLE_TABLE, [32, 1], 8, [slicing.Chunk(pipeline.PLE_TABLE, 0, 34)]))
        data[pipeline.PLE_TABLE] = raw

    for name, raw in data.items():
        (path.parent / f"src-{name}").write_bytes(raw)
    slicing.write(str(path), kvs, tensors, slicing.Hooks(path=lambda name: str(path.parent / f"src-{name}")))
    return data


def layout_of(path):
    return gguf_layout.parse(path.read_bytes())


def tensor(path, name):
    lay = layout_of(path)
    t = lay.by_name[name]
    return path.read_bytes()[lay.data_start + t.offset :][: t.nbytes]


def test_layout_roundtrip_and_truncation(tmp_path):
    p = tmp_path / "m.gguf"
    data = write_model(p)
    lay = layout_of(p)
    assert gguf_layout.expert_count(lay) == N_EXP and lay.data_start % 32 == 0
    for name, raw in data.items():
        assert tensor(p, name) == raw
    with pytest.raises(gguf_layout.Truncated):
        gguf_layout.parse(p.read_bytes()[:40])


def test_slice_keeps_chosen_experts_in_order(tmp_path):
    src, dst = tmp_path / "in.gguf", tmp_path / "out.gguf"
    data = write_model(src)
    keep = {0: [1, 2, 5], 1: [5, 0, 3]}
    pipeline.slice_file(str(src), str(dst), keep)
    lay = layout_of(dst)
    assert gguf_layout.expert_count(lay) == 3
    for layer, k in keep.items():
        ex = tensor(dst, f"blk.{layer}.ffn_gate_exps.weight")
        assert [ex[i * 34] for i in range(3)] == [e + 16 * layer for e in k]
        router = data[f"blk.{layer}.ffn_gate_inp.weight"]
        per = 4 * N_COLS
        assert tensor(dst, f"blk.{layer}.ffn_gate_inp.weight") == b"".join(router[e * per : (e + 1) * per] for e in k)
        assert lay.by_name[f"blk.{layer}.ffn_gate_exps.weight"].dims == [32, 1, 3]
        assert tensor(dst, f"blk.{layer}.attn_norm.weight") == data[f"blk.{layer}.attn_norm.weight"]


def test_slice_merges_consecutive_ranges_and_rejects_bad_keep(tmp_path):
    src = tmp_path / "in.gguf"
    write_model(src)
    lay = layout_of(src)
    assert len(slicing.sliced(lay, "s", "blk.0.ffn_gate_exps.weight", [0, 1, 2, 4]).chunks) == 2
    for bad in ([], [0, 0], [N_EXP]):
        with pytest.raises(ValueError):
            slicing.sliced(lay, "s", "blk.0.ffn_gate_exps.weight", bad)
    with pytest.raises(ValueError, match="unequal"):
        slicing.equal_count({0: [0, 1], 1: [0]})


def test_assemble_and_check(tmp_path):
    coder, quant, final = tmp_path / "coder.gguf", tmp_path / "q.gguf", tmp_path / "final.gguf"
    write_model(coder, router_type=30)  # BF16 router as in the Coder
    keep = {0: [0, 3, 4], 1: [1, 2, 5]}
    pipeline.slice_file(str(coder), str(quant), keep)  # stands in for the requantized experts
    pipeline.assemble(str(coder), str(quant), keep, str(final))
    pipeline.check(str(coder), str(final), keep)
    assert tensor(final, "blk.1.ffn_gate_exps.weight") == tensor(quant, "blk.1.ffn_gate_exps.weight")
    with pytest.raises(SystemExit, match="rows"):
        pipeline.check(str(coder), str(final), {0: [0, 3, 5], 1: [1, 2, 5]})


def test_split_keys_dropped():
    kvs = [kv_u32("split.count", 6), kv_str("general.architecture", "mini")]
    assert [k.key for k in slicing.drop_split_kvs(kvs)] == ["general.architecture"]


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def test_bf16_round_to_nearest_even():
    assert router_match.bf16_round(f32_bits(1.0)) == 0x3F80
    assert router_match.bf16_round(0x3F808000) == 0x3F80  # tie, even stays
    assert router_match.bf16_round(0x3F818000) == 0x3F82  # tie, odd rounds up
    assert router_match.bf16_round(0x3F808001) == 0x3F81


def test_match_layer_finds_indices_and_rejects_ambiguity():
    n_cols = router_match.KEY_COLS + 4
    base_rows = [[math.sin(i * 7 + c) + i for c in range(n_cols)] for i in range(8)]
    base = b"".join(struct.pack(f"<{n_cols}f", *r) for r in base_rows)

    def coder(idx):
        return b"".join(
            struct.pack(f"<{n_cols}H", *(router_match.bf16_round(f32_bits(v)) for v in base_rows[i])) for i in idx
        )

    assert router_match.match_layer(coder([1, 4, 6]), base, n_cols) == [1, 4, 6]
    dup = base + struct.pack(f"<{n_cols}f", *base_rows[4])
    with pytest.raises(ValueError, match="2 matching"):
        router_match.match_layer(coder([4]), dup, n_cols)
    tail_differs = [*base_rows[2][:-1], base_rows[2][-1] + 1]
    with pytest.raises(ValueError, match="0 matching"):
        router_match.match_layer(
            struct.pack(f"<{n_cols}H", *(router_match.bf16_round(f32_bits(v)) for v in tail_differs)), base, n_cols
        )


def test_gate_weights_softmax_over_top_k():
    row = [0.0] * 12
    row[3], row[7] = 2.0, 1.0
    w = select_experts.gate_weights(row, k=2)
    assert set(w) == {3, 7} and math.isclose(w[3], math.e / (math.e + 1)) and math.isclose(sum(w.values()), 1.0)


def test_select_from_trace_and_profile(tmp_path):
    trace = tmp_path / "t.bin"
    rows = []
    for layer in range(2):
        tok = [0.0] * 256
        for rank, e in enumerate([5, 9, 1, 2, 3, 4, 6, 7, 8, 10]):
            tok[e] = 20.0 - rank
        rows.append(bytes([layer, 1]) + struct.pack("<256f", *tok) + struct.pack("<256f", *([0.0] * 256)))
    trace.write_bytes(b"".join(rows))
    score = select_experts.scores_from_trace([str(trace)], 2, 256)
    assert math.isclose(sum(score[0]), 1.0) and select_experts.ranked(score[0])[:2] == [5, 9]
    keep = select_experts.keep_equal(score, 3)
    assert keep == [[1, 5, 9], [1, 5, 9]]
    prof = tmp_path / "p.bin"
    prof.write_bytes(struct.pack("<3I", select_experts.PROFILE_MAGIC, 1, 4) + struct.pack("<4I", 3, 9, 0, 9))
    assert select_experts.keep_equal(select_experts.scores_from_profile(str(prof)), 2) == [[1, 3]]
    out = tmp_path / "keep.json"
    select_experts.write_keep(str(out), keep, score, "test")
    assert select_experts.read_keep(str(out)) == {0: [1, 5, 9], 1: [1, 5, 9]}
    assert json.loads(out.read_text())["dropped_mass"][0] < 0.5


def test_keep_adaptive_same_total_and_floor():
    score = [[float(256 - e) for e in range(256)], [1.0] * 10 + [0.0] * 246]
    keep = select_experts.keep_adaptive(score, 20, 12)
    assert sum(map(len, keep)) == 40 and min(map(len, keep)) >= 12
    assert len(keep[1]) < len(keep[0])  # layer 1 holds its mass in 10 experts


def test_variant_override_lines_and_command():
    v = variants.VARIANTS["A"]
    assert variants.tensor_types(v).splitlines() == [
        "ffn_gate_exps=iq2_xxs", "ffn_up_exps=iq2_xxs", "ffn_down_exps=q2_0",
    ]  # fmt: skip
    assert variants.tensor_types(v, "iq4_nl").splitlines()[-1] == "per_layer_token_embd=iq4_nl"
    cmd = variants.quantize_cmd("llama-quantize", "im.gguf", "t.txt", "in.gguf", "out.gguf")
    assert cmd[:3] == ["nice", "-n", "10"] and cmd[-2:] == ["Q8_0", "4"] and "--allow-requantize" in cmd
    assert {v.n_expert for v in variants.VARIANTS.values()} == {256, 216, 192}


def test_https_only_refuses_plain_http_and_redirects():
    import urllib.request

    import pytest
    from coder_requant import sources

    with pytest.raises(OSError):
        sources.open_https(urllib.request.Request("http://example.invalid/x"), 1)
    handler = sources.HttpsOnly()
    req = urllib.request.Request("https://example.invalid/x")
    with pytest.raises(OSError):
        handler.redirect_request(req, None, 302, "Found", {}, "http://example.invalid/y")


def test_check_reports_split_and_rejects_changed_metadata(tmp_path, capsys):
    split = [kv_u32("split.no", 0), kv_u32("split.count", 2), kv_u32("split.tensors.count", 9)]
    coder, quant, final = tmp_path / "c.gguf", tmp_path / "q.gguf", tmp_path / "f.gguf"
    write_model(coder, extra=split)
    keep = {0: [0, 3], 1: [1, 2]}
    pipeline.slice_file(str(coder), str(quant), keep)
    pipeline.assemble(str(coder), str(quant), keep, str(final))
    pipeline.check(str(coder), str(final), keep)
    assert "part 1 of 2" in capsys.readouterr().err
    plain = tmp_path / "plain.gguf"  # same model without the split keys: a final file that lost them
    write_model(plain)
    pipeline.slice_file(str(plain), str(quant), keep)
    pipeline.assemble(str(plain), str(quant), keep, str(plain.with_name("lost.gguf")))
    with pytest.raises(SystemExit, match=r"split\.no is None"):
        pipeline.check(str(coder), str(plain.with_name("lost.gguf")), keep)


def test_drop_ple_leaves_the_table_out_of_the_quantize_input(tmp_path):
    src, kept, dropped = tmp_path / "s.gguf", tmp_path / "k.gguf", tmp_path / "d.gguf"
    write_model(src, ple=True)
    keep = {0: [0, 3], 1: [1, 2]}
    pipeline.slice_file(str(src), str(kept), keep)
    pipeline.slice_file(str(src), str(dropped), keep, drop_ple=True)
    assert pipeline.PLE_TABLE in layout_of(kept).by_name
    assert pipeline.PLE_TABLE not in layout_of(dropped).by_name
    assert tensor(dropped, "blk.1.ffn_gate_exps.weight") == tensor(kept, "blk.1.ffn_gate_exps.weight")


def test_slice_mtp_cuts_the_single_expert_layer(tmp_path):
    src, dst, keep_file = tmp_path / "mtp.gguf", tmp_path / "cut.gguf", tmp_path / "keep.json"
    data = write_model(src, layers=1)
    keep_file.write_text(json.dumps({"keep": [[4, 1, 2]]}))  # unsorted on purpose: the cut sorts
    pipeline.main(["slice-mtp", "--input", str(src), "--keep-list", str(keep_file), "--out", str(dst)])
    lay = layout_of(dst)
    assert gguf_layout.expert_count(lay) == 3 and lay.by_name["blk.0.ffn_gate_exps.weight"].dims == [32, 1, 3]
    src_experts = data["blk.0.ffn_gate_exps.weight"]
    assert tensor(dst, "blk.0.ffn_gate_exps.weight") == b"".join(src_experts[e * 34 : (e + 1) * 34] for e in (1, 2, 4))
    assert tensor(dst, "blk.0.attn_norm.weight") == data["blk.0.attn_norm.weight"]


def test_mtp_keep_list_rejects_bad_input(tmp_path):
    src = tmp_path / "mtp.gguf"
    write_model(src, layers=1)
    lay = layout_of(src)
    for bad in ([], [1, 1], [-1, 2], [0, N_EXP]):
        with pytest.raises(ValueError):
            mtp.keep_for(lay, sorted(bad))
    two = tmp_path / "two.gguf"
    write_model(two)
    with pytest.raises(ValueError, match="one layer"):
        mtp.keep_for(layout_of(two), [0, 1])
    flat = tmp_path / "flat.json"
    flat.write_text("[5, 2]")
    assert mtp.read_keep_list(str(flat)) == [2, 5]
    flat.write_text(json.dumps({"keep": [[0], [1]]}))
    with pytest.raises(ValueError, match="one layer"):
        mtp.read_keep_list(str(flat))
