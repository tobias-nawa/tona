import hashlib
import json
import stat
import struct
from pathlib import Path

import make_mini_model
import mini_token_check
import pytest
from mini_model import gguf_io, headers
from test_make_mini_model import SMALL, fake_source

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "mini-models"


def with_dropped_key(src: str) -> str:
    """The fake source plus a notice of origin (kept in the header, dropped from the mini model) and a build path."""
    kvs, tensors = gguf_io.read_header(src)
    raw = gguf_io.encode_scalar(gguf_io.STR, "somebody")
    kvs.append(gguf_io.KV("general.quantized_by", gguf_io.STR, -1, raw))
    kvs.append(gguf_io.KV("quantize.imatrix.file", gguf_io.STR, -1, raw))
    Path(src).write_bytes(gguf_io.header_bytes(kvs, tensors))
    return src


@pytest.mark.parametrize("mtp", [False, True])
def test_round_trip_keeps_keys_and_tensors(tmp_path, mtp):
    src = with_dropped_key(fake_source(tmp_path / "src.gguf", mtp))
    kvs, tensors = gguf_io.read_header(src)
    dest = tmp_path / ("x" + headers.SUFFIX)
    assert headers.export_header(kvs, tensors, str(dest)) == dest.stat().st_size
    kvs2, tensors2 = headers.load_header(str(dest))
    assert [k for k in kvs if k.key != "quantize.imatrix.file"] == kvs2 and tensors == tensors2
    assert "general.quantized_by" in {k.key for k in kvs2}  # notices of origin travel with the header


def test_export_is_deterministic(tmp_path):
    kvs, tensors = gguf_io.read_header(fake_source(tmp_path / "src.gguf", False))
    a, b = tmp_path / "a.xz", tmp_path / "b.xz"
    headers.export_header(kvs, tensors, str(a))
    headers.export_header(kvs, tensors, str(b))
    assert a.read_bytes() == b.read_bytes()


@pytest.mark.parametrize("mtp", [False, True])
def test_mini_from_header_equals_mini_from_model(tmp_path, mtp):
    src = with_dropped_key(fake_source(tmp_path / "src.gguf", mtp))
    header = tmp_path / ("x" + headers.SUFFIX)
    headers.export_header(*gguf_io.read_header(src), str(header))
    a, b = tmp_path / "a.gguf", tmp_path / "b.gguf"
    make_mini_model.write_model(str(a), src, mtp, SMALL, 1)
    make_mini_model.write_model(str(b), str(header), mtp, SMALL, 1)
    assert hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest()
    assert b"somebody" not in a.read_bytes()


def test_split_model_header_lists_the_tensors_of_every_part(tmp_path):
    first = tmp_path / "m-00001-of-00002.gguf"
    kvs, tensors = gguf_io.read_header(fake_source(first, False))
    part2 = gguf_io.SourceTensor("late.weight", [4], 0)
    first.write_bytes(gguf_io.header_bytes(kvs, tensors[:-1]))
    (tmp_path / "m-00002-of-00002.gguf").write_bytes(gguf_io.header_bytes([], [tensors[-1], part2]))
    dest = tmp_path / ("m" + headers.SUFFIX)
    headers.export_header(*make_mini_model.read_source(str(first)), str(dest))
    names = [t.name for t in headers.load_header(str(dest))[1]]
    assert names[-2:] == [tensors[-1].name, "late.weight"] and len(names) == len(tensors) + 1


def test_not_a_header_is_refused(tmp_path):
    bad = tmp_path / ("bad" + headers.SUFFIX)
    import lzma

    bad.write_bytes(lzma.compress(b"NOPE" + struct.pack("<IQQ", 3, 0, 0)))
    with pytest.raises(ValueError, match="not a GGUF file"):
        headers.load_header(str(bad))
    with pytest.raises(SystemExit):
        make_mini_model.header_id(str(tmp_path / "other.header.gguf.xz"))


def test_unknown_model_id_is_refused(tmp_path):
    assert make_mini_model.export_header("nope", None, str(tmp_path)) == 2


def test_committed_headers_match_the_checked_models():
    for model_id in mini_token_check.MODELS:
        kvs, tensors = headers.load_header(str(DATA / (model_id + headers.SUFFIX)))
        arch = next(k for k in kvs if k.key == "general.architecture")
        assert arch.raw[8:].decode() == make_mini_model.ARCH_OF_ID[model_id]
        assert tensors
    expected = json.loads((DATA / "expected" / "cpu.json").read_text())
    assert set(expected["models"]) == set(mini_token_check.MODELS)
    assert expected["env"] == {"TONA_DETERMINISTIC": "1"}
    for model in expected["models"].values():
        assert [r["prompt"] for r in model["runs"]] == list(mini_token_check.PROMPTS)
        assert all(len(r["tokens"]) == r["n"] == mini_token_check.N_TOKENS for r in model["runs"])


FAKE_REF = """#!/usr/bin/env python3
import sys
sys.stderr.buffer.write(b"Prompt: 2 tokens\\n")
for i in range(4):
    sys.stderr.buffer.write(b"ref-hash %d 00000000000000a%d\\n" % (i, i))
    if i >= 1:
        sys.stderr.buffer.write(b"  top: %d 1.5 '\\x90bad'\\n" % (10 + i))
"""


def test_run_reference_reads_ids_and_hashes(tmp_path, monkeypatch):
    ref = tmp_path / "tona-ref"
    ref.write_text(FAKE_REF)
    ref.chmod(ref.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(mini_token_check, "N_TOKENS", 3)
    run = mini_token_check.run_reference(str(ref), "model.gguf", "hi")
    assert run["tokens"] == [11, 12, 13] and run["steps"] == 4
    steps = "".join(f"{i} 00000000000000a{i}\n" for i in range(4))
    assert run["logits_sha256"] == hashlib.sha256(steps.encode()).hexdigest()


def one_run(tokens, digest="a"):
    return {"n": len(tokens), "tokens": tokens, "logits_sha256": digest}


def test_differences_name_the_first_divergence():
    want = {"models": {"coder": {"model_sha256": "m", "runs": [one_run([1, 2, 3]), one_run([4, 5], "x")]}}}
    same = {"models": {"coder": {"model_sha256": "m", "runs": [one_run([1, 2, 3]), one_run([4, 5], "x")]}}}
    assert mini_token_check.differences(want, same) == []
    got = {"models": {"coder": {"model_sha256": "n", "runs": [one_run([1, 9, 3]), one_run([4, 5], "y")]}}}
    found = mini_token_check.differences(want, got)
    assert len(found) == 3
    assert "mini model differs" in found[0] and "first at 1" in found[1] and "different logits" in found[2]
