"""The command lines of make_mini_model, mini_token_check and get_model, with fakes: no big model, no engine."""

import json
import stat
from pathlib import Path

import get_model
import make_mini_model
import mini_token_check
import pytest
from mini_model import gguf_io, headers
from test_make_mini_model import fake_source


def run_main(module, monkeypatch, *argv) -> int:
    monkeypatch.setattr("sys.argv", ["prog", *argv])
    return module.main()


@pytest.fixture
def recorded(monkeypatch):
    """The writers replaced by recorders: the full-size shapes would write hundreds of megabytes."""
    calls = []
    for name in ("write_ornith", "write_bolt", "write_coder", "write_model"):
        monkeypatch.setattr(make_mini_model, name, lambda *a, _n=name: calls.append((_n, *a)))
    return calls


def test_export_header_command_writes_the_header(tmp_path, monkeypatch, capsys):
    src = fake_source(tmp_path / "src.gguf", False)
    out = tmp_path / "out"
    assert run_main(make_mini_model, monkeypatch, "--export-header", "coder", "--source", src, "--out", str(out)) == 0
    assert (out / ("coder" + headers.SUFFIX)).exists()
    assert "bytes" in capsys.readouterr().out
    assert run_main(make_mini_model, monkeypatch, "--export-header", "nope") == 2


@pytest.mark.parametrize(
    ("model_id", "writer"), [("ornith", "write_ornith"), ("bolt", "write_bolt"), ("coder", "write_coder")]
)
def test_from_header_picks_the_writer_of_the_id(tmp_path, monkeypatch, recorded, model_id, writer):
    header = tmp_path / (model_id + headers.SUFFIX)
    header.write_bytes(b"")
    out = tmp_path / "out"
    assert run_main(make_mini_model, monkeypatch, "--from-header", str(header), "--out", str(out)) == 0
    assert [c[0] for c in recorded] == [writer]
    assert recorded[0][1].startswith(str(out))
    if model_id == "coder":  # the MTP head comes from the header next to it
        assert recorded[0][3].endswith("coder-mtp" + headers.SUFFIX)


def test_from_header_f32_bolt(tmp_path, monkeypatch, recorded):
    header = tmp_path / ("bolt" + headers.SUFFIX)
    run_main(make_mini_model, monkeypatch, "--from-header", str(header), "--out", str(tmp_path), "--f32")
    assert recorded[0][1].endswith("mini-bolt-f32.gguf") and recorded[0][-1] is True


def test_default_folders_for_each_arch(tmp_path, monkeypatch, recorded):
    monkeypatch.setattr(make_mini_model, "MODELS", str(tmp_path))
    for arch in ("qwen35moe", "nemotron_h_moe", "qwen4exp"):
        assert run_main(make_mini_model, monkeypatch, "--arch", arch, "--source", "s.gguf") == 0
    assert [c[0] for c in recorded] == ["write_ornith", "write_bolt", "write_coder"]
    assert {p.name for p in tmp_path.iterdir()} == {"mini-ornith", "mini-bolt"}


def test_unknown_header_name_is_refused():
    with pytest.raises(SystemExit):
        make_mini_model.build_from_header("somewhere/other.header.gguf.xz", None, 1, False)


def test_not_a_gguf_file_is_named_in_the_error(tmp_path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOPE" + bytes(40))
    with pytest.raises(ValueError, match=r"bad\.gguf: not a GGUF file"):
        gguf_io.read_header(str(bad))


def test_get_model_builds_a_mini_model_from_its_header(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(make_mini_model, "build_from_header", lambda *a: calls.append(a))
    assert run_main(get_model, monkeypatch, "mini-ornith", "--dest", str(tmp_path)) == 0
    header, out, seed, f32 = calls[0]
    assert header.endswith("ornith.header.gguf.xz") and out == str(tmp_path / "mini-ornith")
    assert (seed, f32) == (1, False)
    assert capsys.readouterr().out.strip() == str(tmp_path / "mini-ornith" / "mini-ornith.gguf")


FAKE_REF = """#!/usr/bin/env python3
import sys
sys.stderr.write("ref-hash 0 00000000000000aa\\n  top: 7 1.0 'x'\\nref-hash 1 00000000000000bb\\n  top: 8 1.0 'y'\\n")
"""


@pytest.fixture
def fake_check(tmp_path, monkeypatch):
    """One mini model, a fake tona-ref and an expected file in tmp_path."""
    build = tmp_path / "build"
    build.mkdir()
    ref = build / "tona-ref"
    ref.write_text(FAKE_REF)
    ref.chmod(ref.stat().st_mode | stat.S_IXUSR)
    model = tmp_path / "models" / "mini.gguf"

    def fake_build(model_id, models):
        model.parent.mkdir(exist_ok=True)
        model.write_bytes(b"weights")
        return str(model)

    monkeypatch.setattr(mini_token_check, "build_model", fake_build)
    monkeypatch.setattr(mini_token_check, "MODELS", {"coder": ("mini", "mini.gguf")})
    monkeypatch.setattr(mini_token_check, "PROMPTS", ("p",))
    monkeypatch.setattr(mini_token_check, "N_TOKENS", 2)
    monkeypatch.setattr(mini_token_check, "EXPECTED", str(tmp_path / "expected" / "cpu.json"))
    return ["--build", str(build), "--models", str(tmp_path / "models")]


def test_record_then_compare_is_identical(fake_check, monkeypatch, capsys):
    assert run_main(mini_token_check, monkeypatch, "record", *fake_check) == 0
    saved = json.loads(Path(mini_token_check.EXPECTED).read_text())
    assert saved["models"]["coder"]["runs"][0]["tokens"] == [7, 8]
    assert run_main(mini_token_check, monkeypatch, "compare", *fake_check) == 0
    assert "identical" in capsys.readouterr().out


def test_compare_fails_when_the_tokens_change(fake_check, monkeypatch, capsys):
    run_main(mini_token_check, monkeypatch, "record", *fake_check)
    saved = json.loads(Path(mini_token_check.EXPECTED).read_text())
    saved["models"]["coder"]["runs"][0]["tokens"] = [7, 9]
    Path(mini_token_check.EXPECTED).write_text(json.dumps(saved))
    assert run_main(mini_token_check, monkeypatch, "compare", *fake_check) == 1
    assert "tokens differ (first at 1" in capsys.readouterr().out


def test_failing_reference_stops_the_check(fake_check, monkeypatch):
    ref = Path(fake_check[1]) / "tona-ref"
    ref.write_text("#!/bin/sh\necho broken >&2\nexit 3\n")
    with pytest.raises(SystemExit, match="tona-ref failed"):
        run_main(mini_token_check, monkeypatch, "record", *fake_check)


def test_build_model_uses_the_header_of_the_id(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(make_mini_model, "build_from_header", lambda *a: calls.append(a))
    path = mini_token_check.build_model("bolt", str(tmp_path))
    assert path == str(tmp_path / "mini-bolt" / "mini-bolt.gguf")
    assert calls[0][0].endswith("bolt.header.gguf.xz") and calls[0][2] == mini_token_check.SEED
    assert len(mini_token_check.sha256_file(__file__)) == 64
