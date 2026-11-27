import sys
from pathlib import Path

import pytest
from conftest import load_script

isa = load_script("scripts/isa-diff.py")

DISASSEMBLY = """
file format elf64-amdgpu

Disassembly of section .text:

<k_add>:
\ts_load_dwordx2 s[0:1], s[4:5], 0x0 // 000000000100: comment
\tv_add_f32 v0, v1, v2
\t...
\ts_endpgm

<helper>:
\ts_endpgm
"""

NOTES = """
amdhsa.kernels:
  - .args:
      - .name: a
        .offset: 0
    .group_segment_fixed_size: 0
    .name: k_add
    .vgpr_count: 24
    .sgpr_count: 18
    .other: 5
  - .group_segment_fixed_size: 128
    .name: k_mul
amdhsa.version:
  - 1
"""


def test_kernels_strip_addresses_comments_and_padding(monkeypatch):
    monkeypatch.setattr(isa, "tool", lambda name, *args: DISASSEMBLY)
    assert isa.kernels(Path("x.co")) == {
        "k_add": ["s_load_dwordx2 s[0:1], s[4:5], 0x0", "v_add_f32 v0, v1, v2", "s_endpgm"],
        "helper": ["s_endpgm"],
    }


def test_metadata_keeps_the_resource_fields_of_each_record_before_and_after_the_name(monkeypatch):
    monkeypatch.setattr(isa, "tool", lambda name, *args: NOTES)
    assert isa.metadata(Path("x.co")) == {
        "k_add": {"group_segment_fixed_size": "0", "vgpr_count": "24", "sgpr_count": "18"},
        "k_mul": {"group_segment_fixed_size": "128"},
    }


def test_code_object_is_none_without_device_code(monkeypatch, tmp_path):
    monkeypatch.setattr(isa, "tool", lambda name, *args: ".text .data")
    assert isa.code_object(Path("a.o"), tmp_path) is None


def test_code_object_unbundles_the_gfx1200_part(monkeypatch, tmp_path):
    calls = []

    def fake(name, *args):
        calls.append((name, args))
        return ".hip_fatbin" if name == "llvm-readelf" else ""

    monkeypatch.setattr(isa, "tool", fake)
    co = isa.code_object(Path("dir/a.o"), tmp_path)
    assert co == tmp_path / "a.o.co"
    assert [c[0] for c in calls] == ["llvm-readelf", "llvm-objcopy", "clang-offload-bundler"]
    assert f"--targets={isa.TARGET}" in calls[2][1]


def test_built_objects_lists_only_existing_objects(monkeypatch, tmp_path):
    (tmp_path / "a.o").write_text("")
    listing = "a.o: CXX_COMPILER\nstale.o: CXX_COMPILER\nlib.a: link\nb.cpp: x\n"

    class Done:
        stdout = listing

    monkeypatch.setattr(isa.subprocess, "run", lambda *a, **k: Done())
    assert isa.built_objects(tmp_path) == [tmp_path / "a.o"]


def fake_build(monkeypatch, builds):
    """collect() over invented builds: name -> (kernels, metadata); the first object of a build wins a name."""
    monkeypatch.setattr(isa, "built_objects", lambda build: [Path(build)])
    monkeypatch.setattr(isa, "code_object", lambda obj, tmp: obj)
    monkeypatch.setattr(isa, "kernels", lambda co: builds[str(co)][0])
    monkeypatch.setattr(isa, "metadata", lambda co: builds[str(co)][1])


def run_main(monkeypatch, a, b):
    monkeypatch.setattr(sys, "argv", ["isa-diff.py", a, b])
    with pytest.raises(SystemExit) as stop:
        isa.main()
    return stop.value.code


def test_identical_builds(monkeypatch, capsys):
    same = ({"k": ["a", "b"]}, {"k": {"vgpr_count": "8"}})
    fake_build(monkeypatch, {"A": same, "B": same})
    assert run_main(monkeypatch, "A", "B") == 0
    assert "all kernels identical" in capsys.readouterr().out


def test_every_kind_of_difference_is_reported(monkeypatch, capsys):
    a = ({"gone": ["x"], "k": ["a"], "r": ["a"]}, {"r": {"vgpr_count": "8"}})
    b = ({"new": ["x"], "k": ["b"], "r": ["a"]}, {"r": {"vgpr_count": "9"}})
    fake_build(monkeypatch, {"A": a, "B": b})
    assert run_main(monkeypatch, "A", "B") == 1
    out = capsys.readouterr().out
    for line in (
        "removed: gone",
        "added: new",
        "code differs: k",
        "resources differ: r: {'vgpr_count': '8'} -> {'vgpr_count': '9'}",
    ):
        assert line in out


def test_wrong_argument_count_prints_the_usage(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["isa-diff.py"])
    with pytest.raises(SystemExit) as stop:
        isa.main()
    assert "Compare the gfx1200 machine code" in str(stop.value)


def test_demangled_renames_after_demangling(monkeypatch):
    class Done:
        stdout = "void k<1, A>(int)\nhelper()\n"

    monkeypatch.setattr(isa.subprocess, "run", lambda *a, **k: Done())
    table = {"_Z1": ["s_endpgm"], "_Z2": ["v_nop"]}
    assert isa.demangled(table, [(r"k<(\d), A>", r"k<\1, P<A>>")]) == {
        "void k<1, P<A>>(int)": ["s_endpgm"],
        "helper()": ["v_nop"],
    }


def test_parse_args_collects_rename_rules_and_both_builds():
    rules, builds = isa.parse_args(["--rename", "a=b", "x", "--rename", "c=d=e", "y"])
    assert rules == [("a", "b"), ("c", "d=e")]
    assert builds == ["x", "y"]
