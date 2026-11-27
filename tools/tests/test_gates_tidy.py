import json
import stat
import subprocess
import sys

import pytest
import tidy
from conftest import commit_all, git, write

# The venv follows the python3 of whoever ran setup.sh; the gate finds the toolchain under any Python version
LLVM_BIN = "work/rocm-10.1/lib/python3.12/site-packages/_rocm_sdk_core/lib/llvm/bin"


def test_pick_base_prefers_the_requested_ref(repo):
    write(repo, "a.txt")
    commit_all(repo)
    assert tidy.pick_base(None) == "main"
    assert tidy.pick_base("HEAD") == "HEAD"
    with pytest.raises(SystemExit) as stop:
        tidy.pick_base("no-such-ref")
    assert stop.value.code == 2


def test_changed_files_include_uncommitted_and_untracked(repo):
    write(repo, "keep.cpp")
    write(repo, "edit.cpp")
    commit_all(repo)
    git(repo, "checkout", "-q", "-b", "topic")
    write(repo, "edit.cpp", "// changed\n")
    commit_all(repo)
    write(repo, "keep.cpp", "// uncommitted\n")
    write(repo, "fresh.cpp")
    assert tidy.changed_files("main") == ["edit.cpp", "fresh.cpp", "keep.cpp"]


def test_lintable_needs_a_tree_a_suffix_and_an_existing_file(repo):
    for name in ("engine/a.cpp", "installer/b.h", "engine/third_party/c.cpp", "tools/d.cpp", "engine/e.txt"):
        write(repo, name)
    names = [
        "engine/a.cpp",
        "installer/b.h",
        "engine/third_party/c.cpp",
        "tools/d.cpp",
        "engine/e.txt",
        "engine/gone.cpp",
    ]
    assert tidy.lintable(names) == ["engine/a.cpp", "installer/b.h"]


def test_lintable_leaves_out_sources_of_another_platform_and_says_so(repo, monkeypatch, capsys):
    for name in ("engine/a.cpp", "engine/backend/metal/m.cpp"):
        write(repo, name)
    monkeypatch.setattr(tidy, "OTHER_PLATFORM", ("engine/backend/metal/",))
    assert tidy.lintable(["engine/a.cpp", "engine/backend/metal/m.cpp"]) == ["engine/a.cpp"]
    assert "engine/backend/metal/m.cpp" in capsys.readouterr().err
    monkeypatch.setattr(tidy, "OTHER_PLATFORM", ())
    assert tidy.lintable(["engine/a.cpp", "engine/backend/metal/m.cpp"]) == [
        "engine/a.cpp",
        "engine/backend/metal/m.cpp",
    ]


def test_database_choice_with_overrides():
    assert tidy.database_for("engine/a.cpp", {}) == "work/build/engine"
    assert tidy.database_for("installer/a.cpp", {"installer/": "other"}) == "other"
    assert tidy.database_for("engine/a.cpp", {"engine/": None}) == "work/build/engine"


def write_database(repo, files, db="work/build/engine"):
    entries = [{"file": str(repo / f), "directory": str(repo), "command": "c++ x"} for f in files]
    write(repo, f"{db}/compile_commands.json", json.dumps(entries))


def test_missing_database_is_an_error(repo):
    with pytest.raises(SystemExit) as stop:
        tidy.database_files("work/build/engine")
    assert stop.value.code == 2


def test_check_database(repo):
    write(repo, "engine/a.cpp")
    write_database(repo, ["engine/a.cpp"])
    tidy.check_database("work/build/engine", ["engine/a.cpp"])
    with pytest.raises(SystemExit):  # the file is not in the database: it would be linted by nothing
        tidy.check_database("work/build/engine", ["engine/b.cpp"])
    write(repo, "work/build/engine/compile_commands.json", json.dumps([{"file": "/elsewhere/a.cpp"}]))
    with pytest.raises(SystemExit):  # a database of another checkout
        tidy.check_database("work/build/engine", ["engine/a.cpp"])


def test_include_graph(repo):
    write(repo, "engine/util.h")
    write(repo, "engine/mid.h", '#include "util.h"\n')
    write(repo, "engine/a.cpp", '#include "mid.h"\n#include <vector>\n')
    write(repo, "engine/sub/b.cpp", '#include "util.h"\n')  # found through the tree root, not the own directory
    write(repo, "engine/lonely.h")
    commit_all(repo)
    users = tidy.reverse_includes("engine/")
    assert users["engine/util.h"] == {"engine/mid.h", "engine/sub/b.cpp"}
    assert tidy.including_sources("engine/util.h", "engine/", users) == {"engine/a.cpp", "engine/sub/b.cpp"}
    assert tidy.including_sources("engine/lonely.h", "engine/", users) == set()
    assert tidy.resolve_include("engine/a.cpp", "mid.h", "engine/") == "engine/mid.h"
    assert tidy.resolve_include("engine/a.cpp", "missing.h", "engine/") is None


def test_plan_sources_lints_headers_through_their_includers(repo):
    write(repo, "engine/util.h")
    write(repo, "engine/lonely.h")
    write(repo, "engine/a.cpp", '#include "util.h"\n')
    commit_all(repo)
    known = {str(repo / "engine/a.cpp")}
    sources, orphans = tidy.plan_sources(["engine/util.h", "engine/lonely.h", "engine/a.cpp"], known)
    assert sources == ["engine/a.cpp"] and orphans == ["engine/lonely.h"]


def test_config_dir_is_the_nearest_clang_tidy(repo):
    write(repo, ".clang-tidy")
    write(repo, "engine/gpu/.clang-tidy")
    write(repo, "engine/gpu/k.cpp")
    write(repo, "engine/a.cpp")
    assert tidy.config_dir("engine/gpu/k.cpp") == repo / "engine/gpu"
    assert tidy.config_dir("engine/a.cpp") == repo


def test_small_helpers(repo):
    assert tidy.is_header("a.h") and tidy.is_header("a.hpp") and not tidy.is_header("a.cpp")
    assert tidy.tree_of("installer/x.cpp") == "installer/"
    regex = tidy.file_filter(["engine/a.cpp"])
    assert regex.startswith("^(") and "engine/a\\.cpp" in regex


def test_findings_count_only_in_changed_files(capsys):
    output = "/r/engine/a.cpp:3:5: warning: bad [x]\n/r/engine/other.h:1:1: error: old [y]\nnoise\n"
    assert tidy.own_findings(output, {"/r/engine/a.cpp"}) == {"/r/engine/a.cpp:3:5: warning: bad [x]"}
    with pytest.raises(SystemExit) as stop:
        tidy.own_findings("Error while processing /r/x.cpp", set())
    assert stop.value.code == 2


def test_parse_args(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["tidy.py", "--all", "-j", "3", "--base", "main", "a.cpp"])
    args = tidy.parse_args()
    assert args.all and args.j == 3 and args.base == "main" and args.files == ["a.cpp"]


def install_fake_llvm(repo, finding=""):
    """A toolchain whose run-clang-tidy and clang-tidy print a fixed text."""
    for tool in ("run-clang-tidy", "clang-tidy"):
        path = write(repo, f"{LLVM_BIN}/{tool}", f"#!/bin/sh\ncat <<'EOF'\n{finding}\nEOF\n")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)


def run_main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["tidy.py", *args])
    monkeypatch.setattr(tidy, "SYSTEM", "Linux")  # the fake toolchain is the ROCm layout
    return tidy.main()


def test_main_without_toolchain_is_an_error_not_a_pass(repo, monkeypatch):
    with pytest.raises(SystemExit) as stop:
        run_main(monkeypatch)
    assert stop.value.code == 2


def test_main_without_changed_cpp_files(repo, monkeypatch, capsys):
    write(repo, "a.txt")
    commit_all(repo)
    install_fake_llvm(repo)
    assert run_main(monkeypatch, "--base", "main") == 0
    assert "no changed C++/HIP files" in capsys.readouterr().out


def test_main_reports_findings_of_the_changed_file(repo, monkeypatch, capsys):
    write(repo, "engine/a.cpp", "int main() {}\n")
    commit_all(repo)
    write_database(repo, ["engine/a.cpp"])
    install_fake_llvm(repo, f"{repo}/engine/a.cpp:1:1: warning: bad [bugprone-x]")
    assert run_main(monkeypatch, "engine/a.cpp") == 1
    assert "1 finding(s) in 1 changed file(s)" in capsys.readouterr().err


def test_main_passes_on_clean_output(repo, monkeypatch, capsys):
    write(repo, "engine/a.cpp", "int main() {}\n")
    commit_all(repo)
    write_database(repo, ["engine/a.cpp"])
    install_fake_llvm(repo)
    assert run_main(monkeypatch, "engine/a.cpp") == 0
    assert "tidy: ok (1 checked)" in capsys.readouterr().out


def test_main_with_a_header_only_file_lints_it_on_its_own(repo, monkeypatch, capsys):
    write(repo, "engine/lonely.h", "#pragma once\n")
    commit_all(repo)
    write(repo, "engine/a.cpp")
    commit_all(repo)
    write_database(repo, ["engine/a.cpp"])
    install_fake_llvm(repo, f"{repo}/engine/lonely.h:1:1: warning: bad [x]")
    assert run_main(monkeypatch, "--all") == 1


def test_each_platform_leaves_out_what_only_the_other_builds():
    assert "engine/backend/metal/" in tidy.other_platform("Linux")
    assert "engine/gpu/" not in tidy.other_platform("Linux")
    assert "engine/gpu/" in tidy.other_platform("Darwin")
    assert "engine/backend/metal/" not in tidy.other_platform("Darwin")


def test_objective_cpp_sources_are_lintable(repo, monkeypatch):
    write(repo, "engine/backend/metal/pass.mm")
    monkeypatch.setattr(tidy, "OTHER_PLATFORM", ())
    assert tidy.lintable(["engine/backend/metal/pass.mm"]) == ["engine/backend/metal/pass.mm"]


def test_linux_toolchain_is_the_rocm_one(repo):
    install_fake_llvm(repo)
    tc = tidy.toolchain("Linux", repo)
    assert tc.clang_tidy == repo / LLVM_BIN / "clang-tidy"
    assert tc.run_tidy == [str(repo / LLVM_BIN / "run-clang-tidy")]
    assert tc.extra == []


def test_macos_toolchain_is_the_wheel_with_the_sdk(repo, monkeypatch):
    llvm = "work/venv-tidy/lib/python3.12/site-packages/clang_tidy/data/bin"
    for tool in ("clang-tidy", "run-clang-tidy.py"):
        write(repo, f"{llvm}/{tool}")
    sdk = subprocess.CompletedProcess([], 0, stdout="/sdk/MacOSX.sdk\n", stderr="")
    monkeypatch.setattr(tidy.subprocess, "run", lambda *a, **k: sdk)
    tc = tidy.toolchain("Darwin", repo)
    assert tc.clang_tidy == repo / llvm / "clang-tidy"
    assert tc.run_tidy == [sys.executable, str(repo / llvm / "run-clang-tidy.py")]
    assert tc.extra == ["-extra-arg=-isysroot/sdk/MacOSX.sdk"]


def test_missing_toolchain_is_an_error_on_both_platforms(repo):
    for system in ("Linux", "Darwin"):
        with pytest.raises(SystemExit) as stop:
            tidy.toolchain(system, repo)
        assert stop.value.code == 2


def test_system_toolchain_is_the_one_on_the_path(monkeypatch):
    monkeypatch.setattr(tidy.shutil, "which", lambda name: f"/usr/bin/{name}")
    tc = tidy.system_toolchain()
    assert str(tc.clang_tidy) == "/usr/bin/clang-tidy" and tc.run_tidy == ["/usr/bin/run-clang-tidy"]
    monkeypatch.setattr(tidy.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit) as stop:
        tidy.system_toolchain()
    assert stop.value.code == 2


def test_system_mode_leaves_out_what_the_host_database_lacks(repo, capsys):
    write(repo, "engine/a.cpp")
    write(repo, "engine/gpu/b.cpp")
    write_database(repo, ["engine/a.cpp"])
    kept = tidy.host_only("work/build/engine", ["engine/a.cpp", "engine/gpu/b.cpp", "engine/gpu/c.h"])
    assert kept == ["engine/a.cpp", "engine/gpu/c.h"]
    assert "1 source(s) not in work/build/engine" in capsys.readouterr().err
