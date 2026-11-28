import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import select_checks as sc
import select_graph as sg
import select_rules as rules
from conftest import ROOT, commit_all, git, write

# The status script runs with PATH=/usr/bin:/bin, its environment on the Linux GPU machine. macOS has bash 3.2 there
# (no mapfile) and Python 3.9, and the script is a tool of the reference GPU only (docs/use/macos.md)
needs_status_shell = pytest.mark.skipif(
    not shutil.which("bash") or sys.platform == "darwin", reason="bash of the Linux GPU machine"
)

ALL = ["gpu", "cpu plan", "prefill", "prefill dflash", "server"]
GEN = ["gpu", "cpu plan", "prefill", "prefill dflash"]

# What `ninja -t deps` prints: object, then the files it was compiled from (the dependency log)
DEPS = """\
CMakeFiles/tona_gpu.dir/gpu/ops.hip.o: #deps 3, deps mtime 1 (VALID)
    /repo/engine/gpu/ops.hip
    /repo/engine/gpu/gemm_dev.h
    /usr/include/stdio.h

CMakeFiles/tona_server.dir/server/api.cpp.o: #deps 2, deps mtime 1 (VALID)
    /repo/engine/server/api.cpp
    /repo/engine/server/api.h
"""
# What `ninja -t inputs tona-server` prints: sources and objects of the whole link, outside files too
INPUTS = """\
.
/repo/engine/server/api.cpp
/repo/work/rocm/lib/libamdhip64.so
CMakeFiles/tona_server.dir/server/api.cpp.o
/usr/lib64/libpthread.a
"""


def graph(**tests) -> sg.Graph:
    """A small build: the generator and the server share the GPU library, the tests have one binary each."""
    shared = {"engine/gpu/forward.hip", "engine/gpu/mega.h", "engine/gpu/gemm_dev.h", "engine/backend/stall.h"}
    return sg.Graph(
        Path("/repo"),
        {
            "tona-gpu-gen": frozenset(shared | {"engine/tools/tona_gpu_gen.cpp", "engine/cpu/ref.cpp"}),
            "tona-server": frozenset(
                shared | {"engine/server/engine.cpp", "engine/server/api.cpp", "engine/server/api.h"}
            ),
            "tona": frozenset({"engine/tools/tona.cpp", "engine/update/fetch.cpp"}),
            "test_x": frozenset({"engine/tests/test_x.cpp", "engine/server/api.h"}),
            "test_stall": frozenset({"engine/tests/test_stall.cpp", "engine/backend/stall.h"}),
        },
        frozenset({"engine/gpu/forward.hip", "engine/gpu/gemm_dev.h"}),
        {"x": ("test_x", True), "stall": ("test_stall", True), "model_shapes": ("tona", False)} | tests,
    )


@pytest.fixture
def fake_graph(monkeypatch):
    monkeypatch.setattr(sg, "load", lambda build: graph())


def plan(*paths: str, status: str = "M") -> dict:
    return sc.plan_for([(status, p) for p in paths], Path("build")).to_dict()


def test_only_a_readme_needs_only_the_gates(fake_graph):
    data = plan("README.md", "docs/contribute/verification.md", "research/measurements/x.md")
    assert (data["class"], data["goldens"], data["recipes"]) == ("C0", [], ["R0"])
    assert not data["unit_tests"] and not data["python_tests"] and not data["bit_exact"]


def test_a_docs_change_needs_no_build(monkeypatch):
    def broken(build):
        raise sg.GraphError("no build")

    monkeypatch.setattr(sg, "load", broken)
    assert plan("README.md")["class"] == "C0"


def test_a_script_needs_the_gates_and_the_python_tests(fake_graph):
    data = plan("scripts/bench-set.py", "bench/needle/run.py")
    assert (data["class"], data["goldens"], data["python_tests"]) == ("C1", [], True)
    assert data["recipes"] == ["R0", "R1"]


@pytest.mark.parametrize(
    "path",
    ["scripts/golden.sh", "scripts/golden-server.sh", "scripts/bench-long-code.txt", "scripts/build-mtp-vocab.py",
     "tools/gen_tables.py", "toolchain/rocm-10.1.lock", "engine/CMakeLists.txt", "cmake/x.cmake", "unknown.bin"],
)  # fmt: skip
def test_what_steers_the_build_or_the_golden_runs_triggers_everything(fake_graph, path):
    data = plan(path)
    assert data["goldens"] == ALL and data["everything"] and data["unit_all"] and data["bit_exact"]
    assert data["class"] == "C5"


def test_the_tools_and_this_script_trigger_nothing(fake_graph):
    data = plan("tools/select_checks.py", "scripts/gpu-status.sh", "tools/gates/tidy.py")
    assert data["goldens"] == [] and data["class"] == "C1"


def test_mini_model_data_triggers_no_golden(fake_graph):
    data = plan("data/mini-models/coder.header.gguf.xz", "data/mini-models/expected/cpu.json")
    assert (data["class"], data["goldens"], data["python_tests"]) == ("C1", [], True)


def test_a_test_file_runs_its_unit_test_and_no_golden(fake_graph):
    data = plan("engine/tests/test_x.cpp")
    assert (data["class"], data["goldens"], data["unit_tests"]) == ("C1", [], ["x"])
    assert data["skipped"] == ALL and not data["bit_exact"] and not data["speed"]


def test_a_test_named_after_long_context_code_raises_no_hint(monkeypatch):
    g = graph()
    g.targets["t"] = frozenset({"engine/tests/test_rope_yarn.cpp"})
    monkeypatch.setattr(sg, "load", lambda build: g)
    assert plan("engine/tests/test_rope_yarn.cpp")["needs"] == []


def test_a_device_header_runs_everything_and_asks_for_bit_exact_and_speed(fake_graph):
    data = plan("engine/gpu/gemm_dev.h")
    assert data["goldens"] == ALL and data["class"] == "C5"
    assert data["bit_exact"] and data["speed"] and data["kernel_identity"]
    assert data["recipes"] == ["R0", "R1", "R3", "R4", "R5"]


def test_host_code_in_the_gpu_directory_is_compute_but_not_device_code(fake_graph):
    data = plan("engine/gpu/mega.h")
    assert data["bit_exact"] and data["speed"] and not data["kernel_identity"] and data["class"] == "C3"


def test_a_server_file_that_only_the_server_contains_runs_only_the_server_golden(fake_graph):
    data = plan("engine/server/engine.cpp")
    assert data["goldens"] == ["server"] and data["skipped"] == GEN
    assert data["skipped_seconds"] == sum(sc.GOLDEN_SECONDS[g] for g in GEN)
    assert data["golden_seconds"] == sc.GOLDEN_SECONDS["server"]
    assert data["class"] == "C3" and not data["bit_exact"]


def test_an_api_file_is_off_the_hot_path_and_reaches_its_unit_test(fake_graph):
    data = plan("engine/server/api.h")
    assert (data["class"], data["goldens"], data["unit_tests"]) == ("C2", ["server"], ["x"])


def test_the_generator_tool_runs_the_generator_goldens_only(fake_graph):
    data = plan("engine/tools/tona_gpu_gen.cpp")
    assert data["goldens"] == GEN and data["skipped"] == ["server"]


def test_a_binary_without_golden_check_needs_only_its_unit_tests(fake_graph):
    data = plan("engine/update/fetch.cpp")
    assert (data["class"], data["goldens"], data["unit_tests"]) == ("C2", [], [])
    assert data["recipes"] == ["R0", "R1"]


def test_the_unit_tests_of_a_shared_header_are_named(fake_graph):
    assert plan("engine/backend/stall.h")["unit_tests"] == ["stall"]


def test_an_engine_file_in_no_binary_triggers_everything(fake_graph):
    for status in ("M", "D"):
        assert plan("engine/server/orphan.cpp", status=status)["goldens"] == ALL


def test_a_new_header_is_covered_by_the_includers_in_the_change(fake_graph):
    assert plan("engine/server/new.h", status="A")["goldens"] == []
    data = plan("engine/server/new.h", "engine/server/engine.cpp", status="A")
    assert data["goldens"] == ["server"]


def test_test_data_runs_all_unit_tests(fake_graph):
    data = plan("engine/tests/data/chat_cases.json")
    assert data["unit_all"] and data["goldens"] == [] and data["class"] == "C1"


def test_the_installer_has_its_own_tests(monkeypatch):
    g = graph()
    g.targets["tona"] |= {"engine/third_party/x.h"}
    monkeypatch.setattr(sg, "load", lambda build: g)
    assert plan("installer/core/hw.cpp")["installer_tests"]
    assert plan("engine/third_party/x.h")["installer_tests"]
    assert not plan("installer/core/hw.cpp")["goldens"]


def test_a_missing_build_triggers_everything_and_says_why(monkeypatch):
    data = sc.plan_for([("M", "engine/server/engine.cpp")], Path("/nonexistent/build")).to_dict()
    assert data["goldens"] == ALL and data["everything"] and "nonexistent" in data["error"]


def test_long_context_and_pressure_paths_are_named(monkeypatch):
    g = graph()
    g.targets["tona-server"] |= {
        "engine/server/checkpoint_file.cpp",
        "engine/server/prompt_cache.cpp",
        "engine/gpu/mapping.cpp",
    }
    monkeypatch.setattr(sg, "load", lambda build: g)
    assert plan("engine/server/prompt_cache.cpp")["needs"] == ["long-context run"]
    data = plan("engine/server/checkpoint_file.cpp")
    assert data["needs"] == ["long-context run", "pressure series"]
    assert data["class"] == "C8" and "C7" in data["classes"] and {"R7", "R8", "R4"} <= set(data["recipes"])
    # a name part decides, not a substring: `mapping` is no `pin`
    assert plan("engine/gpu/mapping.cpp")["needs"] == []


def test_the_class_follows_the_highest_file(fake_graph):
    data = plan("README.md", "scripts/ab.py", "engine/server/engine.cpp", "engine/gpu/gemm_dev.h")
    assert data["class"] == "C5" and data["classes"] == ["C0", "C1", "C3", "C5"]


def test_object_deps_and_target_files_read_the_ninja_output():
    deps = sg.object_deps(DEPS, Path("/repo"))
    assert deps["CMakeFiles/tona_gpu.dir/gpu/ops.hip.o"] == {"engine/gpu/ops.hip", "engine/gpu/gemm_dev.h"}
    files = sg.target_files(INPUTS, Path("/repo"), deps)
    assert files == {"engine/server/api.cpp", "engine/server/api.h"}  # outside files and work/ drop out


def test_a_graph_without_headers_is_refused():
    with pytest.raises(sg.GraphError):
        sg.check_complete({"a": frozenset({"engine/a.cpp"})})
    with pytest.raises(sg.GraphError):
        sg.check_complete({})


def test_changed_files_read_the_diff_against_the_merge_base(repo):
    write(repo, "a.txt", "1")
    commit_all(repo)
    git(repo, "checkout", "-q", "-b", "topic")
    write(repo, "b.txt", "2")
    (repo / "a.txt").unlink()
    commit_all(repo)
    assert sorted(sc.changed_files("main")) == [("A", "b.txt"), ("D", "a.txt")]


def test_the_report_and_the_json_name_class_goldens_and_hints(fake_graph):
    data = plan("engine/server/engine.cpp")
    text = sc.render(data, "origin/main")
    assert (
        "Change class C3" in text
        and f"golden: server ({sc.GOLDEN_SECONDS['server']} s)" in text
        and "skipped: gpu" in text
    )
    assert json.loads(json.dumps(data))["goldens"] == ["server"]


def test_the_tables_agree_with_the_policy_and_the_status_script():
    assert set(rules.RECIPES) == {"C0", "C1", "C2", "C3", "C5"}
    script = (ROOT / "scripts/gpu-status.sh").read_text()
    for name in rules.ALL_GOLDENS:
        assert f'"{name}"' in script or f"\n        {name})" in script
    assert "ENGINE_PATHS" not in script


@pytest.fixture
def status_repo(repo):
    """A repository with the status script, the tool and golden stubs that only record that they ran."""
    for name in (
        "scripts/gpu-status.sh",
        "scripts/gpu-lock.sh",
        "tools/select_checks.py",
        "tools/select_graph.py",
        "tools/select_rules.py",
        "tools/select_cmake.py",
    ):
        write(repo, name, (ROOT / name).read_text())
    for name in ("golden.sh", "golden-server.sh"):
        script = 'label="${GOLDEN_MODE:-decode}/${GOLDEN_PLAN:-gpu}/${GOLDEN_DRAFTER:-none} $(basename "$0")"\n'
        script += 'echo "$label" >>"$RAN"\n[[ "$label" != "${FAIL:-}" ]]\n'
        write(repo, f"stub/{name}", "#!/usr/bin/env bash\n" + script)
        (repo / "stub" / name).chmod(0o755)
    commit_all(repo)
    git(repo, "checkout", "-q", "-b", "topic")
    return repo


def run_status(repo, ran: Path, build: str = "missing", *args: str, fail: str = "x") -> subprocess.CompletedProcess:
    env = {
        "FAIL": fail,
        "TONA_GPU_LOCK_HELD": "1",  # the lock itself is tested in test_gpu_lock.py
        "PATH": "/usr/bin:/bin",
        "RAN": str(ran),
        "HOME": str(repo),
        "GPU_STATUS_GOLDEN": str(repo / "stub/golden.sh"),
        "GPU_STATUS_GOLDEN_SERVER": str(repo / "stub/golden-server.sh"),
    }
    return subprocess.run(
        ["bash", "scripts/gpu-status.sh", "--base", "main", "--build", build, *args],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@needs_status_shell
def test_the_status_script_skips_the_goldens_for_a_script_change(status_repo, tmp_path):
    write(status_repo, "scripts/bench-set.py", "x = 1")
    commit_all(status_repo)
    ran = tmp_path / "ran"
    done = run_status(status_repo, ran, "missing", "--own")
    assert done.returncode == 0, done.stderr
    assert "Status gpu-golden" in done.stdout and "C1: no golden check needed; skipped 5" in done.stdout
    assert not ran.exists()


@needs_status_shell
def test_the_status_script_runs_everything_when_the_build_graph_is_missing(status_repo, tmp_path):
    write(status_repo, "engine/gpu/x.h", "x")
    commit_all(status_repo)
    ran = tmp_path / "ran"
    done = run_status(status_repo, ran, "missing", "--own")
    assert done.returncode == 0, done.stderr
    assert len(ran.read_text().splitlines()) == 5
    assert "C5: golden identical (prefill dflash;gpu;cpu plan;server;prefill)" in done.stdout


@needs_status_shell
def test_the_status_script_runs_the_cheapest_first_and_stops_at_the_first_red(status_repo, tmp_path):
    # the gpu check fails: second in the order
    write(status_repo, "engine/gpu/x.h", "x")
    commit_all(status_repo)
    ran = tmp_path / "ran"
    done = run_status(status_repo, ran, "missing", "--own", fail="decode/gpu/none golden.sh")
    assert done.returncode == 1
    assert ran.read_text().splitlines() == ["prefill/gpu/dflash golden.sh", "decode/gpu/none golden.sh"]
    assert "not run after the first failure: cpu plan,server,prefill" in done.stdout


@needs_status_shell
def test_the_status_script_keeps_going_when_asked(status_repo, tmp_path):
    write(status_repo, "engine/gpu/x.h", "x")
    commit_all(status_repo)
    ran = tmp_path / "ran"
    done = run_status(status_repo, ran, "missing", "--own", "--keep-going", fail="decode/gpu/none golden.sh")
    assert done.returncode == 1
    assert len(ran.read_text().splitlines()) == 5 and "not run" not in done.stdout


def test_load_builds_the_graph_from_the_ninja_and_ctest_output(tmp_path, monkeypatch):
    (tmp_path / "CMakeCache.txt").write_text("CMAKE_HOME_DIRECTORY:INTERNAL=/repo/engine\n")
    ctest = {"tests": [{"name": "x", "command": ["/b/test_x"], "properties": [{"name": "LABELS", "value": ["unit"]}]},
                       {"name": "m", "command": ["/b/tona-model", "m.gguf"]}]}  # fmt: skip

    def fake(cmd):
        if "targets" in cmd:
            return (
                "tona-server: CXX_EXECUTABLE_LINKER__tona-server_RelWithDebInfo\nlibx.a: CXX_STATIC_LIBRARY_LINKER__x\n"
            )
        if "deps" in cmd:
            return DEPS
        if "inputs" in cmd:
            return INPUTS
        return json.dumps(ctest)

    monkeypatch.setattr(sg, "run", fake)
    monkeypatch.setattr(sg.shutil, "which", lambda name: "/usr/bin/ninja")
    loaded = sg.load(tmp_path)
    assert loaded.root == Path("/repo") and list(loaded.targets) == ["tona-server"]
    assert loaded.device == {"engine/gpu/ops.hip", "engine/gpu/gemm_dev.h"}
    assert loaded.tests == {"x": ("test_x", True), "m": ("tona-model", False)}
    assert loaded.reaching("engine/server/api.h") == ["tona-server"] and loaded.known("engine/server/api.cpp")


def test_load_refuses_a_missing_build_a_missing_cache_and_a_failing_tool(tmp_path, monkeypatch):
    with pytest.raises(sg.GraphError):
        sg.load(tmp_path / "none")
    monkeypatch.setattr(sg.shutil, "which", lambda name: "/usr/bin/ninja")
    with pytest.raises(sg.GraphError):
        sg.load(tmp_path)  # no CMakeCache.txt
    (tmp_path / "CMakeCache.txt").write_text("nothing\n")
    with pytest.raises(sg.GraphError):
        sg.load(tmp_path)  # no source directory in the cache
    with pytest.raises(sg.GraphError):
        sg.run(["/nonexistent/tool"])


def test_main_prints_the_report_and_the_json_for_a_docs_change(repo, monkeypatch, capsys):
    write(repo, "README.md", "1")
    commit_all(repo)
    git(repo, "checkout", "-q", "-b", "topic")
    write(repo, "README.md", "2")
    commit_all(repo)
    monkeypatch.setattr("sys.argv", ["select_checks.py", "--base", "main"])
    assert sc.main() == 0
    assert "Change class C0" in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["select_checks.py", "--base", "main", "--json", "--head", "HEAD"])
    assert sc.main() == 0
    assert json.loads(capsys.readouterr().out)["class"] == "C0"


def test_a_failing_git_call_stops_the_tool_so_that_callers_check_everything(repo):
    with pytest.raises(SystemExit):
        sc.changed_files("no-such-ref")


def test_the_unit_test_line_names_what_to_run():
    assert sc.unit_text({"unit_all": True}) == "all (just test)"
    data = {"unit_all": False, "unit_tests": ["a", "b"], "python_tests": True, "installer_tests": True}
    assert sc.unit_text(data) == "ctest -R '^(a|b)$'; just test; installer build and ctest"
    assert (
        sc.unit_text({"unit_all": False, "unit_tests": [], "python_tests": False, "installer_tests": False}) == "none"
    )


def cmake_plan(monkeypatch, differing, *paths):
    monkeypatch.setattr(sc.select_cmake, "changed_binaries", lambda repo, base, head: differing)
    monkeypatch.setattr(sc, "git", lambda *args: "/repo\n")
    return sc.plan_for([("M", p) for p in paths], Path("build"), ("base", "head")).to_dict()


def test_a_cmake_change_that_leaves_the_golden_commands_alone_needs_no_golden(fake_graph, monkeypatch):
    data = cmake_plan(monkeypatch, [], "engine/CMakeLists.txt")  # e.g. a new test target
    assert data["goldens"] == [] and data["class"] == "C1" and data["unit_all"] and not data["everything"]


def test_changed_compiler_flags_of_the_server_run_only_the_server_golden(fake_graph, monkeypatch):
    data = cmake_plan(monkeypatch, ["tona-server"], "engine/CMakeLists.txt")
    assert data["goldens"] == ["server"] and data["class"] == "C3" and not data["bit_exact"]
    data = cmake_plan(monkeypatch, ["tona-gpu-gen", "tona-server"], "engine/CMakeLists.txt")
    assert data["goldens"] == ALL and data["class"] == "C5"


def test_a_configuration_error_or_missing_refs_trigger_everything(fake_graph, monkeypatch):
    def broken(repo, base, head):
        raise sg.GraphError("cmake failed")

    monkeypatch.setattr(sc.select_cmake, "changed_binaries", broken)
    monkeypatch.setattr(sc, "git", lambda *args: "/repo\n")
    assert sc.plan_for([("M", "engine/CMakeLists.txt")], Path("b"), ("a", "b")).to_dict()["goldens"] == ALL
    assert sc.plan_for([("M", "engine/CMakeLists.txt")], Path("b")).to_dict()["goldens"] == ALL


def test_select_cmake_compares_normalized_commands(tmp_path, monkeypatch):
    import select_cmake as cm

    monkeypatch.setattr(cm, "SCRATCH", tmp_path)
    commands = {
        "base": {"tona-gpu-gen": "cc -I{src}/a -o {build}/x\ncc2", "tona-server": "cc -O2 -I{src}/a"},
        "head": {"tona-gpu-gen": "cc2\ncc -I{src}/a -o {build}/x", "tona-server": "cc -O3 -I{src}/a"},
    }

    def fake_configure(repo, rev, dest):
        (dest / "build").mkdir(parents=True)
        return dest / "build"

    def fake_sh(cmd, cwd):
        name, target = ("base" if "/base/" in cmd else "head"), cmd.rsplit(" ", 1)[-1]
        build = cmd.split("-C ")[1].split(" ")[0]
        return commands[name][target].format(src=str(Path(build).parent / "src"), build=build)

    monkeypatch.setattr(cm, "configure", fake_configure)
    monkeypatch.setattr(cm, "sh", fake_sh)
    assert cm.changed_binaries(tmp_path, "b", "h") == ["tona-server"]
    assert not list(tmp_path.iterdir())  # scratch directories are removed


def test_select_cmake_configures_from_the_archive(tmp_path, monkeypatch):
    import select_cmake as cm

    seen, real_sh = [], cm.sh
    monkeypatch.setattr(cm, "sh", lambda cmd, cwd: seen.append(cmd) or "")
    build = cm.configure(tmp_path / "repo", "abc", tmp_path / "d")
    assert build == tmp_path / "d" / "build" and (tmp_path / "d/src/work").is_symlink()
    assert "git archive abc" in seen[0] and "cmake -S" in seen[1] and "toolchain/env.sh" in seen[1]
    with pytest.raises(sg.GraphError):
        real_sh("exit 3", tmp_path)


@needs_status_shell
def test_the_status_script_refuses_to_run_scripts_from_a_checkout_ahead_of_the_base(status_repo, tmp_path):
    write(status_repo, "engine/gpu/x.h", "x")
    commit_all(status_repo)
    ran = tmp_path / "ran"
    done = run_status(status_repo, ran)
    assert done.returncode == 2 and "not trusted" in done.stderr
    assert not ran.exists()


@needs_status_shell
def test_the_status_script_checks_a_pull_request_head_with_the_scripts_of_the_base(status_repo, tmp_path):
    write(status_repo, "engine/gpu/x.h", "x")
    commit_all(status_repo)
    head = git(status_repo, "rev-parse", "HEAD").strip()
    git(status_repo, "checkout", "-q", "main")
    ran = tmp_path / "ran"
    done = run_status(status_repo, ran, "missing", "--head", head)
    assert done.returncode == 0, done.stderr
    assert f"Status gpu-golden on {head}" in done.stdout
    assert len(ran.read_text().splitlines()) == 5
    assert run_status(status_repo, ran, "missing", "--head", "nonexistent").returncode == 2
