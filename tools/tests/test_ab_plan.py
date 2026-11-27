"""The plan duty of scripts/ab.py: no start without an expectation, plan block, check against the expectation."""

import subprocess

import ab
import ab_plan
import ab_report
import ab_workloads
import pytest
from ab_stats import Plan


def test_parse_expect_reads_ranges_with_signs_and_decimals():
    assert ab_plan.parse_expect(["decode:-0.5..+0.5%", "prefill@30000: 1.25..2%"]) == [
        ("decode", -0.5, 0.5),
        ("prefill@30000", 1.25, 2.0),
    ]


@pytest.mark.parametrize("bad", ["decode", "decode:1", "decode:2..1%", ":1..2%", "decode:a..b%"])
def test_parse_expect_rejects_malformed(bad):
    with pytest.raises(SystemExit):
        ab_plan.parse_expect([bad])


def test_exact_name_beats_group_and_group_covers_bench_set():
    expect = [("decode", -1, 1), ("decode@130000", -5, -3)]
    assert ab_plan.expectation_for("decode@30000", expect) == (-1, 1)
    assert ab_plan.expectation_for("decode@130000", expect) == (-5, -3)
    assert ab_plan.expectation_for("decode-code", expect) == (-1, 1)
    assert ab_plan.expectation_for("prefill@30000", expect) is None


def test_start_is_refused_without_expectation_or_source():
    metrics = ab_workloads.metric_names(ab_workloads.jobs_for(["mid"], [], []))
    with pytest.raises(SystemExit, match="no --expect for"):
        ab_plan.require_plan(metrics, [("decode", -1, 1)], "x")
    with pytest.raises(SystemExit, match="source"):
        ab_plan.require_plan(metrics, [("decode", -1, 1), ("prefill", -1, 1)], " ")
    ab_plan.require_plan(metrics, [("decode", -1, 1), ("prefill", -1, 1)], "A/A of 2026-11-01")


def test_main_refuses_to_start_without_expectation(tmp_path):
    with pytest.raises(SystemExit):
        ab.main(["--a", "x", "--b", "y", "--out", str(tmp_path / "o")])


def test_tooling_lists_only_diagnostic_variables():
    env = {"TONA_DIAG": "1", "OMP_NUM_THREADS": "3", "HSA_X": "y", "PATH": "/bin", "HOME": "/h"}
    assert ab_plan.tooling(env) == {"HSA_X": "y", "TONA_DIAG": "1", "OMP_NUM_THREADS": "3"}


def test_build_commit_unknown_without_cache(tmp_path):
    assert ab_plan.build_commit(tmp_path) == "unknown"


def test_build_commit_describes_the_source_tree(repo, tmp_path):
    from conftest import commit_all, write

    write(repo, "a.txt", "x")
    commit_all(repo)
    build = tmp_path / "build"
    build.mkdir()
    (build / "CMakeCache.txt").write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={repo}\n")
    assert ab_plan.build_commit(build).endswith("(source HEAD at run time)")


def test_build_commit_reads_the_stamp_of_the_build(repo, tmp_path):
    from conftest import commit_all, write

    write(repo, "a.txt", "x")
    commit_all(repo)
    build = tmp_path / "build"
    build.mkdir()
    (build / "CMakeCache.txt").write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={repo}\n")
    (build / "tona-server.commit").write_text("abc1234567-dirty\n")
    write(repo, "b.txt", "y")  # the source moves on after the build
    commit_all(repo)
    assert ab_plan.build_commit(build) == "abc1234567-dirty"


def test_commit_stamp_script_writes_git_describe(repo, tmp_path):
    import shutil

    from conftest import ROOT, commit_all, write

    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("no cmake")
    write(repo, "a.txt", "x")
    commit_all(repo)
    out = tmp_path / "tona-server.commit"
    script = ROOT / "engine/cmake/commit_stamp.cmake"
    subprocess.run([cmake, f"-DSRC={repo}", f"-DOUT={out}", "-P", str(script)], check=True)
    clean = out.read_text().strip()
    assert clean and "dirty" not in clean
    write(repo, "a.txt", "changed")
    subprocess.run([cmake, f"-DSRC={repo}", f"-DOUT={out}", "-P", str(script)], check=True)
    assert out.read_text().strip() == clean + "-dirty"


def info():
    side = {"build": "merged", "commit": "abc", "tooling": {"OMP_NUM_THREADS": "6"}}
    return {
        "expect": [("decode", -1, 1), ("prefill", -1, 1)],
        "source": "A/A of 2026-11-01",
        "workload": "server",
        "why_size": "long context is where the change acts",
        "decode_tokens": 200,
        "targets": "30000",
        "ctx": 140000,
        "A": side,
        "B": {**side, "tooling": {"OMP_NUM_THREADS": "3"}},
    }


def test_plan_block_has_scope_expectation_and_tooling():
    text = ab_plan.plan_markdown(info(), Plan(), 600.0)
    assert "estimate before round 1" in text and "20-60 min" in text
    assert "decode: -1 to +1 %" in text and "A/A of 2026-11-01" in text
    assert "B: build commit abc; OMP_NUM_THREADS=3" in text
    assert (
        "Why this size: long context is where the change acts (decode tokens per prompt as in the targets, default 200"
        in text
    )
    assert "Known noise: none" in text


def test_estimate_grows_with_workloads():
    def est(*names):
        return ab_plan.estimate_round_seconds(ab_workloads.jobs_for(list(names), [], []))

    assert est("mid", "long") > est("mid") > est("short") > 0
    assert est("set") > est("short")
    assert est("huge") > 2400  # 512K + 1M prefill on both arms: about 45 min per round


def test_report_marks_results_outside_the_expectation():
    def measure(key):
        f = 0.9 if key == "B" else 1.0  # -10 %
        return {"decode@30000": 100 * f, "prefill@30000": 1000.0}, {"30000": "s"}

    rounds = []
    ab.run_rounds(measure, Plan(max_rounds=2), rounds)
    side = info()["A"]
    config = {"A": side | {"env": {}}, "B": side | {"env": {}}, "deterministic": False, "plan_info": info()}
    data = {"config": config, "rounds": rounds, "stop_reason": "t", "wall_seconds": 60.0}
    text = ab_report.markdown(data, Plan(max_rounds=2))
    assert "| decode@30000 | -10.00 % | -1 to +1 % | **OUTSIDE the expectation: finding** |" in text
    assert "| prefill@30000 | +0.00 % | -1 to +1 % | within the expectation |" in text


def test_main_refuses_to_start_without_a_reason_for_the_size(tmp_path):
    argv = ["--a", "x", "--b", "y", "--workload", "mid", "--expect", "decode:-1..1%", "--expect", "prefill:-1..1%"]
    with pytest.raises(SystemExit, match="why-size"):
        ab.main([*argv, "--expect-source", "t", "--out", str(tmp_path / "o")])
