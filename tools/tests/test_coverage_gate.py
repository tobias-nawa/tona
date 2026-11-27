import json

import coverage_gate
import pytest
from conftest import commit_all, write

FLOOR = {"cpp": {"lines": 40.0}, "python": {"lines": 70.0}}


def cobertura(path, valid, covered, branches=(0, 0)):
    path.write_text(
        f'<coverage lines-valid="{valid}" lines-covered="{covered}" '
        f'branches-valid="{branches[0]}" branches-covered="{branches[1]}"></coverage>'
    )
    return path


def test_line_and_branch_percent(tmp_path):
    xml = cobertura(tmp_path / "c.xml", 200, 50, (10, 4))
    assert coverage_gate.line_percent(xml) == 25.0
    assert coverage_gate.branch_percent(xml) == 40.0
    empty = cobertura(tmp_path / "e.xml", 0, 0)
    assert coverage_gate.line_percent(empty) == 0.0 and coverage_gate.branch_percent(empty) is None


def test_floor_value_is_the_measurement_cut_to_one_decimal_minus_the_margin():
    assert coverage_gate.floor_value(47.38) == 46.8
    assert coverage_gate.floor_value(47.3) == 46.8  # no float noise
    assert coverage_gate.floor_value(31.83) == 31.3


def test_read_floor():
    assert coverage_gate.read_floor(json.dumps(FLOOR)) == {"cpp": 40.0, "python": 70.0}


def test_total_gate_passes_at_the_floor():
    problems, notes = coverage_gate.judge_total({"cpp": 40.0, "python": 70.0}, {"cpp": 40.0, "python": 70.0}, None)
    assert problems == [] and notes == []


def test_total_gate_fails_below_the_floor():
    problems, _ = coverage_gate.judge_total({"cpp": 39.9, "python": 80.0}, {"cpp": 40.0, "python": 70.0}, None)
    assert len(problems) == 1 and problems[0].startswith("cpp: line coverage 39.90 % is below the floor 40.0 %")


def test_total_gate_asks_to_raise_the_floor_from_one_and_a_half_points_up():
    _, notes = coverage_gate.judge_total({"cpp": 41.4, "python": 71.5}, {"cpp": 40.0, "python": 70.0}, None)
    assert notes == ["python: 71.50 % is above the floor; set it to 71.0"]


def test_lowering_the_floor_fails_against_the_base():
    floor, base = {"cpp": 35.0, "python": 70.0}, {"cpp": 40.0, "python": 70.0}
    problems, _ = coverage_gate.judge_total({"cpp": 36.0, "python": 70.0}, floor, base)
    assert problems == ["cpp: the floor drops from 40.0 % to 35.0 % in tools/gates/coverage-floor.json"]
    problems, _ = coverage_gate.judge_total({"cpp": 41.0, "python": 70.0}, {"cpp": 41.0, "python": 70.0}, base)
    assert problems == []  # raising is fine
    assert coverage_gate.judge_total({"cpp": 36.0, "python": 70.0}, floor, {"python": 70.0})[0] == []


def test_floor_on_base_reads_the_file_of_the_base_commit(repo, monkeypatch):
    monkeypatch.setattr(coverage_gate, "ROOT", repo)
    write(repo, coverage_gate.FLOOR_FILE, json.dumps(FLOOR))
    commit_all(repo)
    assert coverage_gate.floor_on_base("HEAD") == {"cpp": 40.0, "python": 70.0}
    assert coverage_gate.floor_on_base("no-such-ref") is None


def test_summary_text():
    text = coverage_gate.summary_text(
        {"cpp": 41.234, "python": 80.0}, {"cpp": 40.0}, {"cpp": 12.5, "python": None}, "PATCH"
    )
    assert "| cpp | 41.23 % | 40.0 % | 12.5 % |" in text
    assert "| python | 80.00 % | 0.0 % | - |" in text
    assert "Changed lines (threshold 80 %)" in text and text.rstrip().endswith("PATCH")
    assert "Changed lines" not in coverage_gate.summary_text({"cpp": 1.0}, {}, {}, "")


@pytest.fixture
def gate_env(tmp_path, monkeypatch):
    """Floor file and two reports in a scratch repository; the patch gate and the base are replaced."""
    monkeypatch.setattr(coverage_gate, "ROOT", tmp_path)
    write(tmp_path, coverage_gate.FLOOR_FILE, json.dumps(FLOOR))
    monkeypatch.setattr(coverage_gate, "floor_on_base", lambda base: None)
    return tmp_path


def run_gate(root, cpp_covered, py_covered, base=None, summary=None):
    cpp = cobertura(root / "cpp.xml", 100, cpp_covered)
    py = cobertura(root / "py.xml", 100, py_covered)
    return coverage_gate.gate(cpp, py, base, root, summary)


def test_gate_green_and_red(gate_env, capsys):
    assert run_gate(gate_env, 40, 70) == 0
    assert run_gate(gate_env, 39, 70) == 1
    assert "FAIL cpp: line coverage 39.00 %" in capsys.readouterr().err


def test_gate_writes_the_summary_and_the_hint(gate_env):
    summary = gate_env / "summary.md"
    assert run_gate(gate_env, 45, 70, summary=str(summary)) == 0
    text = summary.read_text()
    assert "## Coverage (host code, no GPU)" in text and "set it to 44.5" in text


def test_gate_applies_the_patch_threshold_with_a_base(gate_env, monkeypatch):
    seen = {}

    def fake_patch(reports, base, out):
        seen["args"] = ([r.name for r in reports], base)
        return 1, "uncovered lines"

    monkeypatch.setattr(coverage_gate, "patch_gate", fake_patch)
    assert run_gate(gate_env, 50, 80, base="origin/main") == 1
    assert seen["args"] == (["cpp.xml", "py.xml"], "origin/main")
    monkeypatch.setattr(coverage_gate, "patch_gate", lambda reports, base, out: (0, "fine"))
    assert run_gate(gate_env, 50, 80, base="origin/main") == 0


def test_patch_gate_runs_diff_cover_and_returns_its_markdown(tmp_path, monkeypatch):
    commands = []

    class Done:
        returncode = 2
        stdout = "out\n"
        stderr = ""

    def fake_run(cmd, **kw):
        commands.append(cmd)
        (tmp_path / "patch.md").write_text("# diff report")
        return Done()

    monkeypatch.setattr(coverage_gate, "run", fake_run)
    code, markdown = coverage_gate.patch_gate([tmp_path / "a.xml", tmp_path / "b.xml"], "base", tmp_path)
    assert (code, markdown) == (2, "# diff report")
    cmd = commands[0]
    assert cmd[0] == "diff-cover" and "--fail-under" in cmd and cmd[cmd.index("--fail-under") + 1] == "80.0"
    assert cmd[cmd.index("--compare-branch") + 1] == "base"


def test_parse_args():
    run_args = coverage_gate.parse_args(["run", "--base", "origin/main", "--jobs", "2"])
    assert run_args.cmd == "run" and run_args.base == "origin/main" and run_args.jobs == 2
    check_args = coverage_gate.parse_args(["check", "--cpp", "a.xml", "--python", "b.xml"])
    assert str(check_args.cpp) == "a.xml" and check_args.base is None
    with pytest.raises(SystemExit):
        coverage_gate.parse_args(["check"])


def test_main_check_uses_the_given_reports(gate_env, monkeypatch):
    cobertura(gate_env / "c.xml", 10, 5)
    cobertura(gate_env / "p.xml", 10, 9)
    seen = {}
    monkeypatch.setattr(
        coverage_gate, "gate", lambda cpp, py, base, out, summary: seen.update(cpp=cpp.name, py=py.name) or 0
    )
    out = gate_env / "out"
    assert (
        coverage_gate.main(
            ["check", "--cpp", str(gate_env / "c.xml"), "--python", str(gate_env / "p.xml"), "--out", str(out)]
        )
        == 0
    )
    assert seen == {"cpp": "c.xml", "py": "p.xml"} and out.is_dir()
