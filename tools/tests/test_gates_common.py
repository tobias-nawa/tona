import common
import pytest
from conftest import commit_all, write


def test_tracked_files_filters_suffix_and_vendored_code(repo):
    write(repo, "engine/a.cpp")
    write(repo, "engine/a.h")
    write(repo, "engine/notes.txt")
    write(repo, "engine/third_party/lib.h")
    write(repo, ".gitignore", "ignored.cpp\n")
    write(repo, "ignored.cpp")
    commit_all(repo)
    write(repo, "new.py")  # untracked but not ignored counts
    assert common.tracked_files(common.CPP_SUFFIXES) == ["engine/a.cpp", "engine/a.h"]
    assert common.tracked_files(common.SCRIPT_SUFFIXES) == ["new.py"]


def test_gate_files_without_arguments_takes_everything(repo):
    write(repo, "x.cpp")
    commit_all(repo)
    assert common.gate_files([], common.CPP_SUFFIXES) == (["x.cpp"], False)


def test_gate_files_with_arguments_keeps_only_gated_existing_files(repo):
    write(repo, "src/x.cpp")
    write(repo, "src/y.txt")
    write(repo, "engine/third_party/z.cpp")
    files, explicit = common.gate_files(
        ["src/x.cpp", "src/y.txt", "src/missing.cpp", "engine/third_party/z.cpp", "src/x.cpp"], common.CPP_SUFFIXES
    )
    assert (files, explicit) == (["src/x.cpp"], True)


def test_gate_files_accepts_absolute_paths(repo):
    path = write(repo, "src/x.py")
    assert common.gate_files([str(path)], (".py",)) == (["src/x.py"], True)


def test_fail_if_empty_exits_with_2(capsys):
    common.fail_if_empty(["a"], "gate")
    with pytest.raises(SystemExit) as stop:
        common.fail_if_empty([], "gate")
    assert stop.value.code == 2
    assert "gate: no files found" in capsys.readouterr().err


def test_report(capsys):
    assert common.report("gate", [], 3, "hint") == 0
    assert capsys.readouterr().out == "gate: ok (3 checked)\n"
    assert common.report("gate", ["a: bad", "b: bad"], 3, "Fix it.") == 1
    err = capsys.readouterr().err
    assert "a: bad\nb: bad\n" in err and "gate: 2 violation(s). Fix it." in err
