import shutil
import sys

import functions
import pytest
from conftest import write

needs_lizard = pytest.mark.skipif(shutil.which("lizard") is None, reason="lizard is not installed")


def test_violations_at_the_limits():
    # nloc, ccn, tokens, params, length, location, file, name, long_name, start, end
    row = ["0", "15", "0", "6", "80"] + [""] * 6
    assert functions.violations(row) == []
    row = ["0", "16", "0", "7", "81"] + [""] * 6
    assert functions.violations(row) == ["length 81 > 80", "ccn 16 > 15", "params 7 > 6"]


def run(monkeypatch, *files):
    monkeypatch.setattr(sys, "argv", ["functions.py", *files])
    return functions.main()


@needs_lizard
def test_measure_without_files_runs_nothing(repo):
    assert functions.measure([], "cpp") == []


@needs_lizard
def test_python_function_with_too_many_parameters(repo, monkeypatch, capsys):
    write(repo, "a.py", "def f(a, b, c, d, e, f, g):\n    return a\n")
    assert run(monkeypatch, "a.py") == 1
    assert "params 7 > 6" in capsys.readouterr().err


@needs_lizard
def test_cpp_function_too_long_and_too_complex(repo, monkeypatch, capsys):
    body = "".join(f"    if (x == {i}) x += {i};\n" for i in range(20))
    write(repo, "a.cpp", "int f(int x) {\n" + body + "    return x;\n}\n")
    assert run(monkeypatch, "a.cpp") == 1
    assert "ccn 21 > 15" in capsys.readouterr().err


@needs_lizard
def test_clean_files_pass_and_hip_counts_as_cpp(repo, monkeypatch):
    write(repo, "a.py", "def f(a):\n    return a\n")
    write(repo, "k.hip", "int g(int x) { return x; }\n")
    assert run(monkeypatch, "a.py", "k.hip") == 0
