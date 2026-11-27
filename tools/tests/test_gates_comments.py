import sys

import comments
import pytest
from conftest import write


def run(monkeypatch, *files):
    monkeypatch.setattr(sys, "argv", ["comments.py", *files])
    return comments.main()


def problems(name, source):
    return comments.check_file(name, source)


@pytest.mark.parametrize("text", ["// TODO fix this", "// FIXME: later", "/* TODO */", "// TODO(#x) bad"])
def test_todo_without_issue_number_fails(text):
    assert any("TODO" in p for p in problems("a.cpp", text + "\n"))


def test_todo_with_issue_number_passes():
    assert problems("a.cpp", "// TODO(#48) split this\n") == []
    assert problems("a.py", "x = 1  # TODO(#7): later\n") == []


def test_todo_in_a_string_is_not_a_comment():
    assert problems("a.cpp", 'const char* s = "// TODO";\n') == []
    assert problems("a.py", 's = "# TODO"\n') == []
    assert problems("a.cpp", 'auto r = R"x(// TODO)x";\n') == []


@pytest.mark.parametrize(
    "line",
    ["// ----------", "// ==== Setup ====", "// --- Models ---", "/* ***** */", "#########"],
)
def test_banners_fail(line):
    assert any("banner" in p for p in problems("a.cpp" if line[0] != "#" else "a.sh", line + "\n"))


def test_a_dash_inside_prose_is_no_banner():
    assert problems("a.cpp", "// a - b, then c -- d\n") == []
    assert problems("a.cpp", "// -1 means none\n") == []


@pytest.mark.parametrize(
    "line",
    [
        "// int x = foo(3);",
        "// return a + b;",
        "// for (int i = 0; i < n; i++) {",
        "// if (x) {",
        "// } else {",
        "// #include <vector>",
        "// out.push_back(x);",
        "    // y += 2;",
    ],
)
def test_cpp_commented_out_code_fails(line):
    assert any("commented-out code" in p for p in problems("a.cpp", line + "\n"))


@pytest.mark.parametrize(
    "line",
    [
        "// Per stream c: gate = sigmoid(s) with s = norm(k);",
        "// the value is x = y; see the paper",
        "// for channel c = base[c + d] · t;",
        "// Returns the index of the first set bit.",
        "// (see foo(x) for the layout)",
        "// if the file is missing the loader returns an error",
        "// a == b is checked below;",
    ],
)
def test_cpp_prose_passes(line):
    assert problems("a.cpp", line + "\n") == []


def test_trailing_comment_is_not_checked_for_code():
    assert problems("a.cpp", "int x = 1;  // y = foo(3);\n") == []


def test_block_comment_lines_are_checked_one_by_one():
    src = "/*\n * explains why\n * x = foo(2);\n */\nint y;\n"
    assert [p.split(":")[1] for p in problems("a.cpp", src)] == ["3"]


@pytest.mark.parametrize(
    "line",
    [
        "# x = foo(3)",
        "# import os",
        "# return x",
        "# if x:",
        "# print(x)",
        "# self.a += 1",
        "    # raise ValueError('x')",
    ],
)
def test_python_commented_out_code_fails(line):
    assert any("commented-out code" in p for p in problems("a.py", line + "\n"))


@pytest.mark.parametrize(
    "line",
    [
        "# noqa: E501",
        "# type: ignore",
        "# Note: this is the first row",
        "# the loop runs once per layer",
        "# 100 ms: the longest stall",
        "# x",
        "#!/usr/bin/env python3",
        "# fmt: off",
        "# e.g. foo(x)",
    ],
)
def test_python_prose_and_directives_pass(line):
    assert problems("a.py", line + "\n") == []


@pytest.mark.parametrize("line", ["# if [ -f x ]; then", "# done", "# FOO=$(date)", "# for f in *.c; do"])
def test_shell_commented_out_code_fails(line):
    assert any("commented-out code" in p for p in problems("a.sh", line + "\n"))


def test_shell_prose_and_shebang_pass():
    src = "#!/bin/bash\n# Usage: fuzz.sh [seconds]\n# then the corpus is copied\necho '# not a comment'\n"
    assert problems("a.sh", src) == []


def test_command_line_runs_on_the_given_files(repo, monkeypatch, capsys):
    write(repo, "ok.cpp", "// why: the table is sorted\nint x;\n")
    write(repo, "bad.cpp", "// TODO later\n")
    assert run(monkeypatch, "ok.cpp") == 0
    assert run(monkeypatch, "bad.cpp") == 1
    assert "bad.cpp:1: TODO/FIXME" in capsys.readouterr().err
