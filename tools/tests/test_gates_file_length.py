import sys

import file_length
import pytest
from conftest import write


def run(monkeypatch, *files):
    monkeypatch.setattr(sys, "argv", ["file_length.py", *files])
    return file_length.main()


def test_limit_is_500_lines_inclusive(repo, monkeypatch, capsys):
    write(repo, "ok.cpp", "x\n" * 500)
    write(repo, "long.cpp", "x\n" * 501)
    assert run(monkeypatch, "ok.cpp") == 0
    assert run(monkeypatch, "long.cpp") == 1
    assert "long.cpp: 501 lines (limit 500)" in capsys.readouterr().err


def test_python_and_shell_files_count_too(repo, monkeypatch):
    write(repo, "a.py", "x = 1\n" * 600)
    write(repo, "b.sh", "true\n" * 600)
    assert run(monkeypatch, "a.py", "b.sh") == 1


def test_other_files_are_ignored(repo, monkeypatch):
    write(repo, "data.txt", "x\n" * 900)
    assert run(monkeypatch, "data.txt") == 0


def test_without_arguments_an_empty_repository_is_an_error(repo, monkeypatch):
    with pytest.raises(SystemExit) as stop:
        run(monkeypatch)
    assert stop.value.code == 2


def test_without_arguments_checks_all_tracked_files(repo, monkeypatch):
    write(repo, "a.cpp", "x\n" * 501)
    assert run(monkeypatch) == 1
