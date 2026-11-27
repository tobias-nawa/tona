import sys

import measurement_privacy
import pytest
from conftest import write


def run(monkeypatch, *files):
    monkeypatch.setattr(sys, "argv", ["measurement_privacy.py", *files])
    return measurement_privacy.main()


def test_top_field_is_reported_with_line(repo, monkeypatch, capsys):
    write(repo, "research/measurements/x-logs/r8.jsonl", '{"a": 1}\n{"a": 2, "top": ["p:1"]}\n')
    assert run(monkeypatch, "research/measurements/x-logs/r8.jsonl") == 1
    assert 'r8.jsonl:2: field "top"' in capsys.readouterr().err


def test_clean_log_passes(repo, monkeypatch):
    write(repo, "research/measurements/x-logs/r8.jsonl", '{"a": 1, "load": 0.5}\nnot json\n')
    assert run(monkeypatch, "research/measurements/x-logs/r8.jsonl") == 0


def test_other_directories_are_ignored(repo, monkeypatch):
    write(repo, "bench/out.jsonl", '{"top": 1}\n')
    assert run(monkeypatch, "bench/out.jsonl") == 0


def test_without_arguments_checks_tracked_files(repo, monkeypatch):
    write(repo, "research/measurements/a.jsonl", '{"top": []}\n')
    assert run(monkeypatch) == 1


def test_without_arguments_and_no_logs_is_an_error(repo, monkeypatch):
    with pytest.raises(SystemExit) as stop:
        run(monkeypatch)
    assert stop.value.code == 2


def test_private_home_path_is_reported(repo, monkeypatch, capsys):
    write(repo, "research/measurements/x-logs/a.out", "HIP_PATH=/home/alice/Projects/tona/work/rocm\n")
    assert run(monkeypatch, "research/measurements/x-logs/a.out") == 1
    assert "a.out:1: private home path /home/alice/" in capsys.readouterr().err


def test_mac_home_path_in_docs_is_reported(repo, monkeypatch):
    write(repo, "docs/use/x.md", "see /Users/bob/Library/LaunchAgents\n")
    assert run(monkeypatch, "docs/use/x.md") == 1


def test_placeholders_and_repo_marker_pass(repo, monkeypatch):
    write(repo, "docs/use/y.md", "/home/u/.local/bin and /Users/x/.local/share/tona, <repo>/work, ~/.config\n")
    assert run(monkeypatch, "docs/use/y.md") == 0


def test_test_home_folder_named_home_passes(repo, monkeypatch):
    write(repo, "research/measurements/m.md", "TONA_INSTALL_DIR=work/accept/home/.local/share/tona\n")
    assert run(monkeypatch, "research/measurements/m.md") == 0


def test_agent_scratchpad_path_is_reported(repo, monkeypatch):
    write(repo, "research/n.md", "ran /tmp/claude-1000/-home-a-p/0f0e/scratchpad/x.py\n")
    assert run(monkeypatch, "research/n.md") == 1
