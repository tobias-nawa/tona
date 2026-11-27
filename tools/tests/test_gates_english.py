import sys

import english
from conftest import write


def run(monkeypatch, *files):
    monkeypatch.setattr(sys, "argv", ["english.py", *files])
    return english.main()


def test_english_text_passes(repo, monkeypatch):
    write(repo, "a.md", "The loader reads the header and checks the magic.\n")
    assert run(monkeypatch, "a.md") == 0


def test_german_text_fails_with_its_line(repo, monkeypatch, capsys):
    write(repo, "a.cpp", "// fine\n// Das ist nicht gut und wird nie gehen\nint x;\n")
    assert run(monkeypatch, "a.cpp") == 1
    assert "a.cpp:2: German text" in capsys.readouterr().err


def test_umlaut_fails(repo, monkeypatch):
    write(repo, "a.md", "Größe\n")
    assert run(monkeypatch, "a.md") == 1


def test_allow_list_of_the_audit_holds(repo, monkeypatch):
    write(repo, "scripts/bench-long-code.txt", "Größe und nicht mehr\n")
    assert run(monkeypatch, "scripts/bench-long-code.txt") == 0


def test_screens_have_no_allow_list(repo, monkeypatch):
    for rel in ("engine/client/texts_pages.inc", "installer/tui_pages.cpp", "installer/core/describe.cpp"):
        write(repo, rel, "Größe und nicht mehr\n")
        assert run(monkeypatch, rel) == 1


def test_binary_files_are_skipped(repo, monkeypatch):
    (repo / "a.bin").write_bytes(b"\0Gr\xc3\xb6\xc3\x9fe")
    assert run(monkeypatch, "a.bin") == 0


def test_allowed_paths_are_not_scanned(repo, monkeypatch):
    write(repo, "docs/a.md", "Das ist nicht gut und wird nicht gehen.\n")
    write(repo, "engine/tests/t.cpp", "// Übergröße\n")
    assert run(monkeypatch, "docs/a.md") == 1
    assert run(monkeypatch, "engine/tests/t.cpp") == 0


def test_urls_do_not_count(repo, monkeypatch):
    write(repo, "a.md", "See https://example.org/und/nicht/ist for details.\n")
    assert run(monkeypatch, "a.md") == 0
