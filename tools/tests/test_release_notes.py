"""scripts/release_notes.py: the notes of a version come from CHANGELOG.md, a missing section stops the release."""

import json
import re

import pytest
from conftest import ROOT, load_script

rn = load_script("scripts/release_notes.py")
rm = load_script("scripts/release_manifest.py")

CHANGELOG = """# Changelog

## [Unreleased]

Nothing yet.

## [0.2.0] - 2026-12-01

### Changed

- Breaking: the setting `port` moved.

## [0.1.0] - 2026-11-30

### Added

- First release.
"""


def test_section_of_a_version():
    assert rn.notes_of(CHANGELOG, "0.2.0") == "### Changed\n\n- Breaking: the setting `port` moved."
    assert rn.notes_of(CHANGELOG, "0.1.0") == "### Added\n\n- First release."


def test_missing_or_empty_section_fails():
    with pytest.raises(rn.NotesError, match="no section"):
        rn.notes_of(CHANGELOG, "0.3.0")
    with pytest.raises(rn.NotesError, match=r"not X\.Y\.Z"):
        rn.notes_of(CHANGELOG, "0.2")  # a prefix of another version is not a version
    with pytest.raises(rn.NotesError, match="empty"):
        rn.notes_of("## [0.1.0] - 2026-11-30\n\n## [0.0.9] - 2026-11-01\n- x\n", "0.1.0")


def test_limits_of_the_manifest_parser():
    with pytest.raises(rn.NotesError, match="bytes"):
        rn.notes_of("## [0.1.0] - 2026-11-30\n" + "x" * 20000 + "\n", "0.1.0")
    with pytest.raises(rn.NotesError, match="control"):
        rn.notes_of("## [0.1.0] - 2026-11-30\n- a\x1b[2J\n", "0.1.0")


def test_command_line(tmp_path, capsys, monkeypatch):
    file = tmp_path / "CHANGELOG.md"
    file.write_text(CHANGELOG, encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["release_notes.py", "0.1.0", "--file", str(file)])
    assert rn.main() == 0
    assert capsys.readouterr().out == "### Added\n\n- First release.\n"
    monkeypatch.setattr("sys.argv", ["release_notes.py", "9.9.9", "--file", str(file)])
    assert rn.main() == 1


def test_the_notes_end_up_in_the_manifest(tmp_path):
    """The path release-build.sh takes: section -> --changelog -> the manifest field that tona shows."""
    asset = tmp_path / "tona-server"
    asset.write_bytes(b"binary\n")
    notes = rn.notes_of(CHANGELOG, "0.2.0")
    args = rm.build_parser().parse_args(
        [
            "create",
            "--dir",
            str(tmp_path),
            "--output",
            str(tmp_path / "manifest.json"),
            "--version",
            "0.2.0",
            "--seq",
            "2",
            "--expires-days",
            "180",
            "--source-date-epoch",
            "1793836800",
            "--commit",
            "c" * 40,
            "--lock-sha256",
            "8" * 64,
            "--changelog",
            notes,
            "--file",
            "tona-server:bin/tona-server:0755:linux-x86_64-gfx1200",
        ]
    )
    args.run(args)
    assert json.loads((tmp_path / "manifest.json").read_text())["changelog"] == notes


def test_the_real_changelog_has_the_versioning_statement_and_valid_notes():
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "## [Unreleased]" in text and "Semantic Versioning" in text and "0.Y.0" in text
    # The newest released section is what the next build signs; before the first release the draft sits in
    # [Unreleased], so that one is checked as if it were released.
    newest = re.search(r"^## \[(\d+\.\d+\.\d+)\] - \d{4}-\d{2}-\d{2}$", text, re.M)
    if newest:
        version = newest.group(1)
    else:
        version, text = "0.1.0", text.replace("## [Unreleased]", "## [0.1.0] - 2026-12-01", 1)
    assert "### Added" in rn.notes_of(text, version)  # fits the 16 KiB limit, no control characters
