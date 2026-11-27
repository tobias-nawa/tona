"""licenses/ and the licence section of scripts/release-build.sh: what the release's LICENSES.txt must contain."""

import hashlib
import re

import pytest
from conftest import ROOT

LICENSES = ROOT / "licenses"
# Vendored in the order release-build.sh concatenates them (after overview.txt)
VENDORED = ["llvm-LICENSE.TXT", "llvm-third-party.txt", "rocm-extra.txt", "rocm-sysdeps.txt"]
SCRIPT = (ROOT / "scripts/release-build.sh").read_text()

# One marker per licence that the shipped binaries make mandatory
MARKERS = [
    "Apache License",
    "LLVM Exceptions",
    "University of Illinois/NCSA",
    "Igor Sysoev",
    "Serge Lamikhov-Center",
    "Google Inc.",
    "Henry Spencer",
    "Unicode, Inc.",
    "Yann Collet",
    "Victor Zverovich",
]


def assembled() -> str:
    """The part of LICENSES.txt that comes from the repository (the wheel files follow it in the release)."""
    return "\n".join((LICENSES / name).read_text() for name in ["overview.txt", *VENDORED])


@pytest.mark.parametrize("marker", MARKERS)
def test_marker_present(marker):
    assert marker in assembled()


def test_no_gpl3_text():
    text = assembled()
    assert "GNU GENERAL PUBLIC LICENSE\n                       Version 3" not in text
    assert not re.search(r"GNU GENERAL PUBLIC LICENSE\s+Version 3", text)


def test_llvm_license_is_the_upstream_file():
    digest = hashlib.sha256((LICENSES / "llvm-LICENSE.TXT").read_bytes()).hexdigest()
    assert digest == "8d85c1057d742e597985c7d4e6320b015a9139385cff4cbae06ffc0ebe89afee"


def test_every_excerpt_names_source_and_pin():
    for name in ("llvm-third-party.txt", "rocm-extra.txt"):
        text = (LICENSES / name).read_text()
        for header in re.findall(r"^Source: (.+)$", text, re.M):
            assert re.match(r"https://raw\.githubusercontent\.com/.+/[0-9a-f]{40}/", header), header
        assert text.count("Source: ") == text.count("Included in: ") > 0


def test_overview_names_every_vendored_file():
    overview = (LICENSES / "overview.txt").read_text()
    for name in VENDORED[:-1]:
        assert f"licenses/{name}" in overview
    assert "licenses/rocm-sysdeps.txt" in overview


def test_script_uses_vendored_files_and_not_the_wheel_notices():
    section = SCRIPT.split("license=rocm-")[1].split("chmod")[0]
    for name in VENDORED:
        assert f"licenses/{name}" in section
    assert '"$src/licenses/overview.txt"' in section
    assert "NOTICES.txt" not in section
    assert "llvm/Support/LICENSE.TXT" not in section
    assert "curl" not in SCRIPT
