"""The keys compiled into tona (engine/update/keys.h) are the ones in release/*.pub.

A change to one side alone fails here, in the pull request, not first when a release is signed (release-sign.sh).
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def compiled_key(name: str) -> str:
    text = (ROOT / "engine/update/keys.h").read_text(encoding="utf-8")
    m = re.search(rf'{name}\s*=\s*"([^"]*)"', text)
    assert m, f"{name} not found in engine/update/keys.h"
    return m.group(1)


def published_key(file: str) -> str:
    lines = (ROOT / "release" / file).read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 2 and lines[0].startswith("untrusted comment:"), f"release/{file} is not a minisign public key"
    return lines[1].strip()


def test_main_key_matches() -> None:
    assert compiled_key("kMainPublicKey") == published_key("main.pub")


def test_backup_key_matches() -> None:
    assert compiled_key("kSparePublicKey") == published_key("backup.pub")
