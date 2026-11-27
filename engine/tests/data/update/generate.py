#!/usr/bin/env python3
"""Regenerates the signature test vectors of the updater tests (engine/tests/test_update_*.cpp).

The keys are TEST keys derived from fixed seeds (deterministic, public on purpose); they must never be built into a
release binary. Ed25519 comes from the `cryptography` package, BLAKE2b from hashlib; only the minisign file format is
written here.

    python3 generate.py            # rewrite the vectors next to this file
    python3 generate.py --minisign # also cross-check against the real `minisign` binary (needs it in PATH)

The cross-check creates a throwaway key with `minisign -G -W`, lets minisign sign the manifest, signs the same bytes
again with this script from the same secret key and requires byte-identical output, and has `minisign -V` verify
every valid vector made here. It also writes real/ (public key, manifest and signature made by minisign itself).
"""

import base64
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

HERE = Path(__file__).resolve().parent
COMMENT = "tona schema=1 seq=3 version=0.3.0"
SEED_MAIN = bytes(range(32))
SEED_OTHER = bytes(range(100, 132))
ID_MAIN = bytes.fromhex("0102030405060708")
ID_OTHER = bytes.fromhex("a1a2a3a4a5a6a7a8")

MANIFEST = """{
  "schema": 1,
  "product": "tona",
  "version": "0.3.0",
  "seq": 3,
  "released": "2026-11-02T10:00:00Z",
  "expires": "2027-05-02T00:00:00Z",
  "commit": "0123456789abcdef0123456789abcdef01234567",
  "rocm_lock_sha256": "89abcdef0123456789abcdef0123456789abcdef0123456789abcdef01234567",
  "severity": "normal",
  "changelog": "Test release.",
  "files": [
    {"asset": "tona-server", "path": "bin/tona-server", "platform": "linux-x86_64-gfx1200",
     "size": 1234, "sha256": "%s", "mode": "0755"}
  ]
}
""" % ("ab" * 32)


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def public_raw(key: Ed25519PrivateKey) -> bytes:
    return key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)


def public_file(key: Ed25519PrivateKey, key_id: bytes) -> str:
    return f"untrusted comment: minisign public key: {key_id.hex().upper()}\n{b64(b'Ed' + key_id + public_raw(key))}\n"


def signature(data: bytes, comment: str, key: Ed25519PrivateKey, key_id: bytes, legacy: bool = False) -> str:
    """A .minisig file; legacy=True is the old `Ed` format (signs the file itself, not its hash)."""
    message = data if legacy else hashlib.blake2b(data, digest_size=64).digest()
    sig = key.sign(message)
    head = (b"Ed" if legacy else b"ED") + key_id + sig
    global_sig = key.sign(sig + comment.encode())
    return (
        "untrusted comment: signature from minisign secret key\n"
        f"{b64(head)}\ntrusted comment: {comment}\n{b64(global_sig)}\n"
    )


def vectors() -> dict[str, str]:
    data = MANIFEST.encode()
    main = Ed25519PrivateKey.from_private_bytes(SEED_MAIN)
    other = Ed25519PrivateKey.from_private_bytes(SEED_OTHER)
    good = signature(data, COMMENT, main, ID_MAIN)
    lines = good.split("\n")
    swapped_id = b64(base64.b64decode(lines[1])[:2] + bytes(8) + base64.b64decode(lines[1])[10:])
    return {
        "key.pub": public_file(main, ID_MAIN),
        "other.pub": public_file(other, ID_OTHER),
        "manifest.json": MANIFEST,
        "valid.minisig": good,
        "wrong_key.minisig": signature(data, COMMENT, other, ID_MAIN),
        "unknown_id.minisig": "\n".join([lines[0], swapped_id, *lines[2:]]),
        "tampered_comment.minisig": good.replace("seq=3", "seq=9"),
        "comment_mismatch.minisig": signature(data, "tona schema=1 seq=4 version=0.3.0", main, ID_MAIN),
        "legacy_ed.minisig": signature(data, COMMENT, main, ID_MAIN, legacy=True),
        "truncated.minisig": good[:-20],
        "extra_line.minisig": good + "extra\n",
        "blank_line.minisig": good + "\n",
        "crlf.minisig": good.replace("\n", "\r\n"),
        "too_long.minisig": good.replace("signature from minisign secret key", "x" * 5000),
    }


def write_all() -> None:
    for name, text in vectors().items():
        (HERE / name).write_text(text, newline="")


def unencrypted_secret(path: Path) -> tuple[bytes, bytes]:
    """(key id, 32-byte seed) from a secret key made with `minisign -G -W`."""
    raw = base64.b64decode(path.read_text().splitlines()[1])
    keynum = raw[2 + 2 + 2 + 32 + 8 + 8 :]  # algorithms, salt, opslimit, memlimit, then key id + sk + pk + checksum
    return keynum[:8], keynum[8:40]


def minisign(*args: str, cwd: Path) -> None:
    subprocess.run(["minisign", *args], cwd=cwd, check=True, capture_output=True)


def cross_check() -> None:
    if not shutil.which("minisign"):
        sys.exit("minisign is not installed")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "manifest.json").write_text(MANIFEST)
        minisign("-G", "-W", "-p", "real.pub", "-s", "real.key", cwd=work)
        minisign("-S", "-s", "real.key", "-m", "manifest.json", "-t", COMMENT, cwd=work)
        key_id, seed = unencrypted_secret(work / "real.key")
        mine = signature(MANIFEST.encode(), COMMENT, Ed25519PrivateKey.from_private_bytes(seed), key_id)
        real = (work / "manifest.json.minisig").read_text()
        # Only the untrusted comment may differ; the signed parts must be byte-identical
        assert mine.split("\n")[1:] == real.split("\n")[1:], "own signature differs from minisign's"
        minisign("-V", "-H", "-p", "real.pub", "-m", "manifest.json", cwd=work)
        out = HERE / "real"
        out.mkdir(exist_ok=True)
        for name in ("real.pub", "manifest.json", "manifest.json.minisig"):
            shutil.copy(work / name, out / name)
        # The vectors made here are accepted by the real verifier too
        (work / "own.minisig").write_text(vectors()["valid.minisig"])
        (work / "own.pub").write_text(vectors()["key.pub"])
        minisign("-V", "-H", "-p", "own.pub", "-x", "own.minisig", "-m", "manifest.json", cwd=work)


if __name__ == "__main__":
    write_all()
    if "--minisign" in sys.argv:
        cross_check()
