"""The local rebuild proof: release-verify-build.sh writes it, release-sign.sh refuses to sign without it."""

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from conftest import ROOT

TAG = "v0.0.1"
SUMS = "{}  tona-server\n".format(hashlib.sha256(b"server\n").hexdigest())
# Stands in for gh: "release download" copies the draft, everything else (the attestation check) fails with a marker
FAKE_GH = """#!/bin/sh
if [ "$1 $2" = "release download" ]; then
    while [ $# -gt 0 ]; do [ "$1" = --dir ] && out=$2; shift; done
    cp "$FAKE_DRAFT"/* "$out"/
    exit 0
fi
echo "FAKE-GH-REACHED-ATTESTATION" >&2
exit 1
"""
# The container has neither diff nor minisign; the scripts only check that they exist (the tests never sign)
FAKE_MINISIGN = """#!/bin/sh
exit 0
"""
FAKE_DIFF = """#!/usr/bin/env python3
import sys

sys.exit(0 if open(sys.argv[-2], "rb").read() == open(sys.argv[-1], "rb").read() else 1)
"""
FAKE_BUILD = """#!/bin/sh
mkdir -p "$1"
cp "$FAKE_LOCAL_SUMS" "$1/SHA256SUMS"
"""


def run(cmd: list, env: dict, cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True, check=False)


def git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def proof_env(tmp_path: Path):
    """A throw-away checkout with the release scripts, a tag, a fake draft and a proof directory."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    for name in ("release-verify-build.sh", "release-sign.sh", "release-common.sh", "release_manifest.py"):
        shutil.copy(ROOT / "scripts" / name, repo / "scripts" / name)
    (repo / "scripts/release-build.sh").write_text(FAKE_BUILD)
    (repo / "scripts/release-build.sh").chmod(0o755)
    (repo / "release").mkdir()
    shutil.copy(ROOT / "release/main.pub", repo / "release/main.pub")
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "x")
    git(repo, "tag", TAG)
    commit = git(repo, "rev-parse", "HEAD")

    draft = tmp_path / "draft"
    draft.mkdir()
    (draft / "tona-server").write_bytes(b"server\n")
    (draft / "SHA256SUMS").write_text(SUMS)
    (draft / "manifest.json").write_text(
        json.dumps({"version": "0.0.1", "commit": commit, "files": [{"asset": "tona-server"}]})
    )
    local_sums = tmp_path / "local-sums"
    local_sums.write_text(SUMS)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "gh").write_text(FAKE_GH)
    (bindir / "gh").chmod(0o755)
    for name, body in (("minisign", FAKE_MINISIGN), ("diff", FAKE_DIFF)):
        (bindir / name).write_text(body)
        (bindir / name).chmod(0o755)
    proofs = tmp_path / "state"
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "TONA_RELEASE_PROOF_DIR": str(proofs),
        "FAKE_DRAFT": str(draft),
        "FAKE_LOCAL_SUMS": str(local_sums),
    }
    return {
        "repo": repo,
        "draft": draft,
        "proofs": proofs,
        "env": env,
        "commit": commit,
        "local_sums": local_sums,
        "tmp": tmp_path,
    }


def verify(p: dict, ref: Path | None = None) -> subprocess.CompletedProcess:
    cmd = [
        "bash",
        str(p["repo"] / "scripts/release-verify-build.sh"),
        TAG,
        "--local",
        str(ref or p["draft"]),
        "--work",
        str(p["tmp"] / "work"),
    ]
    return run(cmd, p["env"], cwd=p["repo"])


def sign(p: dict) -> subprocess.CompletedProcess:
    cmd = [
        "bash",
        str(p["repo"] / "scripts/release-sign.sh"),
        TAG,
        "--seq",
        "1",
        "--dir",
        str(p["tmp"] / "dl"),
        "--pubkey",
        str(p["repo"] / "release/main.pub"),
    ]
    (p["tmp"] / "dl").mkdir(exist_ok=True)
    return run(cmd, p["env"], cwd=p["repo"])


def test_match_writes_proof(proof_env) -> None:
    r = verify(proof_env)
    assert r.returncode == 0, r.stderr
    lines = (proof_env["proofs"] / f"{TAG}.proof").read_text().splitlines()
    assert lines == [
        f"tag={TAG}",
        f"commit={proof_env['commit']}",
        "sha256sums=" + hashlib.sha256(SUMS.encode()).hexdigest(),
    ]


def test_mismatch_writes_nothing_and_removes_stale_proof(proof_env) -> None:
    assert verify(proof_env).returncode == 0
    proof_env["local_sums"].write_text("{}  tona-server\n".format("0" * 64))
    r = verify(proof_env)
    assert r.returncode != 0
    assert not (proof_env["proofs"] / f"{TAG}.proof").exists()


def test_sign_refuses_without_proof(proof_env) -> None:
    r = sign(proof_env)
    assert r.returncode != 0
    assert f"scripts/release-verify-build.sh {TAG}" in r.stderr
    assert "FAKE-GH-REACHED-ATTESTATION" not in r.stderr


def test_sign_refuses_proof_of_another_commit(proof_env) -> None:
    assert verify(proof_env).returncode == 0
    proof = proof_env["proofs"] / f"{TAG}.proof"
    proof.write_text(proof.read_text().replace(proof_env["commit"], "a" * 40))
    r = sign(proof_env)
    assert r.returncode != 0
    assert "another tag or commit" in r.stderr


def test_sign_refuses_changed_draft(proof_env) -> None:
    assert verify(proof_env).returncode == 0
    (proof_env["draft"] / "tona-server").write_bytes(b"other\n")
    (proof_env["draft"] / "SHA256SUMS").write_text("{}  tona-server\n".format(hashlib.sha256(b"other\n").hexdigest()))
    r = sign(proof_env)
    assert r.returncode != 0
    assert "draft changed" in r.stderr


def test_sign_accepts_matching_proof(proof_env) -> None:
    assert verify(proof_env).returncode == 0
    r = sign(proof_env)
    assert "ok: rebuilt locally" in r.stdout
    assert "FAKE-GH-REACHED-ATTESTATION" in r.stderr
