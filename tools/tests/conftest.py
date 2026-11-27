"""Shared fixtures: the gate scripts are plain modules next to each other, the others have no package either."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for sub in ("tools/gates", "tools", "tools/cache_sim", "tools/papers", "scripts"):
    sys.path.insert(0, str(ROOT / sub))


# The pull request jobs of CI pass these to the gates; a test that wants one sets it itself.
PULL_REQUEST_ENV = ("PR_BASE", "PR_HEAD", "PR_BODY", "PR_TITLE", "PR_AUTHOR", "PR_AUTHOR_ASSOCIATION")


@pytest.fixture(autouse=True)
def _no_pull_request_env(monkeypatch):
    for name in PULL_REQUEST_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _desktop_guard_off(monkeypatch):
    """The tests run on a desktop with its monitors on; the guard has its own tests with a fake /sys/class/drm."""
    monkeypatch.setenv("TONA_DESKTOP_GUARD", "off")


def load_script(relative: str):
    """Imports a script whose file name is no valid module name (isa-diff.py)."""
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """An empty git repository on branch main, as the current directory (the gates ask git for the root)."""
    git(tmp_path, "init", "-q", "-b", "main")
    for key, value in (("user.email", "t@example.com"), ("user.name", "t"), ("commit.gpgsign", "false")):
        git(tmp_path, "config", key, value)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def write(root: Path, name: str, text: str = "") -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def commit_all(root: Path, message: str = "x") -> None:
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", message)
