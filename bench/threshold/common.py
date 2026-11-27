"""Shared constants and helpers of the threshold benchmark.

Foreign code is only ever read here (ast.parse) or run inside the sandbox (sandbox.py); nothing is imported.
"""

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Raw data stays out of the repository history and off /tmp (tmpfs = RAM on some machines); work/ is ignored.
WORK = Path(os.environ.get("W76_WORK", str(ROOT / "work/tmp/w76")))
MODEL = ROOT / "work/models/coder-iq1_m/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf"
TOKENIZE = ROOT / "work/build/engine/tona-tokenize"

# First guess of the file budget before the bodies are removed; holes.py adjusts it until the holed context is about
# 62,500 tokens (all qualifying functions are holes, so the difference is several thousand tokens).
CONTEXT_TOKENS = 67500
# The sparse attention (QSA) selects nothing below 2,051 positions (research/measurements/2026-10-29-indexcache.md).
QSA_MIN = 2051


@dataclass(frozen=True)
class Repo:
    name: str
    url: str
    commit: str
    license: str
    src_dir: str  # directory of the package, relative to the repository root
    pythonpath: str  # relative to the repository root, for the tests
    test_args: tuple  # pytest arguments that select the tests
    file_order: tuple  # modules (file names in src_dir) in the order they enter the context


REPOS = {
    "pydash": Repo(
        name="pydash",
        url="https://github.com/dgilland/pydash",
        commit="812eb13283d23b0a73d60a950190edc3c9de14ab",
        license="MIT",
        src_dir="src/pydash",
        pythonpath="src",
        test_args=("tests", "--ignore=tests/pytest_mypy_testing", "--ignore=tests/test_annotations.py"),
        file_order=(
            "arrays.py",
            "collections.py",
            "strings.py",
            "functions.py",
            "numerical.py",
            "predicates.py",
            "utilities.py",
            "helpers.py",
            "objects.py",
        ),
    ),
    "boltons": Repo(
        name="boltons",
        url="https://github.com/mahmoud/boltons",
        commit="d31d821d56c8f7756af9c30a864077f2e5855321",
        license="BSD-3-Clause",
        src_dir="boltons",
        pythonpath=".",
        test_args=("tests",),
        file_order=(
            "iterutils.py",
            "strutils.py",
            "dictutils.py",
            "listutils.py",
            "setutils.py",
            "mathutils.py",
            "cacheutils.py",
            "funcutils.py",
            "typeutils.py",
            "queueutils.py",
            "statsutils.py",
            "timeutils.py",
        ),
    ),
}


def repo_dir(repo: Repo) -> Path:
    return WORK / "src" / repo.name


def count_tokens(text: str) -> int:
    return len(token_ids(text))


def token_ids(text: str) -> list:
    out = subprocess.run(
        [str(TOKENIZE), str(MODEL), "--no-special"], input=text.encode(), capture_output=True, check=True
    ).stdout
    return [int(x) for x in out.split()]


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def write_json(path: Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False))
