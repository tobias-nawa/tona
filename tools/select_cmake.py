"""What a change of the CMake files does to the golden binaries. A new test target must not trigger the golden
checks, a changed compiler flag of the server must trigger the server golden only. So base and head are configured
(not built) in scratch directories and the commands that build `tona-gpu-gen` and `tona-server` are compared, with the
source and build directories replaced by placeholders. The golden scripts use no other file of the build than these two
binaries (scripts/golden.sh, scripts/golden-server.sh); everything else they read is under work/ or in toolchain/,
which triggers everything on its own."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from select_graph import GraphError

GOLDEN_TARGETS = ("tona-gpu-gen", "tona-server")
SCRATCH = Path(
    os.environ.get("TONA_SELECT_SCRATCH", str(Path(__file__).resolve().parent.parent / "work/tmp/select-checks"))
)  # not /tmp: that can be RAM
# What engine/CMakeLists.txt reads next to engine/: the benchmark corpus (a fuzz test) and work/ (models, sources)
SOURCE_DIRS = ("engine", "bench/corpus")


def sh(cmd: str, cwd: Path) -> str:
    done = subprocess.run(["bash", "-c", cmd], cwd=cwd, capture_output=True, text=True, check=False)
    if done.returncode:
        raise GraphError(f"{cmd.rsplit('&&', maxsplit=1)[-1].strip()[:80]} failed: {done.stderr.strip()[-300:]}")
    return done.stdout


def configure(repo: Path, rev: str, dest: Path) -> Path:
    """Unpacks the sources of `rev` and configures them like `just build` does; nothing is compiled."""
    src, build = dest / "src", dest / "build"
    src.mkdir(parents=True)
    sh(f"git archive {rev} {' '.join(SOURCE_DIRS)} | tar -x -C {src}", repo)
    (src / "work").symlink_to(repo / "work")  # the model and sources the CMake file looks for
    sh(f"source toolchain/env.sh && cmake -S {src}/engine -B {build} -G Ninja", repo)
    return build


def build_commands(repo: Path, build: Path, target: str) -> list[str]:
    src = str(build.parent / "src")
    lines = sh(f"ninja -C {build} -t commands {target}", repo).splitlines()
    return sorted(line.replace(src, "@SRC@").replace(str(build), "@BUILD@") for line in lines)


def changed_binaries(repo: Path, base_rev: str, head_rev: str) -> list[str]:
    """The golden binaries whose build commands differ between the two revisions. Raises GraphError if either
    revision does not configure: the caller then checks everything."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(dir=SCRATCH))
    try:
        builds = [configure(repo, rev, work / name) for name, rev in (("base", base_rev), ("head", head_rev))]
        return [t for t in GOLDEN_TARGETS if build_commands(repo, builds[0], t) != build_commands(repo, builds[1], t)]
    finally:
        shutil.rmtree(work, ignore_errors=True)
