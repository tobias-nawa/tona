"""scripts/build-fresh.sh: a golden check refuses a build that is older than what it is built from. No GPU.

The cases that need ninja's graph use a tiny real CMake/ninja project and are skipped without cmake, ninja and a C++
compiler.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/build-fresh.sh"
needs_toolchain = pytest.mark.skipif(
    not (shutil.which("cmake") and shutil.which("ninja") and (shutil.which("c++") or shutil.which("g++"))),
    reason="needs cmake, ninja and a C++ compiler to build the small project",
)


def run(build: Path, *binaries: str):
    return subprocess.run([str(SCRIPT), str(build), *binaries], capture_output=True, text=True, check=False)


def bump(path: Path):
    """Make the file newer than everything built so far, whatever the clock resolution."""
    later = path.stat().st_mtime + 10
    os.utime(path, (later, later))


@pytest.fixture
def project(tmp_path):
    src = tmp_path / "src"
    (src / "tests").mkdir(parents=True)
    (src / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.16)\nproject(p CXX)\nadd_executable(tool main.cpp)\n"
    )
    (src / "main.cpp").write_text('#include "util.h"\nint main() { return util(); }\n')
    (src / "util.h").write_text("inline int util() { return 0; }\n")
    (src / "tests/unrelated.cpp").write_text("int unrelated() { return 1; }\n")  # in no target
    build = tmp_path / "build"
    subprocess.run(["cmake", "-S", str(src), "-B", str(build), "-G", "Ninja"], check=True, capture_output=True)
    subprocess.run(["ninja", "-C", str(build)], check=True, capture_output=True)
    return src, build


@needs_toolchain
def test_a_fresh_build_passes(project):
    assert run(project[1], "tool").returncode == 0


@needs_toolchain
def test_a_file_outside_the_dependency_graph_does_not_make_it_stale(project):
    src, build = project
    bump(src / "tests/unrelated.cpp")
    assert run(build, "tool").returncode == 0


@needs_toolchain
@pytest.mark.parametrize("name", ["main.cpp", "util.h", "CMakeLists.txt"])
def test_a_source_header_or_cmake_file_newer_than_the_binary_is_exit_2(project, name):
    src, build = project
    bump(src / name)
    result = run(build, "tool")
    assert result.returncode == 2
    assert "stale" in result.stderr and name in result.stderr


@needs_toolchain
def test_a_binary_ninja_does_not_know_is_exit_2(project):
    _src, build = project
    shutil.copy(build / "tool", build / "other")
    assert run(build, "other").returncode == 2


@needs_toolchain
def test_a_missing_binary_is_exit_2(project):
    result = run(project[1], "nothing")
    assert result.returncode == 2 and "missing" in result.stderr


def test_a_directory_without_cmake_cache_is_exit_2(tmp_path):
    result = run(tmp_path, "tool")
    assert result.returncode == 2 and "CMakeCache" in result.stderr


def test_sources_that_are_gone_are_exit_2(tmp_path):
    (tmp_path / "CMakeCache.txt").write_text("CMAKE_HOME_DIRECTORY:INTERNAL=/nonexistent/engine\n")
    assert run(tmp_path, "tool").returncode == 2
