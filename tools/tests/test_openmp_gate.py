"""The OpenMP gate: a pragma without -fopenmp fails; with the flag, without a pragma, or vendored it passes."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gates"))
import openmp


def entry(root: Path, rel: str, flags: str) -> dict:
    return {"directory": str(root), "file": rel, "command": f"clang++ {flags} -c {rel}"}


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "engine/third_party").mkdir(parents=True)
    (tmp_path / "engine/a.cpp").write_text("void f() {\n#pragma omp parallel for\n    for (;;) {}\n}\n")
    (tmp_path / "engine/b.cpp").write_text("void g() {}\n")
    (tmp_path / "engine/third_party/c.cpp").write_text("#pragma omp parallel\n")
    return tmp_path


def test_pragma_without_the_flag_is_reported(tree):
    assert openmp.missing_openmp([entry(tree, "engine/a.cpp", "-O2")], tree) == ["engine/a.cpp"]


def test_pragma_with_the_flag_passes(tree):
    assert openmp.missing_openmp([entry(tree, "engine/a.cpp", "-O2 -fopenmp")], tree) == []


def test_source_without_a_pragma_passes_without_the_flag(tree):
    assert openmp.missing_openmp([entry(tree, "engine/b.cpp", "-O2")], tree) == []


def test_vendored_code_is_exempt(tree):
    assert openmp.missing_openmp([entry(tree, "engine/third_party/c.cpp", "-O2")], tree) == []


def test_arguments_form_of_a_database_entry_is_read(tree):
    e = {"directory": str(tree), "file": "engine/a.cpp", "arguments": ["clang++", "-O2", "-fopenmp"]}
    assert openmp.missing_openmp([e], tree) == []
