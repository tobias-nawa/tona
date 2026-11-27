import dependencies
import pytest
from conftest import commit_all, write

SHA = "a" * 40
HASH = "b" * 64


def test_hashed_pinned_requirements_pass():
    text = (
        f"# c\nfoo==1.0 \\\n    --hash=sha256:{HASH} \\\n    --hash=sha256:{HASH}\n    # via x\n"
        f"bar_baz==2 --hash=sha256:{HASH}\n"
    )
    names, problems = dependencies.pip_file("r.txt", text)
    assert names == ["foo", "bar_baz"]
    assert problems == []


@pytest.mark.parametrize(
    ("line", "fragment"),
    [
        ("foo>=1.0 --hash=sha256:" + HASH, "not pinned"),
        ("foo --hash=sha256:" + HASH, "not pinned"),
        ("foo==1.0", "no `--hash"),
        ("foo~=1.0", "not pinned"),
    ],
)
def test_unpinned_or_unhashed_requirement_fails(line, fragment):
    _, problems = dependencies.pip_file("r.txt", line + "\n")
    assert any(fragment in p for p in problems)


def test_requirement_options_are_not_requirements():
    names, problems = dependencies.pip_file("r.lock", "--index-url https://example.org/\n-r other.txt\n")
    assert names == [] and problems == []


PRE_COMMIT = f"""repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.17.3
    hooks:
      - id: ruff
  - repo: https://github.com/example/tool
    rev: {SHA}
  - repo: local
    hooks:
      - id: x
        additional_dependencies: ["clang-format==22.1.8", "github.com/a/b/v8@v8.30.1"]
"""


def test_pre_commit_pins_pass_and_names_are_collected():
    names, problems = dependencies.pre_commit_file(".pre-commit-config.yaml", PRE_COMMIT)
    assert names == ["astral-sh/ruff-pre-commit", "example/tool", "clang-format", "github.com/a/b/v8"]
    assert problems == []


@pytest.mark.parametrize("rev", ["main", "master", "v1", "latest", "stable"])
def test_pre_commit_branch_or_floating_rev_fails(rev):
    text = f"repos:\n  - repo: https://github.com/x/y\n    rev: {rev}\n"
    assert dependencies.pre_commit_file("c", text)[1]


def test_pre_commit_unpinned_additional_dependency_fails():
    text = "repos:\n  - repo: local\n    hooks:\n      - additional_dependencies: [lizard, 'foo>=1']\n"
    assert len(dependencies.pre_commit_file("c", text)[1]) == 2


def test_workflow_actions_must_be_pinned_to_a_sha():
    text = (
        f"      - uses: actions/checkout@{SHA} # v7\n"
        "      - uses: actions/cache@v4\n"
        "      - uses: ./.github/actions/local\n"
        f"      - uses: github/codeql-action/init@{SHA}\n"
        "      - uses: docker://alpine:3\n"
    )
    names, problems = dependencies.workflow_file("w.yml", text)
    assert names == ["actions/checkout", "actions/cache", "github/codeql-action"]
    assert len(problems) == 2
    assert "w.yml:2" in problems[0] and "digest" in problems[1]


def test_cmake_fetch_needs_a_sha_or_a_url_hash():
    good = f"FetchContent_Declare(a\n    GIT_REPOSITORY https://x\n    GIT_TAG {SHA}  # v1\n    SYSTEM)\n"
    by_hash = f"FetchContent_Declare(b URL https://x/y.tgz URL_HASH SHA256={HASH})\n"
    tag = "FetchContent_Declare(c GIT_REPOSITORY https://x GIT_TAG v1.2.3)\n"
    branch = "ExternalProject_Add(d GIT_REPOSITORY https://x GIT_TAG main)\n"
    assert dependencies.cmake_file("C", good + by_hash) == (["a", "b"], [])
    assert len(dependencies.cmake_file("C", tag + branch)[1]) == 2


def test_models_need_revision_sha_hashes_and_a_license():
    good = f'[m]\nlicense = "MIT"\nrevision = "{SHA}"\nfiles = [ {{ path = "a", sha256 = "{HASH}" }} ]\n'
    assert dependencies.models_file("m.toml", good)[1] == []
    bad = '[m]\nrevision = "main"\nfiles = [ { path = "a", sha256 = "abc" } ]\n'
    assert len(dependencies.models_file("m.toml", bad)[1]) == 3
    unpublished = '[m]\nrevision = "UNPUBLISHED"\nfiles = []\n'
    assert dependencies.models_file("m.toml", unpublished)[1] == []


def test_vendored_entries_must_be_in_the_readme(tmp_path):
    write(tmp_path, "engine/third_party/README.md", "| `lib.h` | x |\nLICENSE-lib\n")
    write(tmp_path, "engine/third_party/lib.h")
    write(tmp_path, "engine/third_party/LICENSE-lib")
    assert dependencies.vendored(tmp_path) == (["LICENSE-lib"], [])
    write(tmp_path, "engine/third_party/sneaky.h")
    assert "sneaky.h" in dependencies.vendored(tmp_path)[1][0]


def test_listed_is_a_word_match():
    doc = "| pytest, pluggy | Actions/Checkout | MIT |"
    assert dependencies.listed("pytest", doc)
    assert dependencies.listed("actions/checkout", doc)
    assert not dependencies.listed("pytes", doc)
    assert not dependencies.listed("plug", doc)


def test_main_reports_an_unlisted_dependency(repo, capsys):
    write(repo, "docs/contribute/licenses.md", "pytest MIT\n")
    write(repo, "tools/requirements.txt", f"pytest==9 --hash=sha256:{HASH}\nslopsquat==1 --hash=sha256:{HASH}\n")
    commit_all(repo)
    assert dependencies.main() == 1
    err = capsys.readouterr().err
    assert "`slopsquat` is not listed" in err
    assert "pytest`" not in err
    write(repo, "docs/contribute/licenses.md", "pytest slopsquat\n")
    assert dependencies.main() == 0
