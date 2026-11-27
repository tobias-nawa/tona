"""Dependency gate: nothing is pulled in that is not pinned and listed with its licence.

Usage: dependencies.py   (reads the repository; no arguments)

Every dependency of the build, the tests, the gates and the CI is checked at its source file:
- pip requirement files (`requirements*.txt`, `*.lock`): `name==version` and at least one `--hash=sha256:`;
- `.pre-commit-config.yaml`: `rev` is a release tag (`v1.2.3`, `1.2.3`) or a full commit SHA, never a branch; every
  `additional_dependencies` entry is `name==version` or `module@vX.Y.Z`;
- GitHub Actions: `uses:` pinned to a full 40-hex commit SHA (a Docker image to a `sha256:` digest);
- CMake `FetchContent_Declare` / `ExternalProject_Add`: `GIT_TAG` a full commit SHA, or `URL_HASH`;
- `toolchain/models.toml`: `revision` a full commit SHA, `sha256` of 64 hex digits per file, and a `license` text
  (entries that say UNPUBLISHED are refused by the downloader and skipped here);
- `engine/third_party/`: every top-level entry is named in its README, with its licence file.
Each name (package, hook repository, action, fetched project, vendored licence file) must also appear in
docs/contribute/licenses.md, which is where a reader looks up what ships and under which licence.

Why: a coding agent proposes package names from memory, and 19.7% of the recommended packages in a study of 576,000 code
samples did not exist (USENIX Security 2025, "slopsquatting"). A
pin by hash makes the name resolve to the bytes someone looked at; the list makes a new name visible in the diff.

What it cannot check: that the package is the one meant (a human looks it up once), the licence text in the list is
true, a download URL inside a script or a shell command (`pip install x` in a workflow `run:` is not seen), and the
transitive dependencies of an unhashed tool (only hashed requirement files are accepted, so those are covered).
"""

import re
import sys
import tomllib
from pathlib import Path

from common import repo_root, report, tracked_files

LICENSES = "docs/contribute/licenses.md"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
TAG = re.compile(r"^v?\d+(\.\d+)+([-.][0-9A-Za-z]+)*$")
REQUIREMENT = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?\s*([=<>!~]=?[^\s\\;]*)?")
PRE_COMMIT_REPO = re.compile(r"^\s*-\s+repo:\s*(\S+)")
PRE_COMMIT_REV = re.compile(r"^\s*rev:\s*(\S+)")
PRE_COMMIT_DEPS = re.compile(r"additional_dependencies:\s*\[(.*)\]")
USES = re.compile(r"^\s*(?:-\s+)?uses:\s*([^\s#]+)")


def normal(name: str) -> str:
    return re.sub(r"[_.]+", "-", name.strip().lower())


def listed(key: str, licences: str) -> bool:
    """Whether the key stands in the licence list as a word (a name may not hide in a longer one)."""
    pattern = r"(?<![a-z0-9-])" + re.escape(normal(key)) + r"(?![a-z0-9])"
    return re.search(pattern, normal(licences)) is not None


def pip_file(name: str, text: str) -> tuple[list[str], list[str]]:
    """(names, problems) of one requirement file. Logical lines continue on a trailing backslash."""
    joined = re.sub(r"\\\n", " ", text)
    names, problems = [], []
    for line in joined.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "-")):
            continue
        match = REQUIREMENT.match(stripped)
        if not match:
            continue
        package, spec = match.group(1), match.group(2) or ""
        names.append(package)
        if not spec.startswith("=="):
            problems.append(f"{name}: `{package}` is not pinned with `==`")
        if "--hash=sha256:" not in stripped:
            problems.append(f"{name}: `{package}` has no `--hash=sha256:`")
    return names, problems


def pre_commit_file(name: str, text: str) -> tuple[list[str], list[str]]:
    names, problems = [], []
    repo = ""
    for line in text.splitlines():
        match = PRE_COMMIT_REPO.match(line)
        if match:
            repo = match.group(1)
            if repo != "local":
                names.append(re.sub(r"^https?://[^/]+/", "", repo).removesuffix(".git"))
            continue
        match = PRE_COMMIT_REV.match(line)
        if match and not (TAG.match(match.group(1)) or SHA40.match(match.group(1))):
            problems.append(f"{name}: `rev: {match.group(1)}` of {repo} is neither a release tag nor a full commit SHA")
        match = PRE_COMMIT_DEPS.search(line)
        if match:
            for dep in re.findall(r"[^\s,\"'][^,\"']*", match.group(1)):
                package = re.split(r"==|@", dep)[0]
                names.append(package)
                if not re.search(r"==\S+$|@v?\d\S*$", dep):
                    problems.append(
                        f"{name}: additional dependency `{dep}` of {repo} is not pinned (`==` or `@vX.Y.Z`)"
                    )
    return names, problems


def workflow_file(name: str, text: str) -> tuple[list[str], list[str]]:
    names, problems = [], []
    for number, line in enumerate(text.splitlines(), start=1):
        match = USES.match(line)
        if not match or match.group(1).startswith("./"):
            continue
        ref = match.group(1)
        action, _, pinned = ref.partition("@")
        if action.startswith("docker://"):
            if "@sha256:" not in ref:
                problems.append(f"{name}:{number}: `{ref}` is not pinned to a digest")
            continue
        names.append("/".join(action.split("/")[:2]))
        if not SHA40.match(pinned):
            problems.append(f"{name}:{number}: `{ref}` is not pinned to a full commit SHA")
    return names, problems


def cmake_file(name: str, text: str) -> tuple[list[str], list[str]]:
    names, problems = [], []
    for match in re.finditer(r"(?:FetchContent_Declare|ExternalProject_Add)\(\s*(\S+)([^)]*)\)", text):
        project, body = match.group(1), match.group(2)
        names.append(project)
        tag = re.search(r"GIT_TAG\s+(\S+)", body)
        has_hash = re.search(r"URL_HASH\s+SHA256=[0-9a-fA-F]{64}", body)
        if not (tag and SHA40.match(tag.group(1))) and not has_hash:
            problems.append(f"{name}: `{project}` needs `GIT_TAG <full commit SHA>` or `URL_HASH SHA256=...`")
    return names, problems


def models_file(name: str, text: str) -> tuple[list[str], list[str]]:
    problems = []
    for key, entry in tomllib.loads(text).items():
        if not isinstance(entry, dict) or entry.get("revision") == "UNPUBLISHED":
            continue
        if not SHA40.match(str(entry.get("revision", ""))):
            problems.append(f"{name}: [{key}] `revision` is not a full commit SHA")
        if not str(entry.get("license", "")).strip():
            problems.append(f"{name}: [{key}] has no `license`")
        for item in entry.get("files", []):
            if not SHA256.match(str(item.get("sha256", ""))):
                problems.append(f"{name}: [{key}] `{item.get('path', '?')}` has no 64-digit `sha256`")
            if "revision" in item and not SHA40.match(str(item["revision"])):
                problems.append(f"{name}: [{key}] `{item.get('path', '?')}` revision is not a full commit SHA")
    return [], problems


def vendored(root: Path) -> tuple[list[str], list[str]]:
    base = root / "engine/third_party"
    if not base.is_dir():
        return [], []
    readme = (base / "README.md").read_text(encoding="utf-8") if (base / "README.md").is_file() else ""
    names, problems = [], []
    for entry in sorted(base.iterdir()):
        if entry.name == "README.md":
            continue
        if entry.name.startswith("LICENSE-"):
            names.append(entry.name)
        if (
            entry.name not in readme
            and f"{entry.name}/" not in readme
            and entry.name.removeprefix("LICENSE-") not in readme
        ):
            problems.append(f"engine/third_party/{entry.name} is not named in engine/third_party/README.md")
    return names, problems


def is_requirement_file(path: str) -> bool:
    base = Path(path).name
    return ("requirements" in base and base.endswith(".txt")) or base.endswith(".lock")


def handler(path: str):
    """The parser for a tracked file, or None when it is no dependency source."""
    if is_requirement_file(path):
        return pip_file
    if path == ".pre-commit-config.yaml":
        return pre_commit_file
    if path.startswith(".github/workflows/") and path.endswith((".yml", ".yaml")):
        return workflow_file
    if path.endswith(("CMakeLists.txt", ".cmake")):
        return cmake_file
    if path == "toolchain/models.toml":
        return models_file
    return None


def collect(root: Path) -> tuple[list[str], list[str]]:
    names: list[str] = []
    problems: list[str] = []
    for path in tracked_files((".txt", ".lock", ".yml", ".yaml", ".cmake", ".toml")):
        parse = handler(path)
        if parse:
            found, bad = parse(path, (root / path).read_text(encoding="utf-8", errors="replace"))
            names += found
            problems += bad
    found, bad = vendored(root)
    return names + found, problems + bad


def main() -> int:
    root = repo_root()
    names, problems = collect(root)
    licences = (root / LICENSES).read_text(encoding="utf-8")
    for key in sorted(set(names), key=str.lower):
        if not listed(key, licences):
            problems.append(f"`{key}` is not listed in {LICENSES} (name, where it is used, licence)")
    return report("dependencies", sorted(problems), len(set(names)), "See the header of tools/gates/dependencies.py.")


if __name__ == "__main__":
    sys.exit(main())
