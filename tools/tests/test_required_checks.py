"""The checks the ruleset of `main` requires must be named like jobs of the workflows (a wrong name locks every PR)."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Posted by hand after the golden comparison (scripts/gpu-status.sh): no workflow job has this name
COMMIT_STATUSES = {"gpu-golden"}


def required_checks() -> list[str]:
    text = (ROOT / "scripts/github-setup.sh").read_text(encoding="utf-8")
    m = re.search(r"^REQUIRED_CHECKS=\(([^)]*)\)", text, re.MULTILINE)
    assert m, "REQUIRED_CHECKS not found in scripts/github-setup.sh"
    return m.group(1).split()


def job_names(path: Path) -> list[str]:
    """Display names of the jobs of one workflow: `name:` of the job, else its id (what GitHub reports as the check)."""
    names: list[str] = []
    in_jobs = False
    job_id = ""
    named = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if re.match(r"^\S", line):
            in_jobs = line.startswith("jobs:")
            continue
        if not in_jobs:
            continue
        m = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", line)
        if m:
            if job_id and not named:
                names.append(job_id)
            job_id, named = m.group(1), False
            continue
        m = re.match(r"^    name:\s*(.+?)\s*$", line)
        if m and job_id and not named:
            names.append(m.group(1).strip("\"'"))
            named = True
    if job_id and not named:
        names.append(job_id)
    return names


def test_every_required_check_is_a_job_name():
    jobs = {name for wf in (ROOT / ".github/workflows").glob("*.yml") for name in job_names(wf)}
    missing = [c for c in required_checks() if c not in jobs and c not in COMMIT_STATUSES]
    assert not missing, f"required checks without a workflow job of that name: {missing}"


def test_job_name_parser():
    names = job_names(ROOT / ".github/workflows/tests.yml")
    assert names == [
        "changes",
        "engine-cpu-tests",
        "engine-sanitizers",
        "clang-tidy",
        "hardening",
        "mini-tokens",
        "ubuntu-no-gpu",
        "coverage",
        "engine-macos",
        "tests-ok",
    ]
    assert job_names(ROOT / ".github/workflows/check.yml") == [
        "changes",
        "gates",
        "prior-art",
        "pr-text",
        "commits",
        "agent-config",
        "installer",
        "check-ok",
    ]


def test_workflows_use_github_hosted_runners_only():
    # GitHub-hosted runners only, never self-hosted (a public repository, docs/contribute/github.md)
    for wf in (ROOT / ".github/workflows").glob("*.yml"):
        for line in wf.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*runs-on:\s*(\S+)", line)
            assert not m or m.group(1).startswith(("ubuntu-", "macos-")), f"{wf.name}: {line.strip()}"


def test_summary_jobs_need_every_other_job():
    # A job missing from `needs` of its summary job could fail without blocking a merge
    for wf in ("tests", "check", "fuzz"):
        text = (ROOT / f".github/workflows/{wf}.yml").read_text(encoding="utf-8")
        m = re.search(rf"^  {wf}-ok:.*?needs: \[([^\]]*)\]", text, re.MULTILINE | re.DOTALL)
        assert m, f"{wf}-ok not found"
        needed = {n.strip() for n in m.group(1).split(",")}
        ids = set(re.findall(r"^  ([A-Za-z0-9_-]+):\s*$", text.split("\njobs:\n", 1)[1], re.MULTILINE))
        assert needed == ids - {f"{wf}-ok"}, f"{wf}: needs {sorted(needed)} but jobs are {sorted(ids)}"
