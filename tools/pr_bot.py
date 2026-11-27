"""Text of the pull request comment (.github/workflows/pr-bot.yml): how to fix failed text gates, which checks the
change needs, a welcome for first contributions.

Usage: pr_bot.py --dir DIR [--out FILE]
       DIR holds the pull request as the workflow fetched it through the API: title.txt, body.md, files.txt (one path
       per line), association.txt (GitHub's author_association), author.txt. The comment goes to --out (default:
       stdout). Everything in DIR is data written by an outsider: it is read, never executed, and every piece of it
       that reaches the comment stands inside an inline code span (no mention, link, image or HTML renders).

The text is fixed here and written by people; no model is involved. The gates are the ones the checks run (title,
prior art, pull request text), without their online parts: the links and the label `accepted` are left to the CI jobs
`prior-art` and `pr-text`. The areas are path based on purpose: on the runner there is no engine build, so
tools/select_checks.py would say "everything"; it still gives the exact list locally.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "gates"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pr_text
import pr_title
import prior_art
import select_rules
from common import BOT_AUTHORS

FIRST_LINE = "**Pull request checks**"
FIRST_TIMERS = ("FIRST_TIME_CONTRIBUTOR", "FIRST_TIMER", "NONE")
REPO_URL = "https://github.com/tobias-nawa/tona"
MAX_FINDING = 300

TITLE_HELP = (
    "The title becomes the commit on `main`. Format `type(scope): subject (#n)`; `(#n)` is the issue and is"
    " required for `feat`, `fix`, `perf`.\n\nExample: `fix(server): reject an empty prompt (#123)`"
)
PRIOR_ART_HELP = (
    "Engine code needs `Change type:` and a section `## Prior art` in the text (skill `research-before-building`).\n\n"
    "Example:\n\n```text\nChange type: technique\n\n## Prior art\n\n- https://arxiv.org/abs/2401.00001 (read): adopted"
    " the cache layout, not the scheduler\n\nSearch: tools/papers/scan.py topics v3; topics: kv-cache; window:"
    " 2025-11-01..2026-11-01 (12 months)\n```\n\nWithout engine code: `Change type: none` and `n/a` under"
    " `## Prior art`."
)
TEXT_HELP = (
    "The text needs one line `AI use: none`, or `AI use: <agent, model, what it did>` plus one line"
    " `Assisted-by: <agent>:<model>` per agent; a section `## Models` with a row or `n/a`; and, for `feat`, `fix`,"
    " `perf`, a line `Closes #n`.\n\nExample:\n\n```text\nCloses #123\n\nAI use: Claude Code, claude-sonnet-5-5:"
    " wrote the tests\n\nAssisted-by: Claude Code:claude-sonnet-5-5\n```"
)

NEEDS = {
    "all": (
        "Build files, toolchain or golden scripts",
        "all checks (gates, unit tests, goldens, speed) by a maintainer on the reference machine",
    ),
    "device": (
        "Reference backend or compute code (`engine/gpu/`, `engine/cpu/`, quantization)",
        "golden checks, bit-exactness and speed comparison (`scripts/ab.py`) by a maintainer on the reference GPU",
    ),
    "backend": (
        "Another backend (`engine/backend/<name>/`)",
        "that backend's tests and evidence from its owner; if shared code changed, the reference must stay identical"
        " (maintainer: golden check and speed comparison)",
    ),
    "host": (
        "Engine host code",
        "`just test-cpu`; a maintainer adds the golden check of the server and, on the hot path, a speed comparison",
    ),
    "long": ("File names of long-context state", "a long-context run by a maintainer (`scripts/ab.py` workload huge)"),
    "pressure": (
        "File names of memory pressure paths",
        "a pressure series by a maintainer (20 requests, `TONA_DIAG=1`)",
    ),
    "installer": ("Installer only", "its own build and tests; no GPU run"),
    "tools": ("Scripts, tools, bench", "gates and `just test-py`; no GPU run"),
    "ci": ("Workflows or gates", "gates (`just check`); a maintainer reads the diff, no GPU run"),
    "docs": ("Documents, research", "gates only (`just check`); no GPU run"),
}


def code(text: str) -> str:
    """Outsider text as one inline code span: one line, no backtick, bounded."""
    flat = " ".join(text.replace("`", "'").split())
    return f"`{flat[:MAX_FINDING]}`"


def areas_of(path: str) -> set[str]:
    """The keys of NEEDS one changed file asks for (path rules of tools/select_rules.py, no build graph)."""
    if path.startswith((".github/workflows/", "tools/gates/")):
        return {"ci"}
    verdict = select_rules.outside_engine(path)
    if verdict is None:
        return engine_areas(path)
    if verdict.everything:
        return {"all"}
    if verdict.installer:
        return {"installer"}
    return {"docs"} if verdict.cls == "C0" else {"tools"}


def engine_areas(path: str) -> set[str]:
    names = select_rules.parts(path)
    found = {"backend"} if path.startswith("engine/backend/") else {"host"}
    if select_rules.COMPUTE_PATHS.match(path):
        found = {"device"}
    if names & select_rules.LONG_CONTEXT:
        found.add("long")
    if names & select_rules.PRESSURE:
        found.add("pressure")
    return found


def needs_section(paths: list[str]) -> list[str]:
    found: set[str] = set()
    for path in paths:
        found |= areas_of(path)
    lines = ["### Checks this change needs", ""]
    if not found:
        lines.append("No changed files were found.")
    for key, (what, run) in NEEDS.items():
        if key in found:
            lines.append(f"- {what}: {run}.")
    lines += [
        "",
        "These come from the file paths only. The exact list is `python3 tools/select_checks.py` on your machine (it"
        " needs an engine build); a maintainer runs everything that needs the GPU.",
    ]
    return lines


def findings(title: str, body: str, paths: list[str]) -> list[tuple[str, list[str], str]]:
    """(gate, problems, help) for every text gate that fails."""
    visible_body = pr_text.COMMENT.sub("", body)
    gates = [
        ("Title", pr_title.parse(title, visible_body)[1], TITLE_HELP),
        ("Prior art", prior_art.check(body, paths), PRIOR_ART_HELP),
        ("Pull request text", pr_text.check(body, title), TEXT_HELP),
    ]
    return [gate for gate in gates if gate[1]]


def text_section(title: str, body: str, paths: list[str]) -> list[str]:
    failed = findings(title, body, paths)
    if not failed:
        return ["### Text checks", "", "All text checks pass: title, prior art, pull request text."]
    lines = ["### Text checks", "", "These need a fix; this comment updates when you edit the title or the text.", ""]
    for gate, problems, hint in failed:
        lines += [f"#### {gate}", ""] + [f"- {code(problem)}" for problem in problems] + ["", hint, ""]
    return lines[:-1]


def welcome() -> list[str]:
    return [
        "Welcome, and thank you for your first pull request. Start with"
        f" [CONTRIBUTING.md]({REPO_URL}/blob/main/CONTRIBUTING.md); `scripts/setup.sh --no-gpu`"
        f" ([script]({REPO_URL}/blob/main/scripts/setup.sh)) installs the gate tools without a GPU, and `just check`"
        f" ([what it runs]({REPO_URL}/blob/main/docs/contribute/verification.md#1-every-change-gates)) must be green"
        " before a maintainer reviews.",
        "",
    ]


def build_comment(title: str, body: str, paths: list[str], association: str) -> str:
    lines = [FIRST_LINE, "", "Written by a script from fixed text (`tools/pr_bot.py`), not by a model.", ""]
    if association in FIRST_TIMERS:
        lines += welcome()
    lines += [*text_section(title, body, paths), "", *needs_section(paths)]
    lines += ["", "The links in the prior art section and the label `accepted` are checked by the CI jobs."]
    return "\n".join(lines) + "\n"


def read(directory: Path, name: str) -> str:
    path = directory / name
    return path.read_text(encoding="utf-8").strip("\n") if path.is_file() else ""


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", required=True)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    directory = Path(args.dir)
    if read(directory, "author.txt") in BOT_AUTHORS:
        return 0  # Dependabot pull requests have no template; no comment
    paths = [p for p in re.split(r"\r?\n", read(directory, "files.txt")) if p]
    title, body, association = (read(directory, n) for n in ("title.txt", "body.md", "association.txt"))
    text = build_comment(title, body, paths, association)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
