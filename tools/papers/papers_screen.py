"""Screened book (research/papers-screened.tsv) and the watch list of open PRs (research/papers-watch.tsv)."""

import json
import subprocess
from datetime import date
from pathlib import Path

from papers_common import GITHUB_DROPS_TSV, SCREENED_TSV

WATCH_TSV = SCREENED_TSV.parent / "papers-watch.tsv"
SCREENED_HEAD = ["id", "date", "source", "topics", "decision", "reason", "method"]
# How a hit was looked at, so a drop is honest about its depth: the title alone, the abstract, the full text, or only
# a keyword filter over the titles (nobody read the title).
METHODS = ("title-read", "abstract-read", "fulltext-read", "keyword-filter")
KEYWORD_REASONS = ("title-filter", "bulk-no-keyword")
WATCH_HEAD = ["key", "title", "state", "created", "checked"]
DECISIONS = ("drop", "note", "candidate")


def derive_method(decision: str, reason: str) -> str:
    """Method of a row written before the column existed: keyword filters are named by their reason."""
    return "keyword-filter" if reason in KEYWORD_REASONS else "title-read"


def read_table(path: Path, head: list[str]) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        cells = line.split("\t")
        if head is SCREENED_HEAD and len(cells) == len(head) - 1:
            cells.append(derive_method(cells[4], cells[5]))  # book from before the method column
        if len(cells) == len(head):
            rows.append(dict(zip(head, cells, strict=True)))
    return rows


def write_table(path: Path, head: list[str], rows: list[dict]) -> None:
    lines = ["\t".join(head)] + ["\t".join(str(r[c]).replace("\t", " ").replace("\n", " ") for c in head) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def screened_ids(path: Path = SCREENED_TSV) -> set[str]:
    return {r["id"] for r in read_table(path, SCREENED_HEAD)}


def load_overrides(path: Path) -> dict[str, tuple[str, ...]]:
    """`id<TAB>decision<TAB>reason[<TAB>method]` lines (no header) -> {id: (decision, reason[, method])}."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        ident, decision, reason, *method = line.split("\t")
        if decision not in DECISIONS:
            raise ValueError(f"{ident}: decision must be one of {DECISIONS}, not {decision!r}")
        if len(method) > 1 or (method and method[0] not in METHODS):
            raise ValueError(f"{ident}: method must be one of {METHODS}, not {method!r}")
        out[ident] = (decision, reason, *method)
    return out


def screen_rows(
    records: list[dict], overrides: dict, default: tuple[str, str], method: str = "title-read"
) -> list[dict]:
    """One row for every record: its override, else the default. Unknown override ids are an error.

    `method` is the default for rows whose override names none; the default decision of a keyword reason is always
    a keyword-filter.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, not {method!r}")
    keys = {r.get("id") or r["key"] for r in records}
    unknown = set(overrides) - keys
    if unknown:
        raise ValueError(f"overrides for ids that are not in the candidates: {sorted(unknown)[:5]}")
    rows = []
    for rec in records:
        ident = rec.get("id") or rec["key"]
        decision, reason, *own = overrides.get(ident, default)
        source = "arxiv" if "id" in rec else f"github-{rec['kind']}"
        rows.append(
            {
                "id": ident,
                "date": rec.get("date") or rec.get("created", ""),
                "source": source,
                "topics": "+".join(rec.get("topics", [])) or "-",
                "decision": decision,
                "reason": reason,
                "method": own[0] if own else ("keyword-filter" if reason in KEYWORD_REASONS else method),
            }
        )
    return rows


def add_screened(rows: list[dict], path: Path = SCREENED_TSV, drops: Path = GITHUB_DROPS_TSV) -> int:
    """Appends rows whose id is not yet in either book; returns how many were added.

    GitHub rows with decision `drop` go to the local `drops` file (not in the repository), everything else to `path`.
    """
    have = read_table(path, SCREENED_HEAD)
    have_drops = read_table(drops, SCREENED_HEAD)
    seen = {r["id"] for r in have} | {r["id"] for r in have_drops}
    fresh = []
    for row in rows:
        if row["id"] not in seen:
            seen.add(row["id"])
            fresh.append(row)
    local = [r for r in fresh if r["source"].startswith("github") and r["decision"] == "drop"]
    kept = [r for r in fresh if r not in local]
    write_table(path, SCREENED_HEAD, have + kept)
    if local:
        drops.parent.mkdir(parents=True, exist_ok=True)
        write_table(drops, SCREENED_HEAD, have_drops + local)
    return len(fresh)


def watch_rows(records: list[dict], overrides: dict, today: str) -> list[dict]:
    """Open PRs screened as candidate go on the watch list."""
    out = []
    for rec in records:
        decision = overrides.get(rec.get("key"), ("drop", ""))[0]
        if rec.get("kind") == "pr" and rec.get("state") == "open" and decision == "candidate":
            out.append(
                {"key": rec["key"], "title": rec["title"], "state": "open", "created": rec["created"], "checked": today}
            )
    return out


def add_watch(rows: list[dict], path: Path = WATCH_TSV) -> None:
    have = {r["key"]: r for r in read_table(path, WATCH_HEAD)}
    for row in rows:
        have.setdefault(row["key"], row)
    write_table(path, WATCH_HEAD, list(have.values()))


def gh_pull_state(repo: str, number: str) -> str:
    """open, merged or closed, from `gh api repos/<repo>/pulls/<n>`."""
    proc = subprocess.run(
        ["gh", "api", f"repos/{repo}/pulls/{number}"], capture_output=True, text=True, timeout=60, check=False
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    data = json.loads(proc.stdout)
    return "merged" if data.get("merged_at") else data["state"]


def refresh_watch(path: Path = WATCH_TSV, state_of=gh_pull_state, today: str | None = None) -> list[str]:
    """Re-checks every PR; returns the changes as text lines. A failed lookup keeps the old state."""
    today = today or date.today().isoformat()
    rows = read_table(path, WATCH_HEAD)
    changes = []
    for row in rows:
        repo, number = row["key"].rsplit("#", 1)
        try:
            state = state_of(repo, number)
        except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired):
            continue
        if state != row["state"]:
            changes.append(f"{row['key']}: {row['state']} -> {state}  {row['title'][:70]}")
            row["state"] = state
        row["checked"] = today
    write_table(path, WATCH_HEAD, rows)
    return changes
