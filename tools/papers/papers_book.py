"""The coverage book (research/papers-coverage.md): one row per scan run, and the gap check on top of it."""

from datetime import date
from pathlib import Path

from papers_common import COVERAGE_MD, days_between, parse_day

HEADER = """# Paper scan coverage

One row per run of `tools/papers/scan.py` (see `tools/papers/README.md`). A day counts as covered for a source only
when a row with `Scope` = `all`, `Complete` = `yes` and a topic version at least as new as `tools/papers/topics.toml`
contains it. Run `python3 tools/papers/scan.py coverage` first; close every gap up to yesterday before a weekly search.

Before the tool existed (up to 2026-10-23) there is no window-wise coverage, only topic searches in
[papers.md](papers.md); the days from 2026-10-24 on are the first ones to count.

| Source | Topics | Scope | From | To | Requests | Hits | New | Complete | Run date |
|---|---|---|---|---|---|---|---|---|---|
"""
COLUMNS = ["source", "topics", "scope", "from", "to", "requests", "hits", "new", "complete", "run"]


def format_row(row: dict) -> str:
    return "| " + " | ".join(str(row[c]) for c in COLUMNS) + " |"


def append_row(row: dict, path: Path = COVERAGE_MD) -> None:
    if not path.exists():
        path.write_text(HEADER, encoding="utf-8")
    text = path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text + format_row(row) + "\n", encoding="utf-8")


def read_rows(path: Path = COVERAGE_MD) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.startswith("|") or len(cells) != len(COLUMNS) or cells[0] in ("Source", "---"):
            continue
        rows.append(dict(zip(COLUMNS, cells, strict=True)))
    return rows


def covered_days(rows: list[dict], source: str, version: int) -> set[date]:
    out: set[date] = set()
    for row in rows:
        if row["source"] != source or row["scope"] != "all" or row["complete"] != "yes":
            continue
        if not row["topics"].lstrip("v").isdigit() or int(row["topics"].lstrip("v")) < version:
            continue
        out.update(days_between(parse_day(row["from"]), parse_day(row["to"])))
    return out


def gaps(rows: list[dict], sources: list[str], version: int, span: tuple[date, date]) -> dict[str, list]:
    """Per source the runs of consecutive uncovered days inside `span` (first and last day, inclusive)."""
    out: dict[str, list] = {}
    for source in sources:
        have = covered_days(rows, source, version)
        runs: list[tuple[date, date]] = []
        for day in days_between(*span):
            if day in have:
                continue
            if runs and (day - runs[-1][1]).days == 1:
                runs[-1] = (runs[-1][0], day)
            else:
                runs.append((day, day))
        out[source] = runs
    return out
