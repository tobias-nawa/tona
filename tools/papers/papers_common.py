"""Shared helpers of the paper scan: topics, date windows, known entries, pacing."""

import re
import time
import tomllib
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOPICS_FILE = Path(__file__).resolve().parent / "topics.toml"
PAPERS_MD = ROOT / "research" / "papers.md"
COVERAGE_MD = ROOT / "research" / "papers-coverage.md"
OUT_ROOT = ROOT / "work" / "papers"
SCREENED_TSV = ROOT / "research" / "papers-screened.tsv"
# Mass-dropped GitHub hits (not in the repository: ~30 MB for a backfill of 22 months); the coverage book is the proof
GITHUB_DROPS_TSV = OUT_ROOT / "screened-github-drops.tsv"

# No personal data: the tool name only.
USER_AGENT = "tona-paper-scan/1 (open-source inference engine research)"

ARXIV_ID = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?![\d])")
GH_ITEM = re.compile(r"github\.com/([\w.-]+/[\w.-]+)/(?:pull|issues)/(\d+)")


def load_topics(path: Path = TOPICS_FILE) -> dict:
    """The topic file with its `version` and the list `topic`."""
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    if "version" not in data or not data.get("topic"):
        raise ValueError(f"{path}: needs `version` and at least one [[topic]]")
    return data


def parse_day(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def days_between(first: date, last: date) -> list[date]:
    return [first + timedelta(days=n) for n in range((last - first).days + 1)]


def day_windows(first: date, last: date, step: int) -> list[tuple[date, date]]:
    """Consecutive windows of `step` days that cover first..last without gap or overlap."""
    out = []
    cur = first
    while cur <= last:
        end = min(cur + timedelta(days=step - 1), last)
        out.append((cur, end))
        cur = end + timedelta(days=1)
    return out


def known_ids(text: str) -> tuple[set[str], set[str]]:
    """arXiv IDs and `repo#number` keys that already stand in the paper overview."""
    arxiv = {m.group(1) for m in ARXIV_ID.finditer(text)}
    gh = {f"{m.group(1)}#{m.group(2)}" for m in GH_ITEM.finditer(text)}
    return arxiv, gh


def read_known(
    path: Path = PAPERS_MD, screened: Path | None = None, drops: Path | None = None
) -> tuple[set[str], set[str]]:
    """Known = mentioned in the paper overview or already screened (any decision, also the local GitHub drops)."""
    arxiv, gh = known_ids(path.read_text(encoding="utf-8")) if path.exists() else (set(), set())
    for book in (screened if screened is not None else SCREENED_TSV, drops if drops is not None else GITHUB_DROPS_TSV):
        if book.exists():
            for line in book.read_text(encoding="utf-8").splitlines()[1:]:
                ident = line.split("\t", 1)[0]
                (gh if "#" in ident else arxiv).add(ident)
    return arxiv, gh


class Pacer:
    """Waits at least `gap` seconds between two requests (arXiv asks for 3 s)."""

    def __init__(self, gap: float, sleep=time.sleep, clock=time.monotonic):
        self.gap, self.sleep, self.clock, self.last = gap, sleep, clock, None

    def wait(self) -> None:
        if self.last is not None:
            rest = self.gap - (self.clock() - self.last)
            if rest > 0:
                self.sleep(rest)
        self.last = self.clock()


def yesterday() -> date:
    return date.today() - timedelta(days=1)
