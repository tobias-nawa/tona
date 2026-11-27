"""GitHub search through `gh api search/issues`: created-windows that are split until each holds at most 1000 hits."""

import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

REPOS = [
    "ggml-org/llama.cpp",
    "vllm-project/vllm",
    "sgl-project/sglang",
    "ml-explore/mlx",
    "ml-explore/mlx-lm",
    # ROCm/ROCm is renamed to legacy-rocm-build and cannot be searched; the work moved to these three
    "ROCm/rocm-systems",
    "ROCm/TheRock",
    "ROCm/rocm-libraries",
    "ggml-org/ggml",
]
PER_PAGE = 100
SEARCH_CAP = 1000  # the search API never returns more than 1000 results per query
FMT = "%Y-%m-%dT%H:%M:%SZ"


@dataclass
class Net:
    """The outside world of a search: the call, the pacing between calls and the sleep (all replaceable in tests)."""

    call: object
    pacer: object
    sleep: object


@dataclass
class Query:
    base: str  # search text without the created clause
    repo: str
    kind: str  # pr or issue


@dataclass
class SearchResult:
    items: list[dict] = field(default_factory=list)
    requests: int = 0
    complete: bool = True


def gh_call(query: str, page: int) -> dict:
    """One `search/issues` page. Raises RuntimeError with gh's message on failure."""
    cmd = [
        "gh",
        "api",
        "-X",
        "GET",
        "search/issues",
        "-f",
        f"q={query}",
        "-f",
        f"per_page={PER_PAGE}",
        "-f",
        f"page={page}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip())
    return json.loads(proc.stdout)


def window_bounds(first: date, last: date) -> tuple[datetime, datetime]:
    lo = datetime(first.year, first.month, first.day, tzinfo=UTC)
    hi = datetime(last.year, last.month, last.day, 23, 59, 59, tzinfo=UTC)
    return lo, hi


def created_clause(lo: datetime, hi: datetime) -> str:
    return f"created:{lo.strftime(FMT)}..{hi.strftime(FMT)}"


def search_page(query: str, page: int, net: Net, state: SearchResult) -> dict | None:
    """One page with retries (rate limit: wait a minute). None after repeated failure."""
    for attempt in range(1, 6):
        net.pacer.wait()
        state.requests += 1
        try:
            data = net.call(query, page)
        except RuntimeError as exc:
            if "cannot be searched" in str(exc):  # renamed or private repository: retrying cannot help
                return None
            net.sleep(65 if "rate limit" in str(exc).lower() else 5 * attempt)
            continue
        if data.get("incomplete_results"):
            net.sleep(5 * attempt)
            continue
        return data
    return None


def item_record(item: dict, repo: str, kind: str) -> dict:
    return {
        "key": f"{repo}#{item['number']}",
        "repo": repo,
        "kind": kind,
        "number": item["number"],
        "title": item.get("title", ""),
        "state": item.get("state", ""),
        "created": (item.get("created_at") or "")[:10],
        "labels": [lab.get("name", "") for lab in item.get("labels", [])],
        "comments": item.get("comments", 0),
        "body": " ".join((item.get("body") or "").split())[:500],
    }


def search_window(q: Query, lo: datetime, hi: datetime, net: Net) -> SearchResult:
    """All items of `base` created in lo..hi; a window with more than 1000 hits is halved and searched again."""
    res = SearchResult()
    first = search_page(f"{q.base} {created_clause(lo, hi)}", 1, net, res)
    if first is None:
        res.complete = False
        return res
    total = first.get("total_count", 0)
    if total > SEARCH_CAP:
        if hi - lo < timedelta(seconds=2):
            res.complete = False
            return res
        mid = lo + (hi - lo) / 2
        mid = mid.replace(microsecond=0)
        for a, b in ((lo, mid), (mid + timedelta(seconds=1), hi)):
            part = search_window(q, a, b, net)
            res.items.extend(part.items)
            res.requests += part.requests
            res.complete = res.complete and part.complete
        return res
    data, page = first, 1
    while True:
        res.items.extend(item_record(i, q.repo, q.kind) for i in data.get("items", []))
        if page * PER_PAGE >= total:
            break
        page += 1
        data = search_page(f"{q.base} {created_clause(lo, hi)}", page, net, res)
        if data is None:
            res.complete = False
            return res
    res.complete = res.complete and len(res.items) >= total
    return res


def search_prs(repo: str, lo: datetime, hi: datetime, net: Net) -> SearchResult:
    return search_window(Query(f"repo:{repo} is:pr", repo, "pr"), lo, hi, net)


def search_issues(repo: str, term: str, lo: datetime, hi: datetime, net: Net) -> SearchResult:
    quoted = f'"{term}"' if " " in term else term
    q = Query(f"repo:{repo} is:issue {quoted} in:title,body", repo, "issue")
    res = search_window(q, lo, hi, net)
    for item in res.items:
        item["terms"] = [term]
    return res
