#!/usr/bin/env python3
"""Paper scan: arXiv and GitHub by date window, complete pagination, coverage book. See README.md."""

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import papers_arxiv as ax
import papers_book as book
import papers_common as pc
import papers_github as gh
import papers_screen as scr

SOURCES = ["arxiv", "github-pr", "github-issue"]


def tag_item(text: str, topics: list[dict]) -> list[str]:
    low = text.lower()
    return [t["id"] for t in topics if any(term.lower() in low for term in t.get("github_terms", []))]


def scan_arxiv(topics: list[dict], window: tuple[date, date], step: int, fetch=ax.http_fetch, pacer=None, sleep=None):
    """Returns (records by id, requests, complete)."""
    pacer = pacer or pc.Pacer(3.0)
    found: dict[str, dict] = {}
    requests, complete = 0, True
    for lo, hi in pc.day_windows(*window, step):
        for topic in topics:
            res = ax.run_query(ax.window_query(topic, lo, hi), fetch, pacer, sleep)
            requests += res.requests
            complete = complete and res.complete
            print(
                f"arxiv {lo}..{hi} {topic['id']}: {len(res.entries)}/{res.total} complete={res.complete}",
                file=sys.stderr,
            )
            for entry in res.entries:
                rec = found.setdefault(entry["id"], {**entry, "topics": []})
                if topic["id"] not in rec["topics"]:
                    rec["topics"].append(topic["id"])
    return found, requests, complete


def cached_search(cache: Path | None, name: str, run) -> gh.SearchResult:
    """A finished search is kept in `cache`, so an interrupted run resumes instead of starting over."""
    file = cache / (re.sub(r"[^\w.-]+", "_", name) + ".json") if cache else None
    if file and file.exists():
        saved = json.loads(file.read_text(encoding="utf-8"))
        return gh.SearchResult(saved["items"], 0, True)
    res = run()
    if file and res.complete:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(json.dumps({"items": res.items}), encoding="utf-8")
    return res


def scan_github(
    topics: list[dict], repos: list[str], window: tuple[date, date], kinds: set[str], net: gh.Net | None = None
):
    """Returns (records by key, requests, complete). Finished searches are cached under work/papers/cache."""
    # 2.5 s: the search API allows 30 requests per minute
    net = net or gh.Net(gh.gh_call, pc.Pacer(2.5), time.sleep)
    first, last = window
    lo, hi = gh.window_bounds(first, last)
    cache = pc.OUT_ROOT / "cache" / f"{first}_{last}"
    found: dict[str, dict] = {}
    requests, complete = 0, True

    def take(name: str, run) -> None:
        nonlocal requests, complete
        res = cached_search(cache, name, run)
        requests += res.requests
        complete = complete and res.complete
        print(f"github {name} {first}..{last}: {len(res.items)} complete={res.complete}", file=sys.stderr)
        for item in res.items:
            rec = found.setdefault(item["key"], {**item, "terms": []})
            rec["terms"] = sorted(set(rec["terms"]) | set(item.get("terms", [])))

    for repo in repos:
        if "pr" in kinds:
            take(f"{repo} prs", lambda repo=repo: gh.search_prs(repo, lo, hi, net))
        if "issue" in kinds:
            for term in sorted({t for topic in topics for t in topic.get("github_terms", [])}):
                take(f"{repo} issues {term}", lambda repo=repo, term=term: gh.search_issues(repo, term, lo, hi, net))
    for rec in found.values():
        rec["topics"] = tag_item(rec["title"] + " " + rec["body"], topics)
    return found, requests, complete


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# The arXiv API lists a submission only after its announcement, the next weekday evening (US Eastern); a Friday
# submission appears on Sunday night. Seen on 2026-11-12: the 11th still had 0 entries while the 10th had 1,136.
ARXIV_LAG_DAYS = 3


def arxiv_settled(last: date, today: date | None = None) -> bool:
    """True when every day up to `last` is old enough to be listed by the arXiv API."""
    return (today or date.today()) - last >= timedelta(days=ARXIV_LAG_DAYS)


def check_window(first: date, last: date) -> None:
    if first > last:
        sys.exit("--from is after --to")
    if last > pc.yesterday():
        sys.exit(f"--to {last} is not over yet: the last complete day is {pc.yesterday()}")


def select_topics(data: dict, wanted: list[str] | None) -> list[dict]:
    topics = data["topic"]
    if wanted:
        unknown = set(wanted) - {t["id"] for t in topics}
        if unknown:
            sys.exit(f"unknown topic: {', '.join(sorted(unknown))}")
        topics = [t for t in topics if t["id"] in wanted]
    return topics


@dataclass
class Outcome:
    records: dict
    requests: int
    complete: bool
    scope: str
    known: set[str]


def finish(source: str, args, data: dict, out: Outcome) -> None:
    """Sort, mark known entries, write the candidates, add the book row."""
    records, requests, complete, scope, known = out.records, out.requests, out.complete, out.scope, out.known
    key = "id" if source == "arxiv" else "key"
    items = sorted(records.values(), key=lambda r: r.get("date") or r.get("created") or "")
    for rec in items:
        rec["known"] = rec[key] in known
    today = date.today().isoformat()
    out = pc.OUT_ROOT / today / f"{source}-{args.first}_{args.last}.jsonl"
    write_jsonl(out, items)
    new = sum(1 for r in items if not r["known"])
    row = {
        "source": source,
        "topics": data["version"],
        "scope": scope,
        "from": args.first,
        "to": args.last,
        "requests": requests,
        "hits": len(items),
        "new": new,
        "complete": "yes" if complete else "no",
        "run": today,
    }
    if not args.no_book:
        book.append_row(row)
    print(f"{source}: {len(items)} hits, {new} not in papers.md, complete={complete}, requests={requests} -> {out}")


def cmd_arxiv(args) -> int:
    check_window(args.first, args.last)
    data = pc.load_topics()
    topics = select_topics(data, args.topic)
    records, requests, complete = scan_arxiv(topics, (args.first, args.last), args.step)
    if not arxiv_settled(args.last):
        print(f"arxiv: {args.last} is younger than {ARXIV_LAG_DAYS} days; the API may not list it yet, complete=no",
              file=sys.stderr)  # fmt: skip
        complete = False
    scope = "all" if len(topics) == len(data["topic"]) else "+".join(t["id"] for t in topics)
    finish("arxiv", args, data, Outcome(records, requests, complete, scope, pc.read_known()[0]))
    return 0 if complete else 1


def cmd_github(args) -> int:
    check_window(args.first, args.last)
    data = pc.load_topics()
    repos = args.repo or gh.REPOS
    kinds = {"pr", "issue"} if args.kind == "both" else {args.kind}
    records, requests, complete = scan_github(data["topic"], repos, (args.first, args.last), kinds)
    scope = "all" if repos == gh.REPOS else "+".join(repos)
    known = pc.read_known()[1]
    for kind in sorted(kinds):
        part = {k: v for k, v in records.items() if v["kind"] == kind}
        finish(f"github-{kind}", args, data, Outcome(part, requests, complete, scope, known))
    return 0 if complete else 1


def cmd_coverage(args) -> int:
    data = pc.load_topics()
    found = book.gaps(book.read_rows(), SOURCES, data["version"], (args.since, args.last))
    bad = False
    for source, runs in found.items():
        if runs:
            bad = True
            print(f"{source}: {sum((b - a).days + 1 for a, b in runs)} day(s) without complete coverage")
            for a, b in runs:
                print(f"  {a}..{b}   scan.py {'arxiv' if source == 'arxiv' else 'github'} --from {a} --to {b}")
        else:
            print(f"{source}: complete {args.since}..{args.last} (topics v{data['version']})")
    return 1 if bad else 0


def cmd_screen(args) -> int:
    """Writes one row per candidate into the screened book; ids without an override get --default."""
    records = [json.loads(line) for line in args.candidates.read_text(encoding="utf-8").splitlines() if line.strip()]
    overrides = scr.load_overrides(args.decisions) if args.decisions else {}
    decision, _, reason = args.default.partition(":")
    if decision not in scr.DECISIONS or not reason:
        sys.exit("--default is decision:reason, decision one of " + "/".join(scr.DECISIONS))
    rows = scr.screen_rows(records, overrides, (decision, reason), args.method)
    added = scr.add_screened(rows)
    scr.add_watch(scr.watch_rows(records, overrides, date.today().isoformat()))
    print(f"screened: {added} new rows ({len(rows) - added} were already in the book)")
    return 0


def cmd_watch(args) -> int:
    changes = scr.refresh_watch()
    print("\n".join(changes) if changes else "watch list: no state changed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("arxiv", cmd_arxiv), ("github", cmd_github)):
        p = sub.add_parser(name)
        p.add_argument("--from", dest="first", type=pc.parse_day, required=True)
        p.add_argument("--to", dest="last", type=pc.parse_day, required=True)
        p.add_argument("--no-book", action="store_true", help="do not write a coverage row (trial run)")
        if name == "arxiv":
            p.add_argument("--step", type=int, default=7, help="days per query window (default 7)")
            p.add_argument("--topic", action="append", help="only this topic id (scope is then not `all`)")
        else:
            p.add_argument("--repo", action="append", help="only this repository (scope is then not `all`)")
            p.add_argument("--kind", choices=["pr", "issue", "both"], default="both")
        p.set_defaults(fn=fn)
    p = sub.add_parser("screen", help="record decisions (drop/note/candidate) for a candidates file")
    p.add_argument("candidates", type=Path)
    p.add_argument("--decisions", type=Path, help="TSV: id, decision, reason[, method] (no header)")
    p.add_argument("--method", choices=scr.METHODS, default="title-read", help="method of rows without their own")
    p.add_argument("--default", default="drop:off-topic", help="decision:reason for ids without a line")
    p.set_defaults(fn=cmd_screen)
    p = sub.add_parser("watch", help="re-check the open PRs of research/papers-watch.tsv")
    p.set_defaults(fn=cmd_watch)
    p = sub.add_parser("coverage", help="list days without complete coverage up to yesterday")
    p.add_argument("--since", type=pc.parse_day, default=date(2026, 10, 24))
    p.add_argument("--to", dest="last", type=pc.parse_day, default=pc.yesterday())
    p.set_defaults(fn=cmd_coverage)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
