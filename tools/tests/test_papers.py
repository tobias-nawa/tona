"""Paper scan: pagination, window splitting, matching against known entries, gap detection. No network."""

import itertools
from datetime import UTC, date, datetime, timedelta

import papers_arxiv as ax
import papers_book as book
import papers_common as pc
import papers_github as gh
import papers_screen as scr
import pytest
import scan


def feed(total: int, ids: list[str]) -> bytes:
    entries = "".join(
        f"""<entry><id>http://arxiv.org/abs/{i}v2</id><title> Title
 {i} </title><published>2026-11-09T10:00:00Z</published><summary>Abs   {i}</summary>
<author><name>A</name></author><author><name>B</name></author>
<category term="cs.LG"/><category term="cs.CL"/></entry>"""
        for i in ids
    )
    return (
        '<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
        f"<opensearch:totalResults>{total}</opensearch:totalResults>{entries}</feed>"
    ).encode()


class NoWait(pc.Pacer):
    def __init__(self):
        super().__init__(0.0)

    def wait(self):
        self.waits = getattr(self, "waits", 0) + 1


def pages(total: int):
    """A fake arXiv that serves `total` entries in pages of ax.PAGE, by the `start` of the URL."""

    def fetch(url: str) -> bytes:
        start = int(url.split("start=")[1].split("&", maxsplit=1)[0])
        ids = [f"2611.{n:05d}" for n in range(start, min(start + ax.PAGE, total))]
        return feed(total, ids)

    return fetch


def test_parse_feed_fields():
    total, entries = ax.parse_feed(feed(1, ["2611.00001"]))
    assert total == 1
    assert entries[0] == {
        "id": "2611.00001",
        "version": 2,
        "title": "Title 2611.00001",
        "authors": 2,
        "date": "2026-11-09",
        "categories": ["cs.LG", "cs.CL"],
        "abstract": "Abs 2611.00001",
    }


def test_arxiv_pagination_reads_every_page(monkeypatch):
    monkeypatch.setattr(ax, "PAGE", 100)
    res = ax.run_query("q", pages(250), NoWait(), lambda s: None)
    assert res.complete and res.total == 250 and len(res.entries) == 250
    assert res.requests == 3
    assert len({e["id"] for e in res.entries}) == 250


def test_arxiv_empty_window_is_complete():
    res = ax.run_query("q", pages(0), NoWait(), lambda s: None)
    assert res.complete and res.entries == [] and res.requests == 1


def test_arxiv_short_page_is_not_complete():
    def fetch(url: str) -> bytes:
        return feed(250, [] if "start=100" in url else [f"2611.{n:05d}" for n in range(100)])

    res = ax.run_query("q", fetch, NoWait(), lambda s: None)
    assert not res.complete and len(res.entries) == 100


def test_arxiv_retries_then_succeeds():
    calls = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        if len(calls) < 3:
            raise OSError("boom")
        return feed(1, ["2611.00001"])

    sleeps = []
    res = ax.run_query("q", fetch, NoWait(), sleeps.append)
    assert res.complete and len(calls) == 3 and sleeps == [30, 60]


def test_arxiv_gives_up_after_attempts():
    def fetch(url: str) -> bytes:
        raise OSError("down")

    res = ax.run_query("q", fetch, NoWait(), lambda s: None)
    assert not res.complete and res.requests == ax.ATTEMPTS


def test_arxiv_error_feed_is_an_error():
    bad = (
        b'<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
        b"<opensearch:totalResults>1</opensearch:totalResults><entry><id>http://arxiv.org/api/errors#x</id>"
        b"<summary>bad query</summary></entry></feed>"
    )
    with pytest.raises(ValueError, match="bad query"):
        ax.parse_feed(bad)


def test_window_query_has_window_topic_and_categories():
    q = ax.window_query({"query": "abs:MoE", "categories": ["cs.LG", "cs.AR"]}, date(2026, 11, 1), date(2026, 11, 7))
    assert q == "submittedDate:[202611010000 TO 202611072359] AND (abs:MoE) AND (cat:cs.LG OR cat:cs.AR)"


def test_day_windows_cover_without_gap():
    wins = pc.day_windows(date(2026, 10, 24), date(2026, 11, 10), 7)
    assert wins[0][0] == date(2026, 10, 24) and wins[-1][1] == date(2026, 11, 10)
    assert all((b[0] - a[1]).days == 1 for a, b in itertools.pairwise(wins))
    assert wins[-1] == (date(2026, 11, 7), date(2026, 11, 10))


def test_known_ids_from_papers_text():
    text = "[a](https://arxiv.org/abs/2610.13778v1) 2609.12978 x12345.6789 [PR](https://github.com/ggml-org/llama.cpp/pull/31853)"
    arxiv, ghub = pc.known_ids(text)
    assert arxiv == {"2610.13778", "2609.12978"}
    assert ghub == {"ggml-org/llama.cpp#31853"}


def test_pacer_waits_the_gap():
    now = [0.0]
    slept = []
    pacer = pc.Pacer(3.0, sleep=slept.append, clock=lambda: now[0])
    pacer.wait()
    now[0] = 1.0
    pacer.wait()
    assert slept == [2.0]


def fake_search(per_day: dict[date, int]):
    """A fake search/issues: `per_day` items per UTC day, items identified by their timestamp."""
    stamps = []
    for day, n in sorted(per_day.items()):
        for k in range(n):
            stamps.append(datetime(day.year, day.month, day.day, 0, 0, tzinfo=UTC) + timedelta(seconds=k * 60))
    calls = []

    def call(query: str, page: int) -> dict:
        calls.append((query, page))
        lo, hi = query.split("created:")[1].split("..")
        lo = datetime.strptime(lo, gh.FMT).replace(tzinfo=UTC)
        hi = datetime.strptime(hi, gh.FMT).replace(tzinfo=UTC)
        hits = [s for s in stamps if lo <= s <= hi]
        chunk = hits[(page - 1) * gh.PER_PAGE : page * gh.PER_PAGE]
        if page * gh.PER_PAGE > gh.SEARCH_CAP:
            raise RuntimeError("Only the first 1000 search results are available")
        items = [{"number": int(s.timestamp()), "title": "t", "created_at": s.strftime(gh.FMT)} for s in chunk]
        return {"total_count": len(hits), "incomplete_results": False, "items": items}

    return call, calls


def net(call):
    return gh.Net(call, NoWait(), lambda s: None)


def test_github_small_window_paginates():
    call, calls = fake_search({date(2026, 11, 1): 250})
    lo, hi = gh.window_bounds(date(2026, 11, 1), date(2026, 11, 1))
    res = gh.search_prs("a/b", lo, hi, net(call))
    assert res.complete and len(res.items) == 250 and len(calls) == 3
    assert len({i["key"] for i in res.items}) == 250


def test_github_window_split_below_the_cap():
    per_day = {date(2026, 11, 1) + timedelta(days=d): 600 for d in range(3)}  # 1800 hits in the full window
    call, calls = fake_search(per_day)
    lo, hi = gh.window_bounds(date(2026, 11, 1), date(2026, 11, 3))
    res = gh.search_prs("a/b", lo, hi, net(call))
    assert res.complete and len(res.items) == 1800
    assert len({i["key"] for i in res.items}) == 1800
    assert all(page * gh.PER_PAGE <= gh.SEARCH_CAP for _, page in calls)


def test_github_unsplittable_window_is_incomplete():
    # 1001 hits in the same second cannot be split further
    stamp = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)

    def same_second(query: str, page: int) -> dict:
        return {
            "total_count": 1001,
            "incomplete_results": False,
            "items": [{"number": 1, "created_at": stamp.strftime(gh.FMT)}],
        }

    lo, hi = gh.window_bounds(date(2026, 11, 1), date(2026, 11, 1))
    res = gh.search_prs("a/b", lo, hi, net(same_second))
    assert not res.complete


def test_github_rate_limit_waits_and_retries():
    state = {"n": 0}

    def call(query: str, page: int) -> dict:
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("API rate limit exceeded")
        return {"total_count": 0, "incomplete_results": False, "items": []}

    sleeps = []
    lo, hi = gh.window_bounds(date(2026, 11, 1), date(2026, 11, 1))
    res = gh.search_prs("a/b", lo, hi, gh.Net(call, NoWait(), sleeps.append))
    assert res.complete and sleeps == [65]


def test_tagging_by_terms():
    topics = [{"id": "spec", "github_terms": ["MTP"]}, {"id": "kv", "github_terms": ["KV cache"]}]
    assert scan.tag_item("Add mtp draft and kv cache reuse", topics) == ["spec", "kv"]
    assert scan.tag_item("fix typo", topics) == []


def row(source, first, last, complete="yes", scope="all", topics="1"):
    return {
        "source": source,
        "topics": topics,
        "scope": scope,
        "from": first,
        "to": last,
        "requests": 1,
        "hits": 1,
        "new": 0,
        "complete": complete,
        "run": "2026-11-11",
    }


def test_gaps_find_uncovered_days():
    rows = [row("arxiv", "2026-10-24", "2026-10-30"), row("arxiv", "2026-11-04", "2026-11-05", complete="no")]
    found = book.gaps(rows, ["arxiv", "github-pr"], 1, (date(2026, 10, 24), date(2026, 11, 6)))
    assert found["arxiv"] == [(date(2026, 10, 31), date(2026, 11, 6))]
    assert found["github-pr"] == [(date(2026, 10, 24), date(2026, 11, 6))]


def test_gaps_ignore_partial_scope_and_old_topic_version():
    rows = [
        row("arxiv", "2026-10-24", "2026-10-26", scope="moe-offload"),
        row("arxiv", "2026-10-27", "2026-10-28", topics="1"),
    ]
    found = book.gaps(rows, ["arxiv"], 2, (date(2026, 10, 24), date(2026, 10, 28)))
    assert found["arxiv"] == [(date(2026, 10, 24), date(2026, 10, 28))]


def test_book_roundtrip(tmp_path):
    path = tmp_path / "cov.md"
    book.append_row(row("arxiv", "2026-10-24", "2026-10-30"), path)
    book.append_row(row("github-pr", "2026-10-24", "2026-10-30", complete="no"), path)
    rows = book.read_rows(path)
    assert [r["source"] for r in rows] == ["arxiv", "github-pr"]
    assert rows[1]["complete"] == "no"
    assert book.gaps(rows, ["arxiv"], 1, (date(2026, 10, 24), date(2026, 10, 30)))["arxiv"] == []


def test_scan_arxiv_merges_topics_and_reports_completeness(monkeypatch):
    topics = [{"id": "a", "query": "abs:x"}, {"id": "b", "query": "abs:y"}]
    found, requests, complete = scan.scan_arxiv(
        topics, (date(2026, 11, 1), date(2026, 11, 14)), 7, pages(3), NoWait(), lambda s: None
    )
    assert complete and requests == 4  # 2 windows x 2 topics
    assert sorted(found) == ["2611.00000", "2611.00001", "2611.00002"]
    assert found["2611.00001"]["topics"] == ["a", "b"]


def test_topics_file_is_valid():
    data = pc.load_topics()
    ids = [t["id"] for t in data["topic"]]
    assert isinstance(data["version"], int) and len(ids) == len(set(ids)) >= 11
    assert all(t["query"] and t["github_terms"] for t in data["topic"])


def test_check_window_refuses_today():
    with pytest.raises(SystemExit):
        scan.check_window(date(2026, 1, 1), date.today())


def test_screen_rows_default_and_override():
    recs = [
        {"id": "2611.00001", "date": "2026-11-01", "topics": ["moe-offload"]},
        {"key": "a/b#7", "kind": "pr", "created": "2026-11-02", "topics": []},
    ]
    rows = scr.screen_rows(recs, {"2611.00001": ("candidate", "high")}, ("drop", "off-topic"))
    assert [(r["id"], r["source"], r["decision"], r["reason"]) for r in rows] == [
        ("2611.00001", "arxiv", "candidate", "high"),
        ("a/b#7", "github-pr", "drop", "off-topic"),
    ]
    assert rows[1]["topics"] == "-"


def test_screen_methods_and_legacy_rows(tmp_path):
    recs = [{"id": "2611.00001", "date": "d"}, {"id": "2611.00002", "date": "d"}, {"id": "2611.00003", "date": "d"}]
    ov = {"2611.00001": ("note", "x", "abstract-read"), "2611.00002": ("drop", "title-filter")}
    rows = scr.screen_rows(recs, ov, ("drop", "off-topic"))
    assert [r["method"] for r in rows] == ["abstract-read", "keyword-filter", "title-read"]
    assert scr.screen_rows(recs[:1], {}, ("drop", "off-topic"), "fulltext-read")[0]["method"] == "fulltext-read"
    with pytest.raises(ValueError, match="method must be"):
        scr.screen_rows(recs, {}, ("drop", "x"), "guess")
    old = tmp_path / "old.tsv"
    old.write_text(
        "id\tdate\tsource\ttopics\tdecision\treason\na\td\tarxiv\t-\tdrop\tbulk-no-keyword\n"
        "b\td\tarxiv\t-\tdrop\toff-topic\n"
    )
    assert [r["method"] for r in scr.read_table(old, scr.SCREENED_HEAD)] == ["keyword-filter", "title-read"]
    bad = tmp_path / "m.tsv"
    bad.write_text("x\tdrop\ty\tguess\n")
    with pytest.raises(ValueError, match="method must be"):
        scr.load_overrides(bad)


def test_screen_rejects_unknown_override_and_decision(tmp_path):
    with pytest.raises(ValueError, match="not in the candidates"):
        scr.screen_rows([{"id": "2611.00001"}], {"2611.99999": ("drop", "x")}, ("drop", "y"))
    bad = tmp_path / "d.tsv"
    bad.write_text("2611.00001\tmaybe\tx\n")
    with pytest.raises(ValueError, match="decision must be"):
        scr.load_overrides(bad)


def test_screened_book_dedupes_and_marks_known(tmp_path):
    path = tmp_path / "s.tsv"
    rows = scr.screen_rows(
        [{"id": "2611.00001", "date": "d"}, {"key": "a/b#7", "kind": "pr", "created": "d"}], {}, ("drop", "off-topic")
    )
    drops = tmp_path / "drops.tsv"
    assert scr.add_screened(rows, path, drops) == 2
    assert scr.add_screened(rows, path, drops) == 0
    arxiv, ghub = pc.read_known(tmp_path / "none.md", path, drops)
    assert arxiv == {"2611.00001"} and ghub == {"a/b#7"}
    # the GitHub drop is in the local file only, the arXiv drop stays in the book
    assert [r["id"] for r in scr.read_table(path, scr.SCREENED_HEAD)] == ["2611.00001"]
    assert [r["id"] for r in scr.read_table(drops, scr.SCREENED_HEAD)] == ["a/b#7"]


def test_screened_book_keeps_github_notes(tmp_path):
    path, drops = tmp_path / "s.tsv", tmp_path / "drops.tsv"
    rows = scr.screen_rows(
        [{"key": "a/b#1", "kind": "pr", "created": "d"}, {"key": "a/b#2", "kind": "pr", "created": "d"}],
        {"a/b#1": ("note", "x")},
        ("drop", "off-topic"),
    )
    scr.add_screened(rows, path, drops)
    assert [r["id"] for r in scr.read_table(path, scr.SCREENED_HEAD)] == ["a/b#1"]
    assert [r["id"] for r in scr.read_table(drops, scr.SCREENED_HEAD)] == ["a/b#2"]


def test_watch_list_and_refresh(tmp_path):
    recs = [
        {"key": "a/b#1", "kind": "pr", "state": "open", "title": "t1", "created": "2026-11-01"},
        {"key": "a/b#2", "kind": "pr", "state": "open", "title": "t2", "created": "2026-11-01"},
        {"key": "a/b#3", "kind": "issue", "state": "open", "title": "t3", "created": "2026-11-01"},
    ]
    ov = {"a/b#1": ("candidate", "x"), "a/b#3": ("note", "y")}
    path = tmp_path / "w.tsv"
    scr.add_watch(scr.watch_rows(recs, ov, "2026-11-11"), path)
    assert [r["key"] for r in scr.read_table(path, scr.WATCH_HEAD)] == ["a/b#1"]

    changes = scr.refresh_watch(path, lambda repo, n: "merged", "2026-11-18")
    assert changes == ["a/b#1: open -> merged  t1"]
    row = scr.read_table(path, scr.WATCH_HEAD)[0]
    assert row["state"] == "merged" and row["checked"] == "2026-11-18"
    assert scr.refresh_watch(path, lambda repo, n: "merged", "2026-11-19") == []


def test_github_unsearchable_repo_fails_fast():
    def call(query: str, page: int) -> dict:
        raise RuntimeError("Validation Failed: The listed users and repositories cannot be searched")

    sleeps = []
    lo, hi = gh.window_bounds(date(2026, 11, 1), date(2026, 11, 1))
    res = gh.search_prs("a/b", lo, hi, gh.Net(call, NoWait(), sleeps.append))
    assert not res.complete and sleeps == [] and res.requests == 1


def test_scan_github_resumes_from_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "OUT_ROOT", tmp_path)
    topics = [{"id": "t", "github_terms": ["MTP"]}]
    call, calls = fake_search({date(2026, 11, 1): 3})
    window = (date(2026, 11, 1), date(2026, 11, 1))
    found, _, complete = scan.scan_github(topics, ["a/b"], window, {"pr"}, net(call))
    assert complete and len(found) == 3 and len(calls) == 1
    again, _, complete = scan.scan_github(topics, ["a/b"], window, {"pr"}, net(call))
    assert complete and len(again) == 3 and len(calls) == 1  # served from the cache


def test_arxiv_days_younger_than_the_lag_are_not_complete():
    import scan

    today = date(2026, 11, 12)
    assert not scan.arxiv_settled(date(2026, 11, 11), today)
    assert not scan.arxiv_settled(date(2026, 11, 10), today)
    assert scan.arxiv_settled(date(2026, 11, 9), today)
