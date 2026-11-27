"""arXiv export API: one query per topic and date window, paginated until totalResults is reached."""

import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date

from papers_common import ARXIV_ID, USER_AGENT, Pacer

API = "https://export.arxiv.org/api/query"
PAGE = 100  # arXiv allows up to 2000; 100 keeps one response small enough to retry cheaply
ATTEMPTS = 8  # arXiv answers 429 for minutes when several scans share an IP; 30 s steps ride that out
NS = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/"}


@dataclass
class QueryResult:
    entries: list[dict] = field(default_factory=list)
    total: int = 0
    requests: int = 0
    complete: bool = False


def window_query(topic: dict, first: date, last: date) -> str:
    """`submittedDate` window AND topic expression AND categories (all of them parenthesized)."""
    stamp = f"submittedDate:[{first:%Y%m%d}0000 TO {last:%Y%m%d}2359]"
    parts = [stamp, f"({topic['query']})"]
    cats = topic.get("categories")
    if cats:
        parts.append("(" + " OR ".join(f"cat:{c}" for c in cats) + ")")
    return " AND ".join(parts)


def page_url(query: str, start: int) -> str:
    args = {
        "search_query": query,
        "start": start,
        "max_results": PAGE,
        "sortBy": "submittedDate",
        "sortOrder": "ascending",  # stable under pagination: new submissions land at the end
    }
    return f"{API}?{urllib.parse.urlencode(args, quote_via=urllib.parse.quote)}"


def http_fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def parse_feed(xml: bytes) -> tuple[int, list[dict]]:
    """(totalResults, entries) of one Atom page; raises ValueError on an error feed."""
    root = ET.fromstring(xml)
    total = root.find("o:totalResults", NS)
    if total is None:
        raise ValueError("no totalResults in feed")
    entries = []
    for ent in root.findall("a:entry", NS):
        raw_id = ent.findtext("a:id", "", NS)
        if "/api/errors" in raw_id:
            raise ValueError(ent.findtext("a:summary", "arXiv error", NS))
        match = ARXIV_ID.search(raw_id)
        if not match:
            continue
        version = raw_id.rsplit("v", 1)[-1]
        entries.append(
            {
                "id": match.group(1),
                "version": int(version) if version.isdigit() else 1,
                "title": " ".join(ent.findtext("a:title", "", NS).split()),
                "authors": len(ent.findall("a:author", NS)),
                "date": ent.findtext("a:published", "", NS)[:10],
                "categories": [c.get("term") for c in ent.findall("a:category", NS)],
                "abstract": " ".join(ent.findtext("a:summary", "", NS).split()),
            }
        )
    return int(total.text), entries


def fetch_page(url: str, fetch, pacer: Pacer, sleep) -> tuple[int, list[dict], int]:
    """One page with retries; returns (total, entries, requests used). Raises after ATTEMPTS failures."""
    for attempt in range(1, ATTEMPTS + 1):
        pacer.wait()
        try:
            total, entries = parse_feed(fetch(url))
            return total, entries, attempt
        except (urllib.error.URLError, ET.ParseError, ValueError, TimeoutError, OSError):
            if attempt == ATTEMPTS:
                raise
            sleep(30 * attempt)
    raise AssertionError("unreachable")


def run_query(query: str, fetch=http_fetch, pacer: Pacer | None = None, sleep=None) -> QueryResult:
    """All pages of one query. `complete` is true only when as many entries were seen as totalResults says."""
    pacer = pacer or Pacer(3.0)
    sleep = sleep or time.sleep
    res = QueryResult()
    start = 0
    while True:
        try:
            total, entries, used = fetch_page(page_url(query, start), fetch, pacer, sleep)
        except (urllib.error.URLError, ET.ParseError, ValueError, TimeoutError, OSError):
            res.requests += ATTEMPTS
            return res
        res.requests += used
        res.total = total
        res.entries.extend(entries)
        start += PAGE
        if start >= total:
            break
        if not entries:  # arXiv sometimes returns an empty page before the end: do not claim completeness
            return res
    res.complete = len(res.entries) >= res.total
    return res
