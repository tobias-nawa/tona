#!/usr/bin/python3 -I
"""Builds the prose corpus (English and German, public-domain text) that the prose workloads of scripts/ab.py cut
their prompts from: the knowledge-and-research workload of the Bolt model (encyclopedia articles, a long report).

Usage: scripts/make_prose_corpus.py [--out FILE] [--pin]

Sources, all public domain: the 1911 Encyclopaedia Britannica (English) and Meyers Konversations-Lexikon 1888, the
Allgemeine Deutsche Biographie and the German translation of Darwin's "Origin of Species" of 1876 (German), read
from Wikisource at a fixed revision (`oldid`). Each source has the sha256 of the text it yields (page furniture
removed by `clean`); a changed text stops the build. `--pin` prints the revision ids and hashes of the current
pages, to paste into SOURCES when a source is added.

Format: documents of both languages in alternation, each `===== <source> =====`, the text, a blank line.
The `prose` requests start at the first character, the `longdoc` request at LONGDOC_OFFSET
(scripts/ab_workloads.py), so no two requests share a prefix.
"""

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import ClassVar

B = "1911 Encyclopædia Britannica/"
# (wiki, title, oldid, sha256 of the cleaned text); oldid 0 and sha "" until pinned with --pin
EN = [
    ("en", B + "Elephant", 9100539, "de445c799df3034b5da07fb823696922fe09d301852b6b580e4283193779b599"),
    ("en", B + "Electricity", 6815637, "5455c3f0b55204271ab1e582f95bcc3f0a86f9c4effaa18eb4c5769135b188c8"),
    ("en", B + "Evolution", 11396681, "c9316fde547cf2636d2ec482191162637b5a5dddadcfb15018420c3a08ea71aa"),
    ("en", B + "Astronomy", 11302863, "316bfa1e4c8c39fa67add6ba0369eb80071ca6afa339022878dae1a7dcff48c9"),
    ("en", B + "Railways", 15317779, "d601bc57e31b13f7f12f9fc67454b4a4d1905fe00060d4ee84141203dbf2925e"),
    ("en", B + "Geology", 11144375, "3ac61f008f4c9fc333d265abd0c05c9763c74b4dfa1b02f8d2817dace38554f3"),
    ("en", B + "Photography", 16333911, "8e2c2081ecc036b7ff0c9ad8f41b6cfec9696e3d62a40ce161413b3b137cb9fb"),
    ("en", B + "Whale", 10868664, "80f0a4961da061517c56cac32da4c2e5938c2136b15b97657bcc39238d480914"),
    ("en", B + "Chemistry", 10436531, "21c0ca543c400b2beb0c4f472cfa8c831978ddda19cc12e31f7997ffbdf80084"),
    ("en", B + "Meteorology", 11518530, "ef1ce59e3eef8f540c0605e945c09f22bf487529d26c78409afdd2b3ff9d0545"),
]
DE = [
    ("de", "ADB:Humboldt, Wilhelm von", 4763748, "6e54740a83c9bb9ff28bc28479a2bcecb1a1253856995f217f28b5c57790050f"),
    ("de", "MKL1888:Elefant", 4582277, "78771192c53a20edb4ee950697ee9777be02449626c310a766883a14296cc2de"),
    ("de", "ADB:Kant, Immanuel", 4763682, "ac9fd8c76453f284851b55197461d13e0515a6c6fd9c03dc19e1c5d97a3546d7"),
    ("de", "MKL1888:Astronomie", 4567935, "77e6445281e44de31a642a643cbaa8b8af90c2591d660543602356d51b11f016"),
    (
        "de",
        "ADB:Leibniz, Gottfried Wilhelm",
        4763693,
        "472110243dff43603791925dfe7ca16bf09e59dc9fc440b8939eb7f7e9aca539",
    ),
    ("de", "MKL1888:Darwinismus", 4567644, "6104dedf4edeffea38d1227e7fd928b51bfe9be643f5a0e2c47a8fda732a3279"),
    ("de", "ADB:Gauß, Carl Friedrich", 4763812, "668339f97b7303685149912b369f54fecefdd31fa4a2d4123d8bfea94c65ca9c"),
    ("de", "MKL1888:Dampfmaschine", 4567600, "bb12325692a0d315fca00ca8e503b6e9a8e2a712a2fbe2426556c784d4424a45"),
    ("de", "ADB:Humboldt, Alexander von", 4763692, "0189e5951129e9e81b2c239d2bbf3998fbfa95b8f5be19d71711bf8aa7e175d9"),
    (
        "de",
        "Entstehung der Arten (1876)/Einleitung",
        1414082,
        "7dc54011a9d767cbd5194ee2a177ade74593fb089e76c1d4191c263816486f15",
    ),
]
USER_AGENT = "tona-corpus-builder/1.0"


class Text(HTMLParser):
    SKIP: ClassVar[set[str]] = {"style", "script", "sup", "table", "noscript", "head"}
    BREAK: ClassVar[set[str]] = {"p", "div", "br", "h1", "h2", "h3", "h4", "li", "tr"}

    def __init__(self) -> None:
        super().__init__()
        self.out: list = []
        self.depth = 0

    def handle_starttag(self, tag, attrs) -> None:
        self.depth += tag in self.SKIP
        if tag in self.BREAK:
            self.out.append("\n")

    def handle_endtag(self, tag) -> None:
        if tag in self.SKIP:
            self.depth = max(0, self.depth - 1)
        if tag in self.BREAK:
            self.out.append("\n")

    def handle_data(self, data) -> None:
        if not self.depth:
            self.out.append(data)


def html_text(page: str) -> str:
    parser = Text()
    parser.feed(page)
    text = re.sub(r"[ \t\xa0]+", " ", "".join(parser.out))
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


def fetch(url: str) -> bytes:
    """One polite GET: Wikimedia answers a fast client with 429, so wait and retry."""
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as r:
                time.sleep(2)
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503):
                raise
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"{url}: still rate limited")


def wikisource(wiki: str, title: str, oldid: int) -> tuple:
    q = {"action": "parse", "prop": "text|revid", "disableeditsection": 1, "format": "json"}
    q.update({"oldid": oldid} if oldid else {"page": title, "redirects": 1})
    url = f"https://{wiki}.wikisource.org/w/api.php?" + urllib.parse.urlencode(q)
    answer = json.loads(fetch(url))
    if "parse" not in answer:
        raise SystemExit(f"{wiki} {title}: {answer.get('error', {}).get('info', answer)}")
    parse = answer["parse"]
    return parse["revid"], html_text(parse["text"]["*"])


def clean(text: str) -> str:
    """Drops the Wikisource navigation above the article (up to the line 'See also ... disclaimer.' or the like)
    and the license footers; what is left is the running text."""
    for marker in ("disclaimer.", "Wikipedia article, quotes, travel guide, Wikidata item"):
        head, sep, rest = text.partition(marker)
        if sep and len(head) < 600:
            text = rest.strip()
    return re.split(r"\n(?:Retrieved from|Abgerufen von|Dieses Werk ist gemeinfrei)", text)[0].strip()


def documents(pin: bool) -> list:
    """[(label, text, oldid, sha)] in corpus order: English and German alternate."""
    docs = []
    for en, de in zip(EN, DE, strict=True):
        for wiki, title, oldid, sha in (en, de):
            revid, raw = wikisource(wiki, title, oldid)
            text = clean(raw)
            got = hashlib.sha256(text.encode()).hexdigest()
            if not pin and sha != got:
                raise SystemExit(f"{title}: sha256 {got} differs from the pinned {sha or '(none)'}; run with --pin")
            docs.append((f"{wiki}.wikisource.org {title}", text, revid, got))
    return docs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="work/corpus/prose.txt")
    ap.add_argument("--pin", action="store_true", help="print revision ids and hashes instead of checking them")
    args = ap.parse_args()
    docs = documents(args.pin)
    if args.pin:
        for label, text, oldid, sha in docs:
            print(f'("{label}", {oldid}, "{sha}"),  # {len(text)} chars')
    body = "".join(f"===== {label} =====\n{text}\n\n" for label, text, _, _ in docs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    sha = hashlib.sha256(body.encode()).hexdigest()
    print(f"{out}: {len(body)} chars, {len(docs)} documents, sha256 {sha}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
