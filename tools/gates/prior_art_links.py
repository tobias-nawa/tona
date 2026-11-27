"""Links of the "Prior art" section: do they have a valid form, and (online, in CI only) do they exist.

Used by prior_art.py. A coding agent invents citations from memory: an arXiv number that was never assigned, a pull
request number of a repository that has fewer. The form check runs everywhere, offline, and catches the clumsy ones
(month 13, a four-digit number after 2015, a non-numeric PR number, a commit that is not hex). The online check asks
arxiv.org and github.com whether the page exists. It never blocks on the network: a timeout, a refusal, a rate limit
or a server error is a warning, and only a clear "not found" (404, 410) is a finding. No credentials are sent.

What it cannot check: that a link which exists says what the text claims, and links on other hosts.
"""

import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import date

ARXIV_URL = re.compile(r"https?://(?:www\.)?arxiv\.org/(?:abs|pdf|html)/([^\s)>\]#?]+)", re.IGNORECASE)
ARXIV_BARE = re.compile(r"\barxiv\s*:\s*([^\s,;)\]]+)", re.IGNORECASE)
ARXIV_NEW = re.compile(r"^(\d{2})(\d{2})\.(\d{4,5})(?:v\d+)?(?:\.pdf)?$")
ARXIV_OLD = re.compile(r"^[a-z]+(?:-[a-z]+)*(?:\.[A-Za-z]{2})?/(\d{2})(\d{2})(\d{3})(?:v\d+)?(?:\.pdf)?$")
GITHUB_URL = re.compile(r"https?://(?:www\.)?github\.com/([^\s)>\]\"'`<]+)", re.IGNORECASE)
OWNER = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")
REPO = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
NEW_ID_SINCE = (2015, 1)  # from 2015-01 arXiv numbers have five digits; before, four
FIRST_ARXIV_YEAR = 1991
MAX_ONLINE = 40
TIMEOUT_S = 10
PAUSE_S = 0.3
MIN_COMMIT_HEX = 7
Fetch = Callable[[str], int | None]


def arxiv_problem(ident: str, today: date) -> str | None:
    """Why an arXiv identifier cannot exist, or None if its form is possible."""
    match = ARXIV_NEW.match(ident)
    if match:
        yy, mm, number = int(match.group(1)), int(match.group(2)), match.group(3)
        year = 2000 + yy
        if yy < 7 or not 1 <= mm <= 12:
            return f"arXiv:{ident} is not a possible identifier (year {year}, month {mm})"
        if (year, mm) > (today.year, today.month):
            return f"arXiv:{ident} is dated after today"
        if ((year, mm) >= NEW_ID_SINCE) != (len(number) == 5) or int(number) == 0:
            return f"arXiv:{ident} has the wrong number of digits for its month, or the number 0"
        return None
    match = ARXIV_OLD.match(ident)
    if match:
        year = 1900 + int(match.group(1)) if int(match.group(1)) >= 91 else 2000 + int(match.group(1))
        if not (FIRST_ARXIV_YEAR <= year <= 2007) or not 1 <= int(match.group(2)) <= 12:
            return f"arXiv:{ident} is not a possible identifier"
        return None
    return f"arXiv:{ident} is not an arXiv identifier (YYMM.NNNNN or archive/YYMMNNN)"


def github_problem(path: str) -> str | None:
    """Why a github.com link cannot be right, or None. `path` is what follows the host."""
    parts = [p for p in re.split(r"/+", path.split("#", maxsplit=1)[0].split("?", maxsplit=1)[0].rstrip(".,;")) if p]
    if len(parts) < 2:
        return None  # a profile or a landing page: nothing to validate
    owner, repo = parts[0], parts[1].removesuffix(".git")
    if not OWNER.match(owner) or owner.endswith("-") or "--" in owner or not REPO.match(repo):
        return f"github.com/{owner}/{repo} is not a possible repository name"
    rest = parts[2:]
    if len(rest) >= 2 and rest[0] in ("pull", "issues", "discussions") and not re.fullmatch(r"[1-9]\d*", rest[1]):
        return f"github.com/{owner}/{repo}/{rest[0]}/{rest[1]}: the number is not a positive integer"
    if len(rest) >= 2 and rest[0] == "commit" and not re.fullmatch(rf"[0-9a-fA-F]{{{MIN_COMMIT_HEX},40}}", rest[1]):
        return f"github.com/{owner}/{repo}/commit/{rest[1]}: not a hexadecimal commit hash of 7 to 40 digits"
    return None


def extract(text: str) -> tuple[list[str], list[str]]:
    """(arXiv identifiers, github.com paths) found in the text, in order, without duplicates."""
    arxiv = list(dict.fromkeys(m.rstrip(".,;") for m in ARXIV_URL.findall(text) + ARXIV_BARE.findall(text)))
    github = list(dict.fromkeys(GITHUB_URL.findall(text)))
    return arxiv, github


def form_problems(text: str, today: date | None = None) -> list[str]:
    today = today or date.today()
    arxiv, github = extract(text)
    found = [arxiv_problem(ident, today) for ident in arxiv] + [github_problem(path) for path in github]
    return [p for p in found if p]


def links(text: str) -> list[str]:
    arxiv, github = extract(text)
    pdf = re.compile(r"\.pdf$")
    urls = [f"https://arxiv.org/abs/{pdf.sub('', ident)}" for ident in arxiv]
    for path in github:
        urls.append("https://github.com/" + re.split(r"[#?]", path)[0].rstrip(".,;"))
    return urls


def fetch_status(url: str) -> int | None:
    """HTTP status of a GET without credentials, or None if the network did not answer."""
    request = urllib.request.Request(url, headers={"User-Agent": "tona-prior-art-check"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return response.status
    except urllib.error.HTTPError as err:
        return err.code
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def online_problems(text: str, fetch: Fetch = fetch_status, pause: float = PAUSE_S) -> tuple[list[str], list[str]]:
    """(findings, warnings): findings are links the host says do not exist; warnings are checks that did not finish."""
    findings, warnings = [], []
    for url in links(text)[:MAX_ONLINE]:
        status = fetch(url)
        if status in (404, 410):
            findings.append(f"{url} does not exist ({status})")
        elif status is None or status >= 400:
            warnings.append(f"{url}: could not be checked (status {status}); not a finding")
        time.sleep(pause)
    return findings, warnings
