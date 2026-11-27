"""Pinned source files: header and byte-range reads from a local file or over HTTP, download with SHA-256 check.

Range reads let the router match run before any of the 188 GB is downloaded (routers are ~5 MB per layer).
"""

import hashlib
import os
import subprocess
import tomllib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from coder_requant.gguf_layout import Layout, Truncated, parse

USER_AGENT = "tona-tools"  # generic client name, no personal data in requests
FIRST_READ = 4 << 20
# resumable (-C -), retries on dropped connections; HTTPS only, also across redirects
CURL = ["curl", "-fL", "--proto", "=https", "--proto-redir", "=https", "--retry", "10", "-C", "-"]


class HttpsOnly(urllib.request.HTTPRedirectHandler):
    """Refuses a redirect to any scheme other than https."""

    def redirect_request(self, req, *args):  # args: fp, code, msg, headers, newurl
        newurl = args[-1]
        if not newurl.lower().startswith("https://"):
            raise OSError(f"{req.full_url}: redirect to a non-HTTPS URL refused")
        return super().redirect_request(req, *args)


def open_https(req: urllib.request.Request, timeout: int):
    """urlopen for pinned URLs: the request itself and every redirect must be https."""
    if not req.full_url.lower().startswith("https://"):
        raise OSError(f"{req.full_url}: not an HTTPS URL")
    return urllib.request.build_opener(HttpsOnly).open(req, timeout=timeout)


@dataclass(frozen=True)
class Pinned:
    repo: str
    revision: str
    path: str
    size: int
    sha256: str

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.path}"

    @property
    def name(self) -> str:
        return Path(self.path).name


def load_pins(toml_path: Path, entry: str) -> list[Pinned]:
    """The files of one entry of a models.toml-style file (repo, revision, files = [{path, size, sha256}])."""
    m = tomllib.loads(toml_path.read_text())[entry]
    return [Pinned(m["repo"], m["revision"], f["path"], f["size"], f["sha256"]) for f in m["files"]]


class Reader:
    """Byte ranges of a local file, or of a pinned URL when the file is not there."""

    def __init__(self, local: Path | None, url: str | None = None):
        if local is None and url is None:
            raise ValueError("need a local file or a URL")
        self.local, self.url = local, url

    def read(self, offset: int, n: int, exact: bool = True) -> bytes:
        """Bytes offset..offset+n; with exact=False fewer at the end of the file."""
        if self.local is not None and self.local.exists():
            with open(self.local, "rb") as f:
                f.seek(offset)
                return f.read(n)
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={offset}-{offset + n - 1}", "User-Agent": USER_AGENT}
        )
        with open_https(req, 120) as r:
            data = r.read()
        if len(data) != n and (exact or len(data) > n):
            raise OSError(f"{self.url}: range read returned {len(data)} of {n} bytes")
        return data

    def layout(self) -> Layout:
        n = FIRST_READ
        while True:
            data = self.read(0, n, exact=False)
            try:
                return parse(data)
            except Truncated:
                if len(data) < n:  # the whole file is read and still ends inside the header
                    raise
                n *= 2

    def tensor(self, layout: Layout, name: str) -> bytes:
        t = layout.by_name[name]
        return self.read(layout.data_start + t.offset, t.nbytes)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(16 << 20):
            h.update(b)
    return h.hexdigest()


def fetch(pin: Pinned, dest_dir: Path) -> Path:
    """Downloads (resuming a .part file) and verifies size and SHA-256; an existing verified file is reused."""
    dest = dest_dir / pin.name
    if dest.exists() and dest.stat().st_size == pin.size and sha256_file(dest) == pin.sha256:
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    # the URL pins the revision; size and SHA-256 are checked below
    subprocess.run(
        [*CURL, "-A", USER_AGENT, "-o", str(part), pin.url],
        check=True,
    )
    if part.stat().st_size != pin.size:
        raise OSError(f"{pin.name}: {part.stat().st_size} bytes, pinned {pin.size}")
    if (got := sha256_file(part)) != pin.sha256:
        raise OSError(f"{pin.name}: SHA-256 {got}, pinned {pin.sha256}")
    os.replace(part, dest)
    return dest
