#!/usr/bin/env python3
"""Fetches a pinned model from toolchain/models.toml into work/models/<name>/ and verifies the SHA-256.

Existing files (e.g. from another engine's model directory) are linked after verification instead of downloaded:
  tools/get_model.py coder-iq1_m --from /path/to/models/coder-IQ1_M
Otherwise a download from Hugging Face with resume (.part). Standard library only.
  tools/get_model.py --list   catalog as JSON (for tona setup): title, kind, size and presence per entry
  tools/get_model.py mini | mini-ornith | mini-bolt   builds a mini model from the committed header (no download)
"""

import argparse
import hashlib
import json
import os
import sys
import tomllib
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHUNK = 16 << 20
MINI_HEADERS = {"mini": "coder", "mini-ornith": "ornith", "mini-bolt": "bolt"}  # name -> header in data/mini-models/


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(CHUNK):
            h.update(b)
    return h.hexdigest()


def disk_name(want: dict) -> str:
    """Name on disk: save_as renames a file that comes from another repository (the split loader needs the prefix of
    part 1, engine/model/gguf.cpp)."""
    return want.get("save_as") or Path(want["path"]).name


def pinned(m: dict) -> bool:
    """Every file has a repository and a full commit hash; an entry written before its upload carries UNPUBLISHED."""
    for want in m["files"]:
        repo, rev = want.get("repo", m.get("repo", "")), want.get("revision", m.get("revision", ""))
        if "/" not in repo or len(rev) != 40 or any(c not in "0123456789abcdef" for c in rev):
            return False
    return True


def verified(path: Path, want: dict, mark: Path) -> bool:
    """Verification mark next to the name on disk (never in the source directory): size, mtime and hash. Saves
    re-hashing ~30 GB."""
    st = path.stat()
    if mark.exists():
        m = json.loads(mark.read_text())
        if m == {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": want["sha256"]}:
            return True
    if st.st_size != want["size"]:
        return False
    print(f"  verifying SHA-256 of {path.name} ...", file=sys.stderr)
    if sha256_file(path) != want["sha256"]:
        return False
    mark.write_text(json.dumps({"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": want["sha256"]}))
    return True


def download(url: str, dest: Path, want: dict) -> None:
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    token = os.environ.get("HF_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab") as f:
        if have and r.status != 206:
            f.truncate(0)
            have = 0
        n = have
        while b := r.read(CHUNK):
            f.write(b)
            n += len(b)
            print(f"\r  {dest.name}: {n / 1e9:.1f} / {want['size'] / 1e9:.1f} GB", end="", file=sys.stderr)
    print(file=sys.stderr)
    if part.stat().st_size != want["size"]:
        sys.exit(f"{dest.name}: size {part.stat().st_size} instead of {want['size']}")
    part.rename(dest)
    if not verified(dest, want, dest.with_name(dest.name + ".sha256ok")):
        sys.exit(f"{dest.name}: SHA-256 mismatch")


def present(out: Path, m: dict) -> bool:
    """All files present with a valid verification mark (without re-hashing)."""
    for want in m["files"]:
        dest = out / disk_name(want)
        mark = out / (dest.name + ".sha256ok")
        if not dest.exists() or not mark.exists():
            return False
        st = dest.resolve().stat()
        if json.loads(mark.read_text()) != {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": want["sha256"]}:
            return False
    return True


def catalog(models: dict, dest: Path) -> list:
    return [
        {
            "id": k,
            "title": m["title"],
            "kind": m.get("kind", "model"),
            "tested": m.get("tested", False),
            "license": m.get("license", ""),
            "quality": m.get("quality", ""),
            "first_file": disk_name(m["files"][0]),
            "size_gb": sum(f["size"] for f in m["files"]) / 1e9,
            "present": present(dest / k, m),
        }
        for k, m in models.items()
    ]


def get_mini(name: str, dest: Path) -> int:
    """Random weights around the real header: building a mini model needs no big model (docs/contribute/testing.md)."""
    sys.path.insert(0, str(REPO / "tools"))
    import make_mini_model

    header = REPO / "data" / "mini-models" / f"{MINI_HEADERS[name]}.header.gguf.xz"
    make_mini_model.build_from_header(str(header), str(dest / name), 1, False)
    print(dest / name / (name + ".gguf"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?")
    ap.add_argument("--list", action="store_true", help="print the catalog as JSON")
    ap.add_argument("--from", dest="src", help="directory with existing files (they are linked)")
    ap.add_argument("--dest", default=str(REPO / "work" / "models"))
    a = ap.parse_args()
    models = tomllib.loads((REPO / "toolchain" / "models.toml").read_text())
    if a.list:
        print(json.dumps(catalog(models, Path(a.dest))))
        return 0
    if a.name in MINI_HEADERS:
        return get_mini(a.name, Path(a.dest))
    if a.name not in models:
        sys.exit(f"unknown: {a.name}; available: {', '.join(models)}")
    m = models[a.name]
    if not pinned(m):
        sys.exit(f"{a.name}: not published yet (no pinned repository and revision in toolchain/models.toml)")
    out = Path(a.dest) / a.name
    out.mkdir(parents=True, exist_ok=True)
    for want in m["files"]:
        fname = disk_name(want)
        dest = out / fname
        mark = out / (fname + ".sha256ok")
        if dest.exists() and verified(dest.resolve(), want, mark):
            print(f"ok      {fname}", file=sys.stderr)
            continue
        # an existing download may carry the name on disk or the name in its repository
        found = [p for n in (fname, Path(want["path"]).name) if a.src and (p := Path(a.src).expanduser() / n).exists()]
        if found:
            src = found[0].resolve()
            if not verified(src, want, mark):
                sys.exit(f"{src}: checksum does not match the pinned revision")
            if dest.is_symlink() or dest.exists():
                dest.unlink()
            dest.symlink_to(src)
            print(f"linked  {fname} -> {src}", file=sys.stderr)
            continue
        repo, rev = want.get("repo", m["repo"]), want.get("revision", m["revision"])
        url = f"https://huggingface.co/{repo}/resolve/{rev}/{want['path']}"
        print(f"fetching {fname} ({want['size'] / 1e9:.1f} GB)", file=sys.stderr)
        download(url, dest, want)
    print(out / disk_name(m["files"][0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
