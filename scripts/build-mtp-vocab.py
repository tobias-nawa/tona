#!/usr/bin/python3 -I
"""Token ID list for the truncated LM head of the MTP drafts -> engine/model/mtp_vocab.inc.

Corpus: real source code and text on this machine per language (Python, Rust, JS/TS, C/C++, Go, C#, shell, Markdown,
web and data formats), up to 8 MB each (duplicate files skipped), tokenized with our own tokenizer (tona-tokenize, CPU
only). Ranking: sum of the relative frequencies over the languages (every language weighs the same). List = special
tokens (the last 400 IDs) + the best IDs up to N_LIST, ~41,500 lines. No third-party list is used.
Reference before this change: union with the code32 list of another project, code 5.27 tokens per pass, full head 5.41
(measurements/2026-10-26-mtp); the new list is measured in measurements/2026-11-13-mtp-vocab-own.

build-mtp-vocab.py [--reuse]   (--reuse: reuse the counts from work/tmp/vocab-build)
"""

import argparse
import collections
import hashlib
import os
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK = ROOT + "/work/build/engine/tona-tokenize"
MODEL = ROOT + "/work/models/draft/mtp-Qwen3.8-Flash-Next-Q8_0.gguf"
WORK = ROOT + "/work/tmp/vocab-build"
CAP = 8 << 20
VOCAB = 248320
N_DYN = 16000  # list of the dynamic head (mtp_vocab_dyn.inc): the scorer finds the rest
N_LIST = 41500  # same size as the list it replaces (41,489), so the MTP head costs the same
H = os.path.expanduser("~")
LANGS = {
    "py": (["/usr/lib/python3.14", H + "/Projects"], (".py",)),
    "rs": ([H + "/.cargo/registry/src", H + "/Projects"], (".rs",)),
    "js": (["/usr/lib/node_modules", H + "/Projects"], (".js", ".mjs", ".ts", ".tsx", ".jsx")),
    "cpp": ([ROOT + "/work/src/llama.cpp", H + "/Projects"], (".c", ".cpp", ".h", ".hpp", ".cc", ".hip", ".cu")),
    "go": ([H + "/Projects", H + "/go", "/usr/lib/golang"], (".go",)),
    "cs": ([H + "/Projects"], (".cs",)),
    "sh": (["/usr/bin", "/usr/share", H + "/Projects", "/etc"], (".sh", ".bash")),
    "md": (["/usr/share/doc", H + "/Projects", "/usr/lib/node_modules"], (".md",)),
    "web": (
        ["/usr/share", "/usr/lib/node_modules", H + "/Projects"],
        (".html", ".css", ".json", ".yml", ".yaml", ".toml", ".sql"),
    ),
}
SKIP = re.compile(r"/(\.git|work/models|work/build|target|dist|build|vendor|\.venv|venv)/")


def _corpus_files(roots, exts) -> list[str]:
    files = []
    for r in roots:
        if not os.path.isdir(r):
            continue
        for dp, dn, fn in os.walk(r):
            if SKIP.search(dp + "/"):
                dn[:] = []
                continue
            dn[:] = [d for d in dn if not d.startswith(".") or d == ".cargo"]
            files += [os.path.join(dp, f) for f in fn if f.endswith(exts)]
    random.seed(1)
    random.shuffle(files)
    return files


def _read_small_text(path) -> str | None:
    """The text of a file up to 512 KiB, or None for a larger, unreadable or non-UTF-8 file."""
    try:
        if os.path.getsize(path) > 512 << 10:
            return None
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _read_corpus(files) -> tuple[list[str], int]:
    buf, total, seen = [], 0, set()
    for f in files:
        t = _read_small_text(f)
        if t is None:
            continue
        digest = hashlib.sha256(t.encode()).digest()
        if digest in seen:  # worktrees and vendored copies hold the same file many times
            continue
        seen.add(digest)
        buf.append(t)
        total += len(t.encode())
        if total >= CAP:
            break
    return buf, total


def count(lang, roots, exts):
    buf, total = _read_corpus(_corpus_files(roots, exts))
    path = f"{WORK}/corpus-{lang}.txt"
    Path(path).write_text("\n\n".join(buf))
    out = subprocess.run([TOK, MODEL, path, "--no-special"], capture_output=True, text=True, check=True).stdout
    ids = [int(x) for x in out.split()]
    c = collections.Counter(ids)
    with open(f"{WORK}/counts-{lang}.txt", "w") as o:
        for k, v in c.most_common():
            o.write(f"{k} {v}\n")
    print(lang, total >> 20, "MB", len(ids), "tokens", file=sys.stderr)


def rank_languages() -> collections.Counter:
    """Sum of relative token frequencies over the languages, so every language weighs the same."""
    score = collections.Counter()
    for lang in LANGS:
        lines = Path(f"{WORK}/counts-{lang}.txt").read_text().splitlines()
        c = {int(a): int(b) for a, b in (x.split() for x in lines)}
        tot = sum(c.values())
        for k, v in c.items():
            score[k] += v / tot
    return score


def main():
    ap = argparse.ArgumentParser(description="Build engine/model/mtp_vocab.inc (see the module docstring).")
    ap.add_argument("--reuse", action="store_true", help="reuse the counts from work/tmp/vocab-build")
    args = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    for lang, (roots, exts) in LANGS.items():
        if not args.reuse or not os.path.exists(f"{WORK}/counts-{lang}.txt"):
            count(lang, roots, exts)
    ranked = [k for k, _ in rank_languages().most_common()]
    for name, n in (("mtp_vocab.inc", N_LIST), ("mtp_vocab_dyn.inc", N_DYN)):
        ids = set(range(VOCAB - 400, VOCAB))
        for k in ranked:
            if len(ids) >= n:
                break
            ids.add(k)
        ids = sorted(ids)
        with open(f"{ROOT}/engine/model/{name}", "w") as o:
            o.write(f"// generated by scripts/build-mtp-vocab.py: {len(ids)} token IDs, ascending\n")
            for i in range(0, len(ids), 16):
                o.write(",".join(map(str, ids[i : i + 16])) + ",\n")
        print(name, len(ids), "IDs", file=sys.stderr)


if __name__ == "__main__":
    main()
