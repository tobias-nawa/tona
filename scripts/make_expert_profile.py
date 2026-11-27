#!/usr/bin/env python3
"""Count which experts a model uses on a mix of code and prose and write its expert profile.

The profile (engine/backend/expert_profile.h) seeds the VRAM expert cache and ranks the pinned experts at start, so a
model with one reaches its steady state sooner (research/measurements/2026-11-19-expert-profiles.md). It counts the
experts of one GGUF and is named after the model id, the folder name under work/models/: toolchain/profiles/<id>.bin
ships with the installer and scripts/ab.py picks it by that id. Coder profiles come from scripts/make-profile.sh
(chat prompts, its own tool); this script drives `tona-bolt-bench`, which runs any model of the generic backend
interface (prefill of a prompt, then greedy decode), once for the code slice and once for the prose slice.

No train-on-test: both slices start at MIN_OFFSET characters, after every prompt of the workloads in
scripts/ab_workloads.py (`prose` and `longdoc` end below 450,000 characters of the prose corpus, `write` starts at
1,000,000 of the code corpus and its longest prompt is 16000 tokens; `huge`, a scale-run workload, cuts from the whole
code corpus and is not covered).

Run it as a `gpu` job of scripts/gpu_queue.py:
  python3 scripts/gpu_queue.py --kind gpu --label profile-<id> -- python3 scripts/make_expert_profile.py \
      --model work/models/<id>/<file>.gguf --build work/build/<dir>
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIN_OFFSET = 1_500_000  # characters into a corpus: past every prompt the A/B workloads cut (see the docstring)
CHARS_PER_TOKEN = 8  # generous: the slice is cut to --tokens after tokenizing
DECODE_MARGIN = 64


def run(cmd: list, env: dict | None = None, stdout=None) -> None:
    subprocess.run(cmd, check=True, env=env, stdout=stdout)


def slice_ids(build: Path, model: Path, corpus: Path, offset: int, tokens: int, work: Path) -> Path:
    """The first `tokens` token ids of the slice of `corpus` that starts at `offset`, one id per line."""
    text = corpus.read_text(encoding="utf-8")[offset : offset + tokens * CHARS_PER_TOKEN]
    if len(text) < tokens * 2:
        raise SystemExit(f"{corpus} has too little text after character {offset}")
    src = work / f"{corpus.stem}.txt"
    src.write_text(text, encoding="utf-8")
    ids = work / f"{corpus.stem}.ids"
    with ids.open("w") as out:
        run([str(build / "tona-tokenize"), str(model), str(src), "--no-special"], stdout=out)
    lines = ids.read_text().split()
    if len(lines) < tokens:
        raise SystemExit(f"{corpus}: {len(lines)} tokens after character {offset}, need {tokens}")
    ids.write_text("\n".join(lines[:tokens]) + "\n")
    return ids


def count(build: Path, model: Path, ids: Path, tokens: int, gen: int, profile: Path) -> None:
    """One prefill of the slice and `gen` decode tokens; the counts are added to `profile` at exit."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("TONA_")}
    env["ROCR_VISIBLE_DEVICES"] = "0"
    env["TONA_EXPERT_PROFILE_OUT"] = str(profile)
    if profile.exists():
        env["TONA_EXPERT_PROFILE"] = str(profile)  # the counts accumulate over the runs
    cmd = [str(build / "tona-bolt-bench"), str(model), str(ids), "--prompt", str(tokens), "--gen", str(gen)]
    cmd += ["--ctx", str(tokens + gen + DECODE_MARGIN)]
    if sys.platform != "darwin":  # a leak ends the scope, not the desktop (docs/contribute/shared-machine.md)
        cmd = ["systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=54G", "-p", "MemorySwapMax=0", *cmd]
    run(cmd, env)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="GGUF, in work/models/<model id>/")
    ap.add_argument("--build", required=True, help="build directory with tona-tokenize and tona-bolt-bench")
    ap.add_argument("--out", default=None, help="default toolchain/profiles/<model id>.bin")
    ap.add_argument("--code-corpus", default="work/corpus/code.txt")
    ap.add_argument("--prose-corpus", default="work/corpus/prose.txt")
    ap.add_argument(
        "--offset", type=int, default=MIN_OFFSET, help=f"characters into each corpus (at least {MIN_OFFSET})"
    )
    ap.add_argument("--tokens", type=int, default=20000, help="prompt tokens per text kind (prefill counts them)")
    ap.add_argument("--gen", type=int, default=300, help="decode tokens per text kind")
    args = ap.parse_args()
    if args.offset < MIN_OFFSET:
        raise SystemExit(f"--offset {args.offset} is inside the prompts of the A/B workloads (minimum {MIN_OFFSET})")
    model, build = (ROOT / args.model).resolve(), (ROOT / args.build).resolve()
    out = Path(args.out) if args.out else ROOT / "toolchain/profiles" / f"{model.parent.name}.bin"
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR") or "/var/tmp") as tmp:
        work = Path(tmp)
        profile = work / "profile.bin"
        for corpus in (args.code_corpus, args.prose_corpus):
            ids = slice_ids(build, model, ROOT / corpus, args.offset, args.tokens, work)
            count(build, model, ids, args.tokens, args.gen, profile)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(profile, out)  # only a complete profile replaces the old one
    print(f"{out}: {out.stat().st_size} bytes, {args.tokens} prompt + {args.gen} decode tokens of code and of prose")
    return 0


if __name__ == "__main__":
    sys.exit(main())
