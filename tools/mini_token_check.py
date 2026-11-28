#!/usr/bin/env python3
"""CPU token check on the mini models: builds them from the committed headers, runs the CPU reference (tona-ref) and
compares token ids and the logits of every step with data/mini-models/expected/cpu.json.

Usage: tools/mini_token_check.py compare [--build DIR] [--models DIR]   exit 1 on any difference
       tools/mini_token_check.py record  [--build DIR] [--models DIR]   rewrite the expected file (reviewed in a PR)
DIR of --build holds tona-ref (default work/build/cpu). Standard library only.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import make_mini_model

ROOT = make_mini_model.ROOT
EXPECTED = os.path.join(make_mini_model.DATA, "expected", "cpu.json")
# Same prompts as scripts/golden.sh; 128 tokens is the longest run that keeps the whole check under a minute
PROMPTS = (
    "Write a Python function that parses a CSV file and returns the average of each numeric column. Only code.",
    "Write a Rust function that checks whether a string is a valid IPv4 address, with tests. Only code.",
)
N_TOKENS = 128
# model id (name of its header) -> folder under the models directory and file name of the mini model
MODELS = {
    "coder": ("mini", "mini.gguf"),
    "ornith": ("mini-ornith", "mini-ornith.gguf"),
    "bolt": ("mini-bolt", "mini-bolt.gguf"),
}
SEED = 1
TOP_LINE = re.compile(r"^  top: (\d+) ", re.MULTILINE)
HASH_LINE = re.compile(r"^ref-hash (\d+) ([0-9a-f]{16})$", re.MULTILINE)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


def build_model(model_id: str, models: str) -> str:
    """Builds the mini model of one header into the models directory and returns the file."""
    folder, name = MODELS[model_id]
    header = os.path.join(make_mini_model.DATA, model_id + ".header.gguf.xz")
    make_mini_model.build_from_header(header, os.path.join(models, folder), SEED, False)
    return os.path.join(models, folder, name)


def run_reference(tona_ref: str, model: str, prompt: str) -> dict:
    """One greedy run of the CPU reference: the chosen token ids and a digest of the logits of every step."""
    env = dict(os.environ, TONA_DETERMINISTIC="1")
    r = subprocess.run(
        [tona_ref, model, prompt, "-n", str(N_TOKENS), "--top", "1", "--hash"],
        capture_output=True, env=env, check=False,
    )  # fmt: skip
    # token pieces can be cut UTF-8; only the ids and hashes are read
    err = r.stderr.decode(errors="replace")
    if r.returncode:
        raise SystemExit(f"tona-ref failed ({r.returncode}) on {model}:\n{err[-2000:]}")
    steps = HASH_LINE.findall(err)
    digest = hashlib.sha256("".join(f"{i} {h}\n" for i, h in steps).encode()).hexdigest()
    return {"prompt": prompt, "n": N_TOKENS, "tokens": [int(t) for t in TOP_LINE.findall(err)],
            "steps": len(steps), "logits_sha256": digest}  # fmt: skip


def measure(build: str, models: str) -> dict:
    tona_ref = os.path.join(build, "tona-ref")
    out = {}
    for model_id in MODELS:
        model = build_model(model_id, models)
        out[model_id] = {
            "header": f"{model_id}.header.gguf.xz",
            "seed": SEED,
            "model_sha256": sha256_file(model),
            "runs": [run_reference(tona_ref, model, p) for p in PROMPTS],
        }
    return {"env": {"TONA_DETERMINISTIC": "1"}, "models": out}


def differences(want: dict, got: dict, *, tokens_only: bool = False) -> list[str]:
    found = []
    for model_id, w in want["models"].items():
        g = got["models"][model_id]
        if w["model_sha256"] != g["model_sha256"]:
            found.append(f"{model_id}: the mini model differs (generator or header changed; record again)")
        for i, (wr, gr) in enumerate(zip(w["runs"], g["runs"], strict=True)):
            if wr["tokens"] != gr["tokens"]:
                pairs = enumerate(zip(wr["tokens"], gr["tokens"], strict=False))
                first = next((k for k, (a, b) in pairs if a != b), None)
                found.append(f"{model_id} run {i}: tokens differ (first at {first}, {len(gr['tokens'])} of {wr['n']})")
            elif not tokens_only and wr["logits_sha256"] != gr["logits_sha256"]:
                found.append(f"{model_id} run {i}: same tokens, different logits")
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("compare", "record"))
    ap.add_argument("--build", default=os.path.join(ROOT, "work", "build", "cpu"))
    ap.add_argument("--models", default=make_mini_model.MODELS)
    ap.add_argument(
        "--tokens-only", action="store_true", help="require equal tokens; ignore compiler-specific logit rounding"
    )
    a = ap.parse_args()
    got = measure(a.build, a.models)
    if a.mode == "record":
        os.makedirs(os.path.dirname(EXPECTED), exist_ok=True)
        with open(EXPECTED, "w") as f:
            json.dump(got, f, indent=1)
            f.write("\n")
        print(f"recorded {EXPECTED}")
        return 0
    with open(EXPECTED) as f:
        found = differences(json.load(f), got, tokens_only=a.tokens_only)
    for line in found:
        print("DIFFERS:", line)
    print("mini token check:", "FAILED" if found else "identical")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
