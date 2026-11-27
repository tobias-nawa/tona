#!/usr/bin/env python3 -I
"""Paired agentic comparison of two server variants (arms A and B), overnight-sized, through scripts/gpu_queue.py.

    run.py compare --out DIR --a 'NAME|MODEL|BUILD|KEY=VAL,...' --b '...' [--suites S,...] [--repeats N]
                   [--limit SUITE=N,...] [--offset polyglot=N] [--first-repeat K] [--base-seed N]
                   [--drafter GGUF] [--report FILE.md] [--dry-run]
    run.py block   --out DIR --arm 'NAME|MODEL|BUILD|ENV' --suites S,... --repeat K ...   (one queue job, internal)
    run.py report  --out DIR --a NAME --b NAME --suites S,...                            (tables only, to stdout)

An arm is `name|model.gguf|build|env`: the model path (relative to the repository root or absolute), the build
(directory under work/build or a path; the same build for both arms is the point of a model comparison) and optional
TONA_* variables of that arm. Both arms play the same tasks with the same seeds: seed = base-seed + 1000 * repeat, the
same for A and B. The jobs are ordered A r1, B r1, A r2, B r2, ..., so a repeat pair runs back to back and a slow drift
of the machine does not fall on one arm. Each job is one gpu_queue job of kind gpu (quality, no timing claim): server
start, all suites in turn, server stop, SWE grading (needs no server). Finished (arm, suite, repeat) files are kept, so
`compare` can be started again after an interruption and resumes. Result files hold task ids and pass/fail only (no
transcripts), see report.py for the statistics.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))
import ab_server  # noqa: E402
import report  # noqa: E402
import suites  # noqa: E402

PORT = 8097  # ab_server's default; 8662 is the user's own server and is never touched
DEFAULT_LIMITS = {"swe": 17, "polyglot": 100, "csharp": 60, "effort": 23, "json": 32}
DEFAULT_LIMITS.update({lang: 1000 for lang in suites.NEW_LANGS})  # all valid exercises


def parse_arm(text: str) -> ab_server.Arm:
    name, model, build, env = ([*text.split("|"), "", "", "", ""])[:4]
    kv = dict(x.split("=", 1) for x in env.split(",") if x)
    return ab_server.Arm(name, build, kv, model if Path(model).is_absolute() else str(ROOT / model) if model else "")


def parse_limits(text: str, base: dict | None = None) -> dict:
    out = dict(DEFAULT_LIMITS if base is None else base)
    out.update({k: int(v) for k, v in (x.split("=") for x in text.split(",") if x)})
    return out


def result_file(out: Path, arm: str, suite: str, repeat: int) -> Path:
    return out / f"{arm}-{suite}-r{repeat}.json"


def run_block(args) -> int:
    """One arm, one repeat: every suite whose result file is missing, on one server."""
    arm = parse_arm(args.arm)
    out, names, lim = Path(args.out), args.suites.split(","), parse_limits(args.limit)
    out.mkdir(parents=True, exist_ok=True)
    suites.OFFSET.update(parse_limits(args.offset, {}))
    todo = [s for s in names if not result_file(out, arm.name, s, args.repeat).exists()]
    if not todo:
        return 0
    seed = args.base_seed + 1000 * args.repeat
    ctx = max(suites.CONTEXT[s] for s in todo)
    cfg = ab_server.ServerConfig(ROOT, arm.model, args.drafter, ctx, PORT)
    tag = f"{arm.name}-r{args.repeat}"
    for tool in ("hw_report", "quiet_check"):
        with (out / f"{tag}.{tool}.txt").open("w") as f:
            subprocess.run([sys.executable, str(ROOT / f"tools/{tool}.py")], stdout=f, check=False)
    proc = ab_server.start_server(cfg, arm, out / f"{tag}.server.log")
    try:
        ab_server.wait_ready(proc, PORT, cfg.ready_timeout)
        url = f"http://127.0.0.1:{PORT}"
        for s in todo:
            sdir = out / f"{tag}-{s}"
            sdir.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            recs = suites.RUNNERS[s](url, sdir, seed, lim[s])
            (out / f"{tag}-{s}.seconds").write_text(f"{time.time() - t0:.0f}\n")
            if s != "swe":
                result_file(out, arm.name, s, args.repeat).write_text(json.dumps(recs))
    finally:
        ab_server.stop_server(proc)
        with (out / f"{tag}.quiet_check_after.txt").open("w") as f:
            subprocess.run([sys.executable, str(ROOT / "tools/quiet_check.py")], stdout=f, check=False)
    if "swe" in todo:
        ids = suites.swe_ids(lim["swe"])
        recs = suites.grade_swe(out / f"{tag}-swe", ids)
        result_file(out, arm.name, "swe", args.repeat).write_text(json.dumps(recs))
    time.sleep(3)
    return 0


def block_command(args, arm_text: str, repeat: int) -> list[str]:
    cmd = [sys.executable, str(HERE / "run.py"), "block", "--out", args.out, "--arm", arm_text, "--repeat", str(repeat)]
    cmd += ["--suites", args.suites, "--limit", args.limit, "--base-seed", str(args.base_seed)]
    cmd += ["--offset", args.offset]
    return [*cmd, "--drafter", args.drafter]


def run_compare(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    a, b = parse_arm(args.a), parse_arm(args.b)
    for k in range(args.first_repeat, args.first_repeat + args.repeats):
        for arm_text, arm in ((args.a, a), (args.b, b)):
            label = f"agent-{arm.name}-r{k}"
            cmd = [sys.executable, str(ROOT / "scripts/gpu_queue.py"), "--kind", "gpu", "--label", label, "--"]
            cmd += block_command(args, arm_text, k)
            if args.dry_run:
                print(" ".join(cmd))
                continue
            while subprocess.run(cmd, check=False).returncode == 75:  # desktop in use: the queue asks to try again
                time.sleep(60)
    if args.dry_run:
        return 0
    text = report.tables(out, a.name, b.name, args.suites.split(","))
    Path(args.report or out / "report.md").write_text(text + "\n")
    print(text)
    return 0


def run_report(args) -> int:
    print(report.tables(Path(args.out), args.a, args.b, args.suites.split(",")))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["compare", "block", "report"])
    p.add_argument("--out", required=True)
    p.add_argument("--a")
    p.add_argument("--b")
    p.add_argument("--arm")
    p.add_argument("--suites", default="polyglot,effort,json")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--first-repeat", type=int, default=1, help="number of the first repeat (night 2: 2)")
    p.add_argument("--offset", default="", help="SUITE=N: skip the first N tasks (polyglot only)")
    p.add_argument("--limit", default="")
    p.add_argument("--base-seed", type=int, default=165)
    p.add_argument("--drafter", default=str(ROOT / "work/models/draft/mtp-Qwen3.8-Flash-Next-Q8_0.gguf"))
    p.add_argument("--report")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    return {"compare": run_compare, "block": run_block, "report": run_report}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
