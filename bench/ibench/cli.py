"""Command line: run, compare, proxy, env."""

import argparse
import contextlib
import json
import sys
import tomllib
from datetime import date
from pathlib import Path

from . import envinfo, report, runner, suite
from .sysmon import SysMon

BENCH = Path(__file__).resolve().parent.parent
REPO = BENCH.parent


def load_target(args) -> dict:
    if args.url:
        t = {"name": args.target or "adhoc", "url": args.url, "api": args.api, "model": args.model or "default"}
    else:
        targets = tomllib.loads((BENCH / "targets.toml").read_text())
        local = REPO / "work" / "bench" / "targets.local.toml"
        if local.exists():
            targets.update(tomllib.loads(local.read_text()))
        if args.target not in targets:
            sys.exit(f"Target {args.target!r} is not in bench/targets.toml or {local}; available: {', '.join(targets)}")
        t = dict(targets[args.target], name=args.target)
        if args.model:
            t["model"] = args.model
    t.setdefault("api", "openai")
    t.setdefault("extra", {})
    return t


def _select_cases(args) -> list[suite.Case]:
    meta, cases = suite.load(args.suite)
    if args.cases:
        wanted = set(args.cases.split(","))
        missing = wanted - {c.name for c in cases}
        if missing:
            sys.exit(f"Unknown cases: {', '.join(sorted(missing))}")
        cases = [c for c in cases if c.name in wanted]
    for c in cases:
        if args.runs is not None:
            c.runs = args.runs
        if args.warmup is not None:
            # a warm case needs at least one warm-up run to fill the cache
            c.warmup = max(args.warmup, 1 if c.cache == "warm" else 0)
    return meta, cases


def _measure(cases, target, interval: float) -> list[dict]:
    mon = SysMon(interval).start()
    results = []
    try:
        for c in cases:
            results.append(runner.run_case(c, target, mon))
    except KeyboardInterrupt:
        print("Interrupted, writing partial result", file=sys.stderr)
    finally:
        mon.stop()
    return results


def _suite_file(path: str) -> str:
    p = Path(path).resolve()
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else path


def cmd_run(args) -> int:
    target = load_target(args)
    meta, cases = _select_cases(args)
    env = envinfo.collect(target)
    for w in env["warnings"]:
        print(f"Warning: {w}", file=sys.stderr)
    if not env["server"].get("/v1/models") and not args.force:
        sys.exit("The server does not respond. Use --force to measure anyway.")

    results = _measure(cases, target, args.interval)
    data = {
        "tool": "ibench",
        "suite": dict(meta, file=_suite_file(args.suite)),
        "env": env,
        "cases": results,
    }
    stem = f"{date.today().isoformat()}-ibench-{target['name']}-{meta['name']}"
    if args.label:
        stem += f"-{args.label}"
    base = report.unique_path(Path(args.out_dir), stem)
    jp, mp = report.write(data, base)
    print(f"Report: {mp}\nRaw data: {jp}", file=sys.stderr)
    return 0


def cmd_compare(args) -> int:
    a, b = (json.loads(Path(p).read_text()) for p in (args.a, args.b))
    text = report.compare(a, b)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


def cmd_proxy(args) -> int:
    from . import proxy

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    srv = proxy.serve(args.listen, args.upstream, args.out)
    print(f"Proxy {args.listen} -> {args.upstream}, trace: {args.out}", file=sys.stderr)
    with contextlib.suppress(KeyboardInterrupt):
        srv.serve_forever()
    return 0


def cmd_env(args) -> int:
    print(json.dumps(envinfo.collect(load_target(args)), indent=1, ensure_ascii=False))
    return 0


def _target_args(p):
    p.add_argument(
        "--target", "-t", default="tona", help="name from bench/targets.toml or work/bench/targets.local.toml"
    )
    p.add_argument("--url", help="instead of --target: base URL without /v1")
    p.add_argument("--api", default="openai", choices=["openai", "anthropic"])
    p.add_argument("--model")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="bench/run", description="Measurement harness for the inference engine")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="measure a suite against a target")
    _target_args(p)
    p.add_argument("--suite", "-s", default=str(BENCH / "suites" / "base.toml"))
    p.add_argument("--cases", "-c", help="only these cases, comma-separated")
    p.add_argument("--runs", "-n", type=int, help="override runs per case")
    p.add_argument("--warmup", type=int, help="override warm-up runs per case")
    p.add_argument("--label", "-l", help="suffix for the file name")
    p.add_argument("--out-dir", default=str(REPO / "research" / "measurements"))
    p.add_argument("--interval", type=float, default=0.1, help="system sampling interval in s")
    p.add_argument("--force", action="store_true", help="measure even if /v1/models does not respond")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("compare", help="compare two JSON reports")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--out", "-o")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("proxy", help="record a client's requests as a trace")
    p.add_argument("--listen", default="127.0.0.1:8099")
    p.add_argument("--upstream", default="http://127.0.0.1:8662")
    p.add_argument("--out", "-o", required=True, help="trace file (JSONL), e.g. bench/traces/name.jsonl")
    p.set_defaults(fn=cmd_proxy)

    p = sub.add_parser("env", help="print the environment")
    _target_args(p)
    p.set_defaults(fn=cmd_env)

    args = ap.parse_args(argv)
    return args.fn(args)
