#!/usr/bin/env python3
"""A/B speed measurement of two server variants that stops as soon as the result is statistically clear.

    source toolchain/env.sh
    scripts/ab.py --a merged --b w88 --b-env TONA_DIAG=1 --workload short,long,set

Per round every arm gets a fresh tona-server (systemd scope, memory limit); the order alternates (AB, BA, AB, ...) so
that drift (warm-up, desktop load) cancels in the pairs. After each round the paired differences B against A in
percent are tested per metric with a group-sequential rule with a fixed error budget (scripts/ab_stats.py): "B
faster", "B slower", "equal" (within +-margin) or "open". The run ends when every metric is decided, at the latest
after --max-rounds. No round is dropped, with one exception: a round during which a system backup ran is repeated
(each start waits for a running backup to end first); the report lists it as discarded. See docs/contribute/testing.md.

The whole measurement holds the GPU lock (TONA_GPU_LOCK, default work/gpu.lock). Under an outer `flock`
pass --no-lock (or TONA_GPU_LOCK_HELD=1, which scripts/gpu-lock.sh sets), locks must not nest. Workloads:
  short, ctx, mid, long, set, huge
             the named workloads of docs/contribute/measurement.md (ab_workloads.py): prompts from the fixed corpus
             (scripts/make_corpus.py) and scripts/bench-set.py; Prefill tok/s, Decode tok/s, SHA of the text
  server     own sizes: --targets TOKENS[:DECODE],... and --ctx N[,N...]
  bench-set  scripts/bench-set.py at --ctx
"""

import argparse
import hashlib
import os
import signal
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

import ab_load
import ab_noise
import ab_plan
import ab_report
import ab_server
import ab_workloads
import make_corpus
from ab_stats import Plan
from gpu_queue import GpuLock

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_LOCK = str(ROOT / "work" / "gpu.lock")
MODEL = "work/models/coder-iq1_m/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf"
DRAFTER = "work/models/draft/mtp-Qwen3.8-Flash-Next-Q8_0.gguf"


def parse_args(argv: list) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", required=True, help="build of arm A: name under work/build/ or a directory")
    ap.add_argument("--b", required=True, help="build of arm B")
    ap.add_argument("--env", action="append", default=[], metavar="K=V", help="environment of both arms")
    ap.add_argument("--a-env", action="append", default=[], metavar="K=V")
    ap.add_argument("--b-env", action="append", default=[], metavar="K=V")
    ap.add_argument(
        "--workload",
        default="",
        help="short, ctx, mid, long, prose, longdoc, set, huge (docs/contribute/measurement.md), or server",
    )
    ap.add_argument(
        "--targets",
        default="30000,130000",
        help="prompt tokens[:decode tokens] of the server workload, e.g. 2000:400,130000",
    )
    ap.add_argument(
        "--decode-tokens",
        type=int,
        default=200,
        help="tokens decoded per prompt unless a target says otherwise (TARGET:TOKENS)",
    )
    ap.add_argument(
        "--why-size", default="", help="required: why this workload size (targets, tokens) answers the question"
    )
    ap.add_argument("--noise", default="bench/noise.toml", help="known noise of the workloads")
    ap.add_argument("--no-noise", action="store_true", help="ignore the noise table (at least --min-rounds rounds)")
    ap.add_argument("--ctx", default="140000", help="server context; several (140000,262144) = one start each")
    ap.add_argument("--port", type=int, default=ab_server.DEFAULT_PORT)
    ap.add_argument("--alpha", type=float, default=Plan.alpha, help="total error budget per metric")
    ap.add_argument("--margin", type=float, default=Plan.margin, help="percent below which a difference is 'equal'")
    ap.add_argument("--max-rounds", type=int, default=Plan.max_rounds)
    ap.add_argument("--min-rounds", type=int, default=Plan.min_rounds)
    ap.add_argument("--prior-sd", type=float, default=Plan.prior_sd, help="percent: noise of one paired difference")
    ap.add_argument("--prior-df", type=float, default=Plan.prior_df)
    ap.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="METRIC:LO..HI%",
        help=("required, repeatable: expected B against A in percent for a metric or group, e.g. decode:-0.5..+0.5%%"),
    )
    ap.add_argument(
        "--primary",
        default="",
        help="metrics (selectors as in --expect, comma separated) that decide when to stop; default all",
    )
    ap.add_argument("--fixed-rounds", action="store_true", help="run all --max-rounds, do not stop early (noise runs)")
    ap.add_argument("--no-warmup", action="store_true", help="skip the discarded first server start of A")
    ap.add_argument(
        "--round-warmup",
        choices=("auto", "on", "off"),
        default="auto",
        help="a discarded start of each arm before its measured one, every round (auto: on under macOS, where the "
        "other arm's model in the page cache gave an order effect of ~5 %%, 2026-11-12-mac-iq3xxs-repack.md)",
    )
    ap.add_argument(
        "--load-log",
        choices=("auto", "on", "off"),
        default="auto",
        help="load beside each start (ours, system, other; scripts/ab_load.py); auto: on under macOS, Linux: on only",
    )
    ap.add_argument("--expect-source", default="", help="required: where the expectation comes from")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--a-model", default="", help="GGUF of arm A when the arms differ in the model (default --model)")
    ap.add_argument("--b-model", default="", help="GGUF of arm B (default --model)")
    ap.add_argument("--a-drafter", default="", help="MTP head of arm A when the arms differ in it (default --drafter)")
    ap.add_argument("--b-drafter", default="", help="MTP head of arm B (default --drafter)")
    ap.add_argument("--drafter", default=DRAFTER, help="MTP head file; none: no --mtp (Ornith: head in the model file)")
    ap.add_argument("--no-expert-profile", action="store_true", help="no shipped expert profile in both arms")
    ap.add_argument("--memory-max", default=None, help="server memory limit (default: RAM minus 6 GiB)")
    ap.add_argument("--corpus", default="work/corpus/code.txt", help="made by scripts/make_corpus.py if missing")
    ap.add_argument("--prose-corpus", default="work/corpus/prose.txt", help="made by scripts/make_prose_corpus.py")
    ap.add_argument("--out", default=None, help="base name of the .json and .md reports (default work/ab/ab-<time>)")
    ap.add_argument("--no-lock", action="store_true", help="the caller already holds the GPU lock")
    return ap.parse_args(argv)


def parse_env(items: list) -> dict:
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"environment entry {item!r} is not K=V")
        out[key] = value
    return out


def default_memory_max() -> str:
    """As the justfile: RAM minus 6 GiB for the rest of the system."""
    if sys.platform == "darwin":  # no /proc on macOS (docs/use/macos.md)
        total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True).stdout)
        return f"{total // 1073741824 - 6}G"
    with open("/proc/meminfo") as f:
        kib = int(next(line.split()[1] for line in f if line.startswith("MemTotal:")))
    return f"{kib // 1048576 - 6}G"


def plan_from(args: argparse.Namespace) -> Plan:
    plan = Plan(args.alpha, args.margin, args.max_rounds, args.min_rounds, args.prior_sd, args.prior_df)
    if plan.min_rounds < 2 or plan.max_rounds < plan.min_rounds:
        raise SystemExit("need 2 <= --min-rounds <= --max-rounds")
    return plan


def lock_wanted(args: argparse.Namespace) -> bool:
    """False under --no-lock and when an outer script holds the lock already (TONA_GPU_LOCK_HELD,
    scripts/gpu-lock.sh)."""
    return not args.no_lock and not os.environ.get("TONA_GPU_LOCK_HELD")


@dataclass(frozen=True)
class Stop:
    primary: list | None = None  # selectors of the metrics that decide when to stop; None: all
    noise: dict | None = None  # {metric: (sd percent, df)} from bench/noise.toml
    expect: list | None = None  # [(selector, low, high)]: a one-round decision outside it is not final
    fixed: bool = False  # run all max_rounds (to collect noise for bench/noise.toml), decisions are still reported
    round_warmup: bool = False  # a discarded start of each arm right before its measured one
    discard: object = None  # () -> reason or None: a round with a system backup is repeated (coordinator rule)
    discarded: list = field(default_factory=list)  # the repeated rounds, listed in the report


def all_decided(rounds: list, plan: Plan, primary: list | None = None, noise=None, expect=None) -> bool:
    """Every primary metric (selectors as in --expect; none given: all) has a decision. The others are measured and
    reported, but do not hold the run."""
    names = [m for m in ab_report.metric_names(rounds) if not primary or any(ab_plan.matches(m, s) for s in primary)]
    decided = ab_report.decide(rounds, plan, noise, expect)
    return all(decided[m][0] for m in names)


def run_rounds(measure, plan: Plan, rounds: list, log=lambda n, entry: None, stop: Stop | None = None) -> str:
    """Appends rounds to `rounds` (so an interrupted run keeps the finished ones) until every metric is decided or
    max_rounds is reached. measure(key) -> (metrics, shas) for arm "A" or "B"; the order alternates AB, BA, AB, ...
    Returns the reason for stopping."""
    stop, n = stop or Stop(), 0
    while n < plan.max_rounds:
        n += 1
        order = "AB" if n % 2 else "BA"
        start, entry = time.monotonic(), {"order": order, "sha": {}}
        for key in order:
            if stop.round_warmup:
                measure(key)  # discarded: the page cache then holds this arm's model, not the other's
            result = measure(key)
            entry[key], entry["sha"][key] = result[0], result[1]
            if len(result) > 2:  # counters of the expert cache per request (hits, misses)
                entry.setdefault("counters", {})[key] = result[2]
        entry["seconds"] = time.monotonic() - start
        if stop.discard and (why := stop.discard()):
            stop.discarded.append({**entry, "discarded": why})
            print(f"round {n} discarded ({why}), repeated", file=sys.stderr, flush=True)
            n -= 1
            continue
        rounds.append(entry)
        log(n, entry)
        if not stop.fixed and all_decided(rounds, plan, stop.primary, stop.noise, stop.expect):
            return f"all metrics decided after round {n}"
    return f"{plan.max_rounds} rounds (undecided metrics stay open)"


class Measurer:
    """One server start per arm, round and job (--ctx and environment of a workload); its requests run in it."""

    def __init__(self, cfg, arms: dict, jobs: list, corpora: ab_server.Corpora, log_dir: Path):
        self.cfg, self.arms, self.jobs, self.corpora, self.log_dir = cfg, arms, jobs, corpora, log_dir
        self.several = ab_workloads.suffixes(jobs)
        self.prompts: dict = {}
        self.loads, self.load_log = [], False  # ab_load summary per server start, discarded ones included
        self.proc = None
        self.count = 0
        self.backup = False  # a system backup ran during a start since the last call of backup_seen()

    def __call__(self, key: str) -> tuple:
        metrics, shas, counters = {}, {}, {}
        for job in self.jobs:
            got_m, got_s, got_c = self.serve(key, job)
            metrics.update(got_m)
            shas.update(got_s)
            counters.update(got_c)
        return metrics, shas, counters

    def serve(self, key: str, job) -> tuple:
        self.count += 1
        cfg = replace(self.cfg, ctx=job.ctx, base_env={**self.cfg.base_env, **job.env})
        url = f"http://127.0.0.1:{cfg.port}"
        ctx_suffix = {r.name: f"/ctx{job.ctx}" for r in job.requests if self.several[r.name]}
        log_path = self.log_dir / f"server-{self.count:02d}-{key}.log"
        load = None
        ab_load.wait_backup()
        try:
            ab_server.check_port(cfg.port, wait=60)  # a previous server may still hold the port
            self.proc = ab_server.start_server(cfg, self.arms[key], log_path)
            load = ab_load.LoadLog().start() if self.load_log else None
            ab_server.wait_ready(self.proc, cfg.port, cfg.ready_timeout)
            if load:
                load.ready()
            counters: dict = {}
            metrics, shas = ab_server.run_context(url, self.corpora, job.requests, self.prompts, ctx_suffix, counters)
            if job.bench_set:
                suffix = f"/ctx{job.ctx}" if self.several["set"] else ""
                metrics.update({f"{m}{suffix}": v for m, v in ab_server.run_bench_set(url, ROOT).items()})
            return metrics, shas, counters
        finally:
            if load:
                self.loads.append({"start": self.count, "arm": key, **load.stop()})
            self.backup |= bool(load and self.loads[-1].get("backup_share")) or ab_load.backup_running()
            self.close()

    def backup_seen(self) -> str | None:
        seen, self.backup = self.backup, False
        return "a system backup ran" if seen else None

    def close(self) -> None:
        ab_server.stop_server(self.proc)
        self.proc = None


def load_corpus(path: Path) -> str:
    if not path.exists():
        print(f"{path} is missing, building it from {make_corpus.DEFAULT_COMMIT}", file=sys.stderr)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(make_corpus.build(make_corpus.DEFAULT_COMMIT), encoding="utf-8")
    return path.read_text(encoding="utf-8")


def setup(args: argparse.Namespace) -> tuple:
    workloads = [w for w in args.workload.split(",") if w]
    arms = ab_server.arms_from(args, ROOT, parse_env)
    if args.port == ab_server.FORBIDDEN_PORT:
        raise SystemExit(f"port {args.port} is the service's; choose another with --port")
    cfg = ab_server.ServerConfig(
        ROOT, str(ROOT / args.model), ab_server.drafter_path(ROOT, args.drafter), 0, args.port,  # ctx 0: see Measurer
        args.memory_max or default_memory_max(), parse_env(args.env), expert_profile=not args.no_expert_profile,
    )  # fmt: skip
    return cfg, arms, workloads


def arm_record(cfg, arm, name: str, env: dict) -> dict:
    full = ab_server.child_env(cfg, arm)
    commit = ab_plan.build_commit(ab_server.build_dir(cfg.root, arm))
    return {
        "build": name,
        "env": env,
        "tooling": ab_plan.tooling(full),
        "commit": commit,
        "model": arm.model or cfg.model,
        "drafter": arm.drafter or cfg.drafter,
    }


def reference_sha(corpus_sha: str, env: dict) -> dict:
    """The output SHAs of main (bench/sha.toml) when they can be compared: deterministic mode on both arms and the
    reference corpus."""
    path = ROOT / "bench/sha.toml"
    if corpus_sha != make_corpus.DEFAULT_SHA256 or not path.exists():
        return {}
    if not all(e.get("TONA_DETERMINISTIC") == "1" for e in env.values()):
        return {}
    return tomllib.loads(path.read_text()).get("server", {})


def config_record(args: argparse.Namespace, cfg, arms: dict, corpus: str, jobs: list) -> dict:
    env = {key: {**cfg.base_env, **arm.env} for key, arm in arms.items()}
    sha = hashlib.sha256(corpus.encode()).hexdigest()
    return {
        "A": arm_record(cfg, arms["A"], args.a, env["A"]),
        "B": arm_record(cfg, arms["B"], args.b, env["B"]),
        "workload": args.workload,
        "targets": ab_workloads.describe(jobs),
        "deterministic": all(e.get("TONA_DETERMINISTIC") == "1" for e in env.values()),
        "plan_info": {
            "expect": ab_plan.parse_expect(args.expect),
            "source": args.expect_source,
            "workload": args.workload,
            "targets": ab_workloads.describe(jobs),
            "primary": args.primary,
            "why_size": args.why_size,
            "decode_tokens": args.decode_tokens,
            "noise": {},
            "warmup": not args.no_warmup,
            "round_warmup": round_warmup(args),
            "load_log": ab_load.enabled(args.load_log),
            "A": arm_record(cfg, arms["A"], args.a, env["A"]),
            "B": arm_record(cfg, arms["B"], args.b, env["B"]),
        },
        "reference_sha": reference_sha(sha, env),
        "corpus_sha256": sha,
        "corpus_is_reference": sha == make_corpus.DEFAULT_SHA256,
        "commit": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip(),
    }


def write_reports(data: dict, plan: Plan, base: Path) -> str:
    base.parent.mkdir(parents=True, exist_ok=True)
    text = ab_report.markdown(data, plan)
    base.with_suffix(".json").write_text(ab_report.to_json(data, plan))
    base.with_suffix(".md").write_text(text)
    return text


def parse_targets(text: str, default_tokens: int) -> list:
    """'2000:400,130000' -> [(2000, 400), (130000, default_tokens)]."""
    out = []
    for item in filter(None, text.split(",")):
        target, _, tokens = item.partition(":")
        try:
            out.append((int(target), int(tokens) if tokens else default_tokens))
        except ValueError:
            raise SystemExit(f"--targets entry {item!r} is not TOKENS[:DECODE_TOKENS]") from None
    return out


def parse_ctxs(text: str) -> list:
    try:
        return [int(c) for c in text.split(",") if c]
    except ValueError:
        raise SystemExit(f"--ctx {text!r} is not a number or a comma separated list") from None


def announce(plan: Plan):
    """Progress line after a round, with the duration estimate refreshed from the measured round time."""

    def log(n: int, entry: dict) -> None:
        print(f"round {n} ({entry['order']}) done in {entry['seconds']:.0f} s", file=sys.stderr, flush=True)
        if n == 1:
            print(f"duration {ab_plan.duration_text(entry['seconds'], plan, True)}", file=sys.stderr, flush=True)

    return log


def round_warmup(args: argparse.Namespace) -> bool:
    """Off under Linux by default, so the reference's numbers stay comparable with earlier runs."""
    return args.round_warmup == "on" or (args.round_warmup == "auto" and sys.platform == "darwin")


def check_args(args: argparse.Namespace, metrics: list) -> None:
    """The refusals: no start without expectation, source and reason for the size, or with a stray --primary."""
    ab_plan.require_plan(metrics, ab_plan.parse_expect(args.expect), args.expect_source)
    if not args.why_size.strip():
        raise SystemExit(
            "no --why-size: say why this workload size answers the question (plan field, AGENTS.md rule 2)"
        )
    for sel in filter(None, args.primary.split(",")):
        if not any(ab_plan.matches(m, sel) for m in metrics):
            raise SystemExit(f"--primary {sel!r} matches no metric of this run ({', '.join(metrics)})")


def known_noise(args: argparse.Namespace, metrics: list, tokens: dict) -> tuple:
    """(table, {metric: (sd, df)} of the usable entries)."""
    table = {} if args.no_noise else ab_noise.load(ROOT / args.noise)
    if args.no_warmup:  # the table was measured with a warm-up; the cold first start would look like an effect
        return table, {}
    return table, ab_noise.noise_for(table, metrics, tokens)


def offer_noise(data: dict, table: dict, args: argparse.Namespace, tokens: dict, base: Path) -> None:
    """Proposes the noise table with this run added: written next to the report, and to the repository file only
    after a yes at a terminal. Never silent."""
    rounds = data["rounds"]
    if len(rounds) < 2:
        return
    diffs = {m: ab_report.diffs_of(rounds, m) for m in ab_report.metric_names(rounds)}
    proposal = ab_noise.propose(table, diffs, tokens, f"{base.name} ({len(rounds)} rounds, {args.a} vs {args.b})")
    target = base.with_name(base.name + "-noise.toml")
    target.write_text(ab_noise.dumps(proposal))
    print(f"noise table with this run added: {target}", file=sys.stderr)
    if sys.stdin.isatty() and input(f"update {args.noise} with it? [y/N] ").strip().lower() == "y":
        (ROOT / args.noise).write_text(ab_noise.dumps(proposal))
        print(f"{args.noise} updated", file=sys.stderr)
    else:
        print(f"not written; to take it: cp {target} {args.noise}", file=sys.stderr)


def execute(args: argparse.Namespace, plan: Plan, measure, data: dict, noise: dict) -> None:
    lock = os.environ.get("TONA_GPU_LOCK", DEFAULT_LOCK)
    start = time.monotonic()
    try:
        with GpuLock(lock, lock_wanted(args)):
            start = time.monotonic()  # the wait for the lock is not measurement time
            if not args.no_warmup:
                # Cold page cache, GPU clocks and expert cache favor the arm that runs first (round 1 of the
                # first A/A run: +0.7 % prefill, +1.0 to +1.8 % decode for B); not a measurement round
                print("warm-up start of A (discarded)", file=sys.stderr, flush=True)
                data["warmup"] = measure("A")[0]
                getattr(measure, "backup_seen", lambda: None)()  # a backup during the warm-up spoils no round
            primary = [p for p in args.primary.split(",") if p]
            stop = Stop(primary, noise, ab_plan.parse_expect(args.expect), round_warmup=round_warmup(args))
            stop = replace(stop, discard=getattr(measure, "backup_seen", None), discarded=data["discarded"])
            data["stop_reason"] = run_rounds(measure, plan, data["rounds"], announce(plan), stop)
    except KeyboardInterrupt:
        data["stop_reason"] = "interrupted (partial result)"
    except Exception as exc:
        data["stop_reason"] = f"error: {exc} (partial result)"
        print(f"error: {exc}", file=sys.stderr)
    finally:
        measure.close()
    data["wall_seconds"] = time.monotonic() - start


def main(argv: list) -> int:
    args = parse_args(argv)
    plan = plan_from(args)
    cfg, arms, workloads = setup(args)
    if not args.workload:
        raise SystemExit("no --workload: short, ctx, mid, long, set, huge (docs/contribute/measurement.md)")
    jobs = ab_workloads.jobs_for(workloads, parse_targets(args.targets, args.decode_tokens), parse_ctxs(args.ctx))
    ab_workloads.check_ctx(jobs)
    tokens = ab_workloads.tokens_by_target(jobs)
    metrics = ab_workloads.metric_names(jobs)
    check_args(args, metrics)
    table, noise = known_noise(args, metrics, tokens)
    corpus = load_corpus(ROOT / args.corpus)
    base = Path(args.out) if args.out else ROOT / "work/ab" / datetime.now().strftime("ab-%Y%m%d-%H%M%S")
    log_dir = base.parent / (base.name + "-logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    needs_prose = any(r.kind in ab_workloads.PROSE_KINDS for job in jobs for r in job.requests)
    prose = (ROOT / args.prose_corpus).read_text(encoding="utf-8") if needs_prose else ""
    measure = Measurer(cfg, arms, jobs, ab_server.Corpora(corpus, prose), log_dir)
    measure.load_log = ab_load.enabled(args.load_log)
    config = config_record(args, cfg, arms, corpus, jobs)
    config["noise"] = noise
    config["plan_info"]["noise"] = {
        m: [sd, df, table[(ab_noise.workload_of(m), m)]["source"]] for m, (sd, df) in noise.items()
    }
    data = {
        "config": config,
        "rounds": [],
        "discarded": [],
        "stop_reason": "",
        "wall_seconds": 0.0,
        "loads": measure.loads,
    }
    shown = replace(plan, min_rounds=1 if len(noise) == len(metrics) else plan.min_rounds)
    per_round = ab_plan.estimate_round_seconds(jobs) * (2 if round_warmup(args) else 1)
    plan_text = ab_plan.plan_markdown(config["plan_info"], shown, per_round)
    base.with_name(base.name + "-plan.md").write_text(plan_text)
    print(plan_text, flush=True)  # on record before the first server starts
    execute(args, plan, measure, data, noise)
    print(write_reports(data, plan, base))
    print(f"raw data: {base.with_suffix('.json')}", file=sys.stderr)
    offer_noise(data, table, args, tokens, base)
    return 0 if not data["stop_reason"].startswith(("error", "interrupted")) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
