"""Plan of a measurement run (AGENTS.md rule 2: plan every run before it starts) and the check against it.

The plan block is printed and saved before the first server starts, so the expectation is on record before there is
a result. It holds the scope, the expectation of the user (B against A in percent, a range per metric with its
source) and the tooling that is on, read from the environment each arm actually gets.
"""

import itertools
import re
import subprocess
from pathlib import Path

TOOLING_PREFIXES = ("TONA_", "HSA_", "HIP_", "OMP_", "ROCR_")
EXPECT = re.compile(r"^([\w@.:-]+):([+-]?\d+(?:\.\d+)?)\.\.([+-]?\d+(?:\.\d+)?)%?$")
# Rough times for the estimate before round 1 (the A/A runs of research/measurements/2026-11-02-ab-tool.md: a round
# of 30K + 130K takes 240 s): server start 20 s, prefill by prefill_seconds, decode at 65 tok/s, bench-set about 200 s
START_S, BENCH_SET_S = 20.0, 200.0


def parse_expect(items: list) -> list:
    """['decode:-0.5..+0.5%'] -> [('decode', -0.5, 0.5)]: expected B against A in percent per metric (or group)."""
    out = []
    for item in items:
        m = EXPECT.match(item.replace(" ", ""))
        if not m:
            raise SystemExit(f"--expect {item!r} is not METRIC:LOW..HIGH% (example: decode:-0.5..+0.5%)")
        lo, hi = float(m.group(2)), float(m.group(3))
        if lo > hi:
            raise SystemExit(f"--expect {item!r}: low is above high")
        out.append((m.group(1), lo, hi))
    return out


def matches(metric: str, selector: str) -> bool:
    kind = metric.split("@", maxsplit=1)[0]
    return selector in (metric, kind) or kind.startswith(selector + "-")


def expectation_for(metric: str, expect: list):
    """The range for a metric: an entry with its exact name wins over one for its group (decode, prefill)."""
    kind = metric.split("@", maxsplit=1)[0]
    for sel, lo, hi in expect:
        if sel == metric:
            return lo, hi
    for sel, lo, hi in expect:
        if sel == kind or kind.startswith(sel + "-"):
            return lo, hi
    return None


def require_plan(metrics: list, expect: list, source: str) -> None:
    """Refuses the start without a written expectation for every metric the run will report."""
    if not source.strip():
        raise SystemExit("no --expect-source: say where the expectation comes from (a measurement file, a paper)")
    missing = [m for m in metrics if expectation_for(m, expect) is None]
    if missing:
        raise SystemExit(f"no --expect for: {', '.join(missing)} (example: --expect 'decode:-0.5..+0.5%')")


def tooling(env: dict) -> dict:
    return {k: v for k, v in sorted(env.items()) if k.startswith(TOOLING_PREFIXES)}


def build_commit(build: Path) -> str:
    """The commit a build was made from: the stamp CMake writes next to tona-server (git describe --dirty).

    Without a stamp (a build from before the stamp existed) it falls back to the source HEAD at run time, which can
    be newer than the binary, and says so; 'unknown' when even that is not available."""
    try:
        stamp = (build / "tona-server.commit").read_text().strip()
        if stamp:
            return stamp
    except OSError:
        pass
    try:
        cache = (build / "CMakeCache.txt").read_text()
        src = re.search(r"^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$", cache, re.M).group(1)
        out = subprocess.run(
            ["git", "-C", src, "describe", "--always", "--dirty", "--abbrev=10"],
            capture_output=True,
            text=True,
            check=False,
        )
        return f"{out.stdout.strip()} (source HEAD at run time)" if out.stdout.strip() else "unknown"
    except (OSError, AttributeError):
        return "unknown"


def prefill_seconds(tokens: int) -> float:
    """Prefill time interpolated between the measured points (docs/contribute/measurement.md, "What a run costs")."""
    pts = [(0, 0.0), (30000, 17.0), (130000, 76.0), (512000, 343.0), (1000000, 911.0)]
    for (x0, y0), (x1, y1) in itertools.pairwise(pts):
        if tokens <= x1:
            return y0 + (y1 - y0) * (tokens - x0) / (x1 - x0)
    return pts[-1][1] * tokens / pts[-1][0]


def estimate_round_seconds(jobs: list) -> float:
    """Both arms, every job: one server start and its requests."""
    total = 0.0
    for job in jobs:
        total += START_S + (BENCH_SET_S if job.bench_set else 0)
        total += sum(prefill_seconds(r.tokens) + r.decode / 65 for r in job.requests)
    return 2 * total


def duration_text(per_round: float, plan, measured: bool) -> str:
    lo, hi = plan.min_rounds * per_round / 60, plan.max_rounds * per_round / 60
    basis = "from round 1" if measured else "estimate before round 1"
    return f"{lo:.0f}-{hi:.0f} min ({plan.min_rounds}-{plan.max_rounds} rounds of {per_round / 60:.1f} min, {basis})"


def noise_line(info: dict) -> str:
    known = info.get("noise") or {}
    if not known:
        return "- Known noise: none for this workload (bench/noise.toml); at least 2 rounds."
    parts = [f"{m} sd {sd} % (df {df})" for m, (sd, df, _) in known.items()]
    return "- Known noise (bench/noise.toml; one round may decide these metrics): " + ", ".join(parts)


ROUND_WARMUP = {True: "; a discarded start of each arm before its measured one, every round.", False: "."}


def plan_markdown(info: dict, plan, per_round: float, measured: bool = False) -> str:
    """The block for the top of the measurement file."""
    lines = [
        "## Plan (written before the run)",
        "",
        f"- Scope: workload {info['workload']} ({info['targets']}); "
        f"A `{info['A']['build']}` against B `{info['B']['build']}`; "
        f"rounds {plan.min_rounds} to {plan.max_rounds}, alternating AB/BA, "
        f"stop when the primary metrics ({info.get('primary') or 'all'}) are decided "
        f"(alpha {plan.alpha}, margin {plan.margin} %); "
        + ("one discarded warm-up start of A first" if info.get("warmup") else "no warm-up start")
        + ROUND_WARMUP[bool(info.get("round_warmup"))],
        f"- Why this size: {info['why_size']} (decode tokens per prompt as in the targets, default "
        f"{info['decode_tokens']})",
        f"- Duration: {duration_text(per_round, plan, measured)}",
        noise_line(info),
        f"- Expectation (B against A, source: {info['source']}):",
    ]
    lines += [f"  - {sel}: {lo:+g} to {hi:+g} %" for sel, lo, hi in info["expect"]]
    lines.append("- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):")
    for key in ("A", "B"):
        env = ", ".join(f"{k}={v}" for k, v in info[key]["tooling"].items()) or "none"
        lines.append(f"  - {key}: build commit {info[key]['commit']}; {env}")
    return "\n".join(lines) + "\n"


def check_rows(rows: list, expect: list) -> list:
    """rows: (metric, mean diff in percent). One line per metric: inside the expectation, or the finding."""
    out = ["| Metric | Measured B vs A | Expected | Verdict |", "|---|---|---|---|"]
    for metric, mean in rows:
        lo, hi = expectation_for(metric, expect)
        verdict = "within the expectation" if lo <= mean <= hi else "**OUTSIDE the expectation: finding**"
        out.append(f"| {metric} | {mean:+.2f} % | {lo:+g} to {hi:+g} % | {verdict} |")
    return out
