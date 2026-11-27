"""Paired comparison of two arms from the result files of bench/agent/run.py (standard library only).

Every record is one (task, repeat) outcome per arm. Per suite and pooled: pass rates with Wilson intervals, the paired
table (both pass, only A, only B, neither), the exact McNemar test and the difference B - A with its interval over the
pooled pairs, and a bootstrap over TASKS (repeats of a task are correlated, so the task is the unit of resampling) of
the mean difference. The resolution line says what difference the pairs could resolve at the observed discordance.
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "needle"))
import stats

DRAWS = 4000


def load(results: Path, arm: str, suite: str) -> dict:
    """{(task, repeat): ok} of one arm and suite from results/ARM-SUITE-rN.json."""
    out = {}
    for f in sorted(results.glob(f"{arm}-{suite}-r*.json")):
        rep = int(f.stem.rsplit("-r", 1)[1])
        for r in json.loads(f.read_text()):
            out[(r["task"], rep)] = bool(r["ok"])
    return out


NEW_LANG_SUITES = ("java", "cpp", "php", "typescript", "dart", "bash")


def loop_count(results: Path, arm: str, suite: str) -> tuple[int, int]:
    """(tasks stopped as loop/timeout, tasks) of one arm and suite: model failures, not infrastructure failures."""
    rows = [r for f in sorted(results.glob(f"{arm}-{suite}-r*.json")) for r in json.loads(f.read_text())]
    return sum("reason" in r for r in rows), len(rows)


def pairs(a: dict, b: dict) -> list[tuple[tuple, bool, bool]]:
    return [(k, a[k], b[k]) for k in sorted(a) if k in b]


def task_bootstrap(rows: list, seed: int = 165) -> tuple[float, float, float]:
    """Mean per-task difference of the pass fraction (B - A) with a 95 % bootstrap interval over tasks."""
    per: dict = {}
    for (task, _), x, y in rows:
        per.setdefault(task, []).append(int(y) - int(x))
    diffs = [sum(v) / len(v) for v in per.values()]
    if len(diffs) < 2:
        return (diffs[0] if diffs else 0.0), 0.0, 0.0
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(DRAWS))
    return sum(diffs) / len(diffs), means[int(0.025 * DRAWS)], means[int(0.975 * DRAWS)]


def verdict(only_a: int, only_b: int, p: float, names: tuple[str, str]) -> str:
    """The per-language rule (plan of 2026-11-18): a net difference of 3 or more tasks decides, else nothing seen;
    a verdict with p above 0.05 is marked weak."""
    net = only_a - only_b
    if abs(net) < 3:
        return "no difference seen"
    return f"{names[0] if net > 0 else names[1]} better" + (" (weak)" if p > 0.05 else "")


def line(name: str, rows: list, names: tuple[str, str] = ("A", "B")) -> str:
    n = len(rows)
    if n == 0:
        return f"| {name} | 0 | | | | | | | |"
    pa, pb = sum(x for _, x, _ in rows), sum(y for _, _, y in rows)
    both = sum(x and y for _, x, y in rows)
    only_a, only_b = pa - both, pb - both
    d, lo, hi = stats.paired_diff(only_a, only_b, n)
    p = stats.mcnemar_exact(only_a, only_b)
    md, mlo, mhi = task_bootstrap(rows)
    wa, wb = stats.wilson(pa, n), stats.wilson(pb, n)
    return (
        f"| {name} | {n} | {pa}/{n} = {pa / n:.1%} [{wa[0]:.0%}, {wa[1]:.0%}]"
        f" | {pb}/{n} = {pb / n:.1%} [{wb[0]:.0%}, {wb[1]:.0%}]"
        f" | {both} / {only_a} / {only_b} / {n - both - only_a - only_b}"
        f" | {d * 100:+.1f} [{lo * 100:+.1f}, {hi * 100:+.1f}] (p {p:.3f})"
        f" | {md * 100:+.1f} [{mlo * 100:+.1f}, {mhi * 100:+.1f}]"
        f" | {verdict(only_a, only_b, p, names)} |"
    )


def resolution(rows: list) -> str:
    n = len(rows)
    if n < 5:
        return "too few pairs for a resolution statement"
    psi = sum(x != y for _, x, y in rows) / n
    if psi == 0:
        return f"no discordant pair in {n}"
    return (
        f"{n} pairs, discordance {psi:.0%}: at 80 % power a difference of "
        f"{stats.detectable(psi, n) * 100:.0f} points or more is resolved; a smaller one is not"
    )


HEAD = (
    "| Suite | Pairs | A pass | B pass | both / only A / only B / none | B - A points, pooled pairs [95 %] "
    "| B - A points, task bootstrap [95 %] | Verdict |\n|---|---|---|---|---|---|---|---|"
)


def language_rows(rows_by: dict, suites: list[str], names: tuple[str, str]) -> list[str]:
    out = []
    if "polyglot" in rows_by:
        # Per language: ISTA says Rust was not preserved (research/64-coder-shard-public-evidence.md)
        langs = sorted({k[0].split("/")[0] for k, _, _ in rows_by["polyglot"]})
        for lang in langs:
            sub = [r for r in rows_by["polyglot"] if r[0][0].split("/")[0] == lang]
            out.append(line(f"polyglot / {lang}", sub, names))
    new = [s for s in suites if s in NEW_LANG_SUITES]
    if len(new) > 1:  # the six language suites each are one language; this row is their sum, the rows above stay
        out.append(line("new languages (sum)", [r for s in new for r in rows_by[s]], names))
    return out


def tables(results: Path, a: str, b: str, suites: list[str]) -> str:
    rows_by = {s: pairs(load(results, a, s), load(results, b, s)) for s in suites}
    out = [HEAD]
    allrows = []
    for s in suites:
        out.append(line(s, rows_by[s], (a, b)))
        allrows += [(((s, k[0]), k[1]), x, y) for k, x, y in rows_by[s]]  # key stays (task, repeat)
    out += language_rows(rows_by, suites, (a, b))
    if len(suites) > 1:
        out.append(line("all suites", allrows, (a, b)))
    if "swe" in rows_by:
        (la, na), (lb, nb) = loop_count(results, a, "swe"), loop_count(results, b, "swe")
        out.append(
            f"\nSWE tasks stopped as loop/timeout (model failures, counted as failed): {a} {la}/{na}, {b} {lb}/{nb}"
        )
    out.append("")
    for s in [*suites, "all suites"] if len(suites) > 1 else suites:
        rows = allrows if s == "all suites" else rows_by[s]
        out.append(f"- {s}: {resolution(rows)}")
    return "\n".join(out)


def per_task(results: Path, a: str, b: str, suite: str) -> str:
    """Pass counts per task over the repeats (task ids only, no content)."""
    ra, rb = load(results, a, suite), load(results, b, suite)
    tasks = sorted({t for t, _ in ra} | {t for t, _ in rb})
    cells = []
    for t in tasks:
        ka = [v for (tt, _), v in ra.items() if tt == t]
        kb = [v for (tt, _), v in rb.items() if tt == t]
        cells.append(f"{t}: {sum(ka)}/{len(ka)} vs {sum(kb)}/{len(kb)}")
    return "; ".join(cells)
