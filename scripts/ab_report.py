"""Report of scripts/ab.py: a Markdown table for the measurement file and the raw data as JSON."""

import json
from statistics import median

import ab_load
import ab_plan
from ab_stats import FASTER, OPEN, SLOWER, Plan, first_decision, summarize, with_noise

RATE_NAMES = {"prefill": "Prefill tok/s", "decode": "Decode tok/s", "decode-code": "Decode tok/s (bench-set code)"}


def metric_label(metric: str) -> str:
    name, _, ctx = metric.partition("/ctx")
    kind, _, target = name.partition("@")
    base = {"decode-text": "Decode tok/s (bench-set text)"}.get(name) or RATE_NAMES.get(kind, kind)
    after = f" after {target} tokens of prompt" if target else ""
    return f"{metric}: {base}{after}" + (f", --ctx {ctx}" if ctx else "")


def metric_names(rounds: list) -> list:
    return list(rounds[0]["A"]) if rounds else []


def diffs_of(rounds: list, metric: str) -> list:
    """Paired difference B against A in percent, one per round."""
    return [100 * (r["B"][metric] / r["A"][metric] - 1) for r in rounds]


def plan_of(metric: str, plan: Plan, noise: dict | None) -> Plan:
    """The plan for one metric: with known noise (bench/noise.toml) one round may decide."""
    return with_noise(plan, *noise[metric]) if noise and metric in noise else plan


OUTLIER_SD = 4.0  # standard deviations of the noise table from the median of the earlier rounds


def round_flags(diffs: list, sd: float | None, expected: tuple | None) -> dict:
    """{round: 'outlier' | 'off plan'} for metrics with known noise sd. Nothing is dropped, the flags only keep a
    decision from being taken on that round.

    Round 1 is 'off plan' when its difference is outside the expected range of the plan. From round 2 on a round is an
    'outlier' when it is more than OUTLIER_SD noise sds away from the median of the earlier rounds and not inside the
    expected range (a round that agrees with the plan is not suspicious). Two rounds that agree with each other, even
    far from the plan, do not flag each other: that is a finding for the plan check, not an outlier.
    """
    if sd is None:
        return {}
    inside = (lambda d: expected[0] <= d <= expected[1]) if expected else (lambda d: False)
    flags = {1: "off plan"} if expected and not inside(diffs[0]) else {}
    for n in range(2, len(diffs) + 1):
        d = diffs[n - 1]
        if not inside(d) and abs(d - median(diffs[: n - 1])) > OUTLIER_SD * sd:
            flags[n] = "outlier"
    return flags


def flags_of(rounds: list, noise: dict | None, expect: list | None) -> dict:
    """{metric: {round: flag}} for all metrics with known noise."""
    out = {}
    for metric in metric_names(rounds):
        sd = noise[metric][0] if noise and metric in noise else None
        out[metric] = round_flags(diffs_of(rounds, metric), sd, ab_plan.expectation_for(metric, expect or []))
    return out


def decide(rounds: list, plan: Plan, noise: dict | None = None, expect: list | None = None) -> dict:
    """Per metric: (round of the decision or None, Look at the decision, Look of all rounds for the interval).

    With known noise one round may decide, but only if that round is inside the plan and no outlier; otherwise the
    metric needs a further round (min_rounds 2 for it, as without a table).
    """
    out, flags = {}, flags_of(rounds, noise, expect)
    for metric in metric_names(rounds):
        diffs, own = diffs_of(rounds, metric), plan_of(metric, plan, noise)
        k, look = first_decision(diffs, own, lambda n, f=flags[metric]: n not in f)
        final = summarize(diffs, own) if len(diffs) >= own.min_rounds else None
        if k and not confirmed(look.decision, final):
            k, look = None, final
        out[metric] = (k, look, final)
    return out


def confirmed(decision: str, final) -> bool:
    """An early "B faster" or "B slower" stands only while the interval of all rounds still excludes zero on its side;
    otherwise the report would claim a difference that its own interval contradicts (2026-11-12-mac-iq3xxs-repack.md:
    "B faster after round 2", interval over 6 rounds [-3.65, +7.15])."""
    if final is None or decision not in (FASTER, SLOWER):
        return True
    return final.ci_diff[0] > 0 if decision == FASTER else final.ci_diff[1] < 0


def fmt_values(values: list) -> str:
    return " / ".join(f"{v:.1f}" for v in values)


def decision_text(k, look) -> str:
    if look is None:
        return "open (one round)"
    if k is None:
        return "open (undecided at the last round)"
    note = " (real, but inside the margin)" if look.inside_margin and look.decision != "equal" else ""
    return f"**{look.decision}** after round {k}{note}"


def metric_rows(rounds: list, plan: Plan, noise: dict | None, expect: list | None) -> list:
    rows = [
        "| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |"
    ]
    rows.append("|---|---|---|---|---|---|---|")
    for metric, (k, look, final) in decide(rounds, plan, noise, expect).items():
        a, b = [r["A"][metric] for r in rounds], [r["B"][metric] for r in rounds]
        ci = f"{final.mean:+.2f} % [{final.ci_diff[0]:+.2f}, {final.ci_diff[1]:+.2f}]" if final else "-"
        rows.append(
            f"| {metric_label(metric)} | {fmt_values(a)} | {fmt_values(b)} "
            f"| {median(a):.1f} ({min(a):.1f}-{max(a):.1f}) | {median(b):.1f} ({min(b):.1f}-{max(b):.1f}) "
            f"| {ci} | {decision_text(k, look)} |"
        )
    return rows


def round_rows(rounds: list, flags: dict | None = None) -> list:
    """One row per round; a round that was flagged (off plan, outlier) says so. It stays in every number."""
    flags = flags or {}
    rows = ["| Round | Order | " + " | ".join(f"{m} B vs A" for m in metric_names(rounds)) + " | Seconds |"]
    rows.append("|---|---|" + "---|" * (len(metric_names(rounds)) + 1))
    for i, r in enumerate(rounds, 1):
        cells = " | ".join(
            f"{100 * (r['B'][m] / r['A'][m] - 1):+.2f} %" + (f" ({flags[m][i]})" if i in flags.get(m, {}) else "")
            for m in metric_names(rounds)
        )
        rows.append(f"| {i} | {r['order']} | {cells} | {r['seconds']:.0f} |")
    return rows


def sha_row(rounds: list, target: str, ref: str | None) -> tuple:
    """(table row, deviates) for one request: stable per arm, equal between the arms, equal to the reference."""
    a = [r["sha"]["A"][target] for r in rounds]
    b = [r["sha"]["B"][target] for r in rounds]
    stable_a, stable_b = len(set(a)) == 1, len(set(b)) == 1
    same = len(set(a + b)) == 1
    ref_ok = ref is None or (b[0] == ref and stable_b)
    ref_text = "-" if ref is None else f"{ref} ({'B = ref' if ref_ok else 'B DIFFERS'})"
    yes = {True: "yes", False: "NO"}
    row = f"| {target} | {a[0]} | {b[0]} | {yes[stable_a]} | {yes[stable_b]} | {yes[same]} | {ref_text} |"
    return row, not (stable_a and stable_b and same and ref_ok)


def sha_rows(rounds: list, deterministic: bool, reference: dict | None = None) -> list:
    """Equality of the output hashes: stable per arm over the rounds, between the arms, and against the reference
    of main (bench/sha.toml) when there is one for the request."""
    reference = reference or {}
    results = [sha_row(rounds, t, reference.get(t)) for t in (rounds[0]["sha"]["A"] if rounds else [])]
    if not results:
        return []
    head = [
        "| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |",
        "|---|---|---|---|---|---|---|",
    ]
    out = head + [row for row, _ in results]
    if any(bad for _, bad in results) and deterministic:
        out += [
            "",
            "**WARNING: deterministic mode, but the texts differ between rounds, arms or from the reference of "
            "main. The change is not bit-exact (or the run is not deterministic).**",
        ]
    elif any(bad for _, bad in results):
        out += ["", "The texts differ (normal mode: sampling of the drafter and rounding may differ; no verdict)."]
    return out


def counter_rows(rounds: list) -> list:
    """Expert cache hits and misses per request (differences of the /status counters of the server). They are counts,
    not rates: no statistics, only whether they are identical over the rounds and between the arms."""
    if not rounds or not rounds[0].get("counters", {}).get("A"):
        return []
    rows = ["| Request | A hits / misses per round | B hits / misses per round | identical |", "|---|---|---|---|"]
    for name in rounds[0]["counters"]["A"]:
        cells, seen = [], []
        for arm in ("A", "B"):
            values = [
                (r["counters"][arm][name]["expert_hits"], r["counters"][arm][name]["expert_misses"]) for r in rounds
            ]
            cells.append(" / ".join(f"{h}, {m}" for h, m in values))
            seen += values
        rows.append(f"| {name} | {cells[0]} | {cells[1]} | {'yes' if len(set(seen)) == 1 else 'NO'} |")
    return rows + pass_rows(rounds)


def pass_rows(rounds: list) -> list:
    """Tokens per decode pass and MTP draft steps per pass of every request, per round and arm (the acceptance behind
    the decode rate; counts of /status, no statistics). The server's own log of the run holds the same per request."""
    first = rounds[0]["counters"]["A"]
    if not all("passes" in v for v in first.values()):
        return []
    head = "| Request | A tokens per pass (draft steps per pass) | B tokens per pass (draft steps per pass) |"
    rows = ["", head, "|---|---|---|"]
    for name in first:
        cells = []
        for arm in ("A", "B"):
            vals = []
            for r in rounds:
                c = r["counters"][arm][name]
                n = max(c["passes"], 1)
                vals.append(f"{c['gen_tokens'] / n:.2f} ({c['mtp_passes'] / n:.2f})")
            cells.append(" / ".join(vals))
        rows.append(f"| {name} | {cells[0]} | {cells[1]} |")
    return rows


def timing_lines(rounds: list, wall: float) -> list:
    n = len(rounds)
    per = sum(r["seconds"] for r in rounds) / max(n, 1)
    return [
        f"Runtime: {wall / 60:.1f} min for {n} rounds ({per / 60:.1f} min per round); a fixed 3 rounds would take "
        f"{3 * per / 60:.1f} min ({(3 - n) * per / 60:+.1f} min saved)."
    ]


def markdown(data: dict, plan: Plan) -> str:
    rounds = data["rounds"]
    if not rounds:
        return "No round completed.\n"
    cfg = data["config"]
    noise = cfg.get("noise")
    expect = (cfg.get("plan_info") or {}).get("expect")
    head = [
        f"A: `{cfg['A']['build']}` env {cfg['A']['env'] or '-'}; "
        f"B: `{cfg['B']['build']}` env {cfg['B']['env'] or '-'}. "
        f"Alpha {plan.alpha}, margin {plan.margin} %, rounds {len(rounds)} of at most {plan.max_rounds}, "
        f"stop: {data['stop_reason']}. All rounds are listed; none was dropped.",
        "",
    ]
    gone = data.get("discarded", [])
    if gone:  # a round with a system backup is repeated; its numbers stay out of the decision but in the report
        head[0] = head[0].replace("none was dropped", f"{len(gone)} discarded, below")
        head += ["Discarded (repeated) rounds:", "", *round_rows(gone), ""]
        head += [f"- {r['order']}: {r['discarded']}" for r in gone] + [""]
    info = cfg.get("plan_info")
    per_round = sum(r["seconds"] for r in rounds) / len(rounds)
    top = [ab_plan.plan_markdown(info, plan, per_round, True)] if info else []
    parts = [
        *top,
        "## Result",
        "",
        *head,
        *metric_rows(rounds, plan, noise, expect),
        "",
        *round_rows(rounds, flags_of(rounds, noise, expect)),
        "",
    ]
    if info:
        means = [(m, f.mean) for m, (_, _, f) in decide(rounds, plan, noise, expect).items() if f]
        parts += ["Result against the plan:", "", *ab_plan.check_rows(means, info["expect"]), ""]
    counters = counter_rows(rounds)
    if counters:
        parts += ["Expert cache per request (counts from /status, no statistics):", "", *counters, ""]
    loads = ab_load.load_rows(data.get("loads", []))
    if loads:
        parts += ["Load per server start (percent of one core or of the GPU's time; ours = server and client):", ""]
        parts += [*loads, ""]
    parts += [
        *sha_rows(rounds, cfg["deterministic"], cfg.get("reference_sha")),
        "",
        *timing_lines(rounds, data["wall_seconds"]),
    ]
    return "\n".join(parts) + "\n"


def to_json(data: dict, plan: Plan) -> str:
    decisions = {}
    cfg = data["config"]
    expect = (cfg.get("plan_info") or {}).get("expect")
    found = decide(data["rounds"], plan, cfg.get("noise"), expect) if data["rounds"] else {}
    for metric, (k, look, final) in found.items():
        decisions[metric] = {
            "decided_at_round": k,
            "decision": look.decision if k else OPEN,
            "mean_diff_percent": final.mean if final else None,
            "ci_diff_percent": final.ci_diff if final else None,
        }
    return json.dumps({**data, "plan": plan.__dict__, "decisions": decisions}, indent=2, default=list) + "\n"
