"""Known noise of the workloads (bench/noise.toml) and how a run adds to it.

An entry is the standard deviation, in percent, of the paired difference B against A between two measurements of the
same build (an A/A run), per workload and metric, with the degrees of freedom of that estimate and its source. With
at least MIN_DF of them ab.py uses the entry as the variance of the decision rule (scripts/ab_stats.py): the sample
variance of the earlier runs is pooled with that of the new rounds, which is the pooled-variance t test, exact when
the noise is the same (Student 1908; for a variance known from earlier data: Gelman et al., Bayesian Data Analysis,
3rd ed., section 3.3). Then one round can decide a clear effect. Without an entry, or with too few earlier runs, the
rule keeps its minimum of two rounds.

Entries are valid for the workload they were measured on (same prompts, same decode length, same --ctx, same
hardware state).
After a run ab.py proposes the new table; it never writes it unasked.
"""

import math
import re
import tomllib
from pathlib import Path

MIN_DF = 4  # five earlier runs
DEFAULT_DECODE_TOKENS = 200


def load(path: Path) -> dict:
    """{(workload, metric): entry} from the TOML file; a missing file is an empty table."""
    if not path.exists():
        return {}
    raw = tomllib.loads(path.read_text())
    return {(w, m): e for w, metrics in raw.items() if isinstance(metrics, dict) for m, e in metrics.items()}


def target_of(metric: str):
    """The prompt length of a server metric (prefill@2000, decode@130000/ctx262144), else None."""
    m = re.match(r"^[a-z]+@(\d+)", metric)
    return int(m.group(1)) if m else None


def tokens_of(metric: str, tokens: dict) -> int:
    """Decode length of the request behind a metric; tokens is {target: decode tokens}."""
    return tokens.get(target_of(metric), DEFAULT_DECODE_TOKENS)


def usable(table: dict, workload: str, metric: str, tokens: int):
    """The entry for a metric if it is based on enough runs and on the same decode length, else None."""
    entry = table.get((workload, metric))
    if not entry or entry.get("df", 0) < MIN_DF:
        return None
    if workload == "server" and entry.get("decode_tokens", DEFAULT_DECODE_TOKENS) != tokens:
        return None
    return entry


def workload_of(metric: str) -> str:
    return "server" if "@" in metric else "bench-set"  # a /ctx<N> suffix is part of the metric name


def noise_for(table: dict, metrics: list, tokens: dict) -> dict:
    """{metric: (sd percent, df)} for the metrics that have a usable entry."""
    out = {}
    for m in metrics:
        entry = usable(table, workload_of(m), m, tokens_of(m, tokens))
        if entry:
            out[m] = (entry["sd_percent"], entry["df"])
    return out


def pooled(old, new_sd: float, new_df: int):
    """(sd, df) of two variance estimates; old is (sd, df) or None."""
    if not old or old[1] <= 0:
        return new_sd, new_df
    df = old[1] + new_df
    return math.sqrt((old[1] * old[0] ** 2 + new_df * new_sd**2) / df), df


def propose(table: dict, diffs: dict, tokens: dict, source: str) -> dict:
    """The table with the rounds of this run added: diffs is {metric: [paired differences in percent]}.

    The sample sd around the mean of the rounds is the noise estimate (n - 1 degrees of freedom); if the two arms
    differ by a stable effect, that effect is not noise and drops out with the mean.
    """
    new = dict(table)
    for metric, values in diffs.items():
        if len(values) < 2:
            continue
        mean = sum(values) / len(values)
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
        key = (workload_of(metric), metric)
        old = table.get(key)
        decode_tokens = tokens_of(metric, tokens)
        same = old and (key[0] != "server" or old.get("decode_tokens", DEFAULT_DECODE_TOKENS) == decode_tokens)
        sd2, df2 = pooled((old["sd_percent"], old["df"]) if same else None, sd, len(values) - 1)
        new[key] = {
            "sd_percent": round(sd2, 3),
            "df": df2,
            "decode_tokens": decode_tokens,
            "source": (old["source"] + "; " if same else "") + source,
        }
    return new


def dumps(table: dict) -> str:
    out = [
        "# Known noise of the workloads of scripts/ab.py: sd_percent is the standard deviation of the paired",
        "# difference B against A in percent between two measurements of the same build; df its degrees of freedom",
        "# (paired differences minus one per run set). Entries with df < 4 are listed but not used. See",
        "# scripts/ab_noise.py. ab.py proposes updates after a run; it never writes this file unasked.",
    ]
    for workload in sorted({w for w, _ in table}):
        for (w, metric), e in sorted(table.items()):
            if w != workload:
                continue
            out += ["", f'[{w}."{metric}"]', f"sd_percent = {e['sd_percent']}", f"df = {e['df']}"]
            if w == "server":
                out.append(f"decode_tokens = {e.get('decode_tokens', DEFAULT_DECODE_TOKENS)}")
            out.append('source = "' + e["source"].replace('"', "'") + '"')
    return "\n".join(out) + "\n"
