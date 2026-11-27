"""Report as JSON (raw data) and Markdown (for research/measurements/), plus the comparison of two runs."""

import json
from pathlib import Path

LABELS = {
    "ttft_s": "TTFT s",
    "prefill_tps_client": "Prefill tok/s (client)",
    "prefill_tps_server": "Prefill tok/s (server)",
    "decode_tps_client": "Decode tok/s (client)",
    "decode_tps_server": "Decode tok/s (server)",
    "tokens_per_pass": "Tokens per pass",
    "draft_accept_rate": "Draft acceptance",
    "e2e_s": "Total s",
    "prompt_tokens": "Prompt tokens",
    "cached_tokens": "of which cached",
    "completion_tokens": "Output tokens",
    "itl_p50_ms": "Chunk gap p50 ms",
    "itl_p99_ms": "Chunk gap p99 ms",
    "vram_peak_mib": "VRAM peak MiB",
    "gtt_peak_mib": "GTT peak MiB",
    "ram_avail_min_mib": "RAM available min MiB",
    "swap_used_max_mib": "Swap max MiB",
    "gpu_busy_mean": "GPU load %",
    "power_mean_w": "Power W",
    "temp_max_c": "Temp. max °C",
    "sclk_mean_mhz": "Clock MHz",
}

# Columns of the overview table
OVERVIEW = [
    "ttft_s",
    "prefill_tps_server",
    "decode_tps_server",
    "decode_tps_client",
    "tokens_per_pass",
    "vram_peak_mib",
]


def num(v: float | None) -> str:
    if v is None:
        return "-"
    a = abs(v)
    if a >= 1000:
        return f"{v:,.0f}"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def cell(s: dict | None) -> str:
    """Median (min-max)."""
    if not s:
        return "-"
    if s["n"] == 1:
        return num(s["median"])
    return f"{num(s['median'])} ({num(s['min'])}-{num(s['max'])})"


def with_ext(base: Path, ext: str) -> Path:
    """Appends an extension; with_suffix would treat ".1" in "rocm-10.1" as one."""
    return base.parent / (base.name + ext)


def unique_path(directory: Path, stem: str) -> Path:
    """One file per run: appends -2, -3, ... if the name is taken."""
    p, i = directory / stem, 2
    while with_ext(p, ".md").exists() or with_ext(p, ".json").exists():
        p, i = directory / f"{stem}-{i}", i + 1
    return p


def write(data: dict, base: Path) -> tuple[Path, Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    jp, mp = with_ext(base, ".json"), with_ext(base, ".md")
    jp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    mp.write_text(markdown(data, jp.name))
    return jp, mp


def _env_rows(env: dict) -> list[tuple[str, str]]:
    t, g, p = env["target"], env.get("gpu", {}), env.get("packages", {})
    rows = [
        ("Target", f"`{t['name']}`, {t['url']}, API {t['api']}, model `{t['model']}`"),
        ("Time", env["time"]),
        ("Repo", f"{env['git']['commit']}{' (modified)' if env['git']['dirty'] else ''}"),
        ("OS / kernel", f"{env.get('os')} / {env.get('kernel')}"),
        ("Packages", ", ".join(f"{k} {v}" for k, v in p.items()) or "-"),
        ("CPU", f"{env.get('cpu')}, governor {env.get('cpu_governor')}, EPP {env.get('cpu_epp')}"),
        (
            "RAM",
            f"{env['mem_total_gib']} GiB, {env['mem_avail_gib']} GiB available before start, "
            f"swap {env['swap_used_mib']} MiB",
        ),
        (
            "GPU",
            f"VRAM {g.get('vram_total_mib')} MiB, {g.get('vram_used_mib')} MiB used before start, "
            f"PCIe {g.get('current_link_speed')} x{g.get('current_link_width')}, "
            f"DPM {g.get('power_dpm_force_performance_level')}",
        ),
    ]
    if t.get("extra"):
        rows.append(("Target extras", f"`{json.dumps(t['extra'], ensure_ascii=False)}`"))
    if t.get("note"):
        rows.append(("Note", t["note"]))
    return rows


def _overview(data: dict) -> list[str]:
    out = ["", "## Overview", "", "Median (min-max) over the measured runs, without warm-up runs.", ""]
    out.append("| Case | ok | Expectation | " + " | ".join(LABELS[k] for k in OVERVIEW) + " |")
    out.append("|---|---|---|" + "---:|" * len(OVERVIEW))
    for c in data["cases"]:
        for k, s in enumerate(c["summary"]["steps"]):
            name = c["case"]["name"] + (f" #{k + 1}" if len(c["summary"]["steps"]) > 1 else "")
            exp = f"{s['expect_pass']}/{s['expect_n']}" if s["expect_n"] else "-"
            cells = " | ".join(cell(s["metrics"].get(m)) for m in OVERVIEW)
            out.append(f"| {name} | {s['ok']}/{s['n']} | {exp} | {cells} |")
    return out


def _step_table(s: dict) -> list[str]:
    out = ["", "| Metric | Median | Min | Max | stdev | n |", "|---|---:|---:|---:|---:|---:|"]
    for key, v in s["metrics"].items():
        out.append(
            f"| {LABELS[key]} | {num(v['median'])} | {num(v['min'])} | {num(v['max'])} | {num(v['stdev'])} | {v['n']} |"
        )
    out += ["", f"Distinct outputs: {s['distinct_outputs']} of {s['ok']}."]
    return out


def _case_section(c: dict) -> list[str]:
    case, summ = c["case"], c["summary"]
    out = ["", f"### {case['name']}", ""]
    if case.get("description"):
        out += [case["description"], ""]
    out.append(
        f"{case['runs']} runs, {case['warmup']} warm-up, cache {case['cache']}, "
        f"max_tokens {case['max_tokens']}, temperature {case['temperature']}."
    )
    if summ.get("run_e2e_s") and len(summ["steps"]) > 1:
        out.append(f"Whole trace: {cell(summ['run_e2e_s'])} s.")
    for k, s in enumerate(summ["steps"]):
        if len(summ["steps"]) > 1:
            out += ["", f"Step {k + 1}:"]
        out += _step_table(s)
    first = next((st for r in c["runs"] if not r["warmup"] for st in r["steps"] if st["ok"]), None)
    if first and first.get("text_head"):
        head = first["text_head"].replace("```", "'''")
        out += ["", "Start of the answer (first run):", "", "```text", head, "```"]
    return out


def markdown(data: dict, json_name: str) -> str:
    env, suite = data["env"], data["suite"]
    out = [f"# {env['time'][:10]}: ibench `{suite['name']}` against `{env['target']['name']}`", ""]
    out.append(f"Generated by `bench/run run`. Raw data: [{json_name}]({json_name}).")
    if suite.get("description"):
        out += ["", suite["description"]]
    out += ["", "## Environment", "", "| | |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in _env_rows(env)]

    warnings = list(env.get("warnings", []))
    for c in data["cases"]:
        warnings += [f"`{c['case']['name']}`: {w}" for w in c["summary"]["warnings"]]
    if warnings:
        out += ["", "## Warnings", ""] + [f"- {w}" for w in warnings]

    out += _overview(data)
    out += ["", "## Cases"]
    for c in data["cases"]:
        out += _case_section(c)
    return "\n".join(out) + "\n"


def compare(a: dict, b: dict) -> str:
    """Compares two reports: medians, the ratio B/A, and whether the min-max ranges are disjoint."""
    ta, tb = a["env"]["target"]["name"], b["env"]["target"]["name"]
    out = [
        f"A = `{ta}` ({a['env']['time']}), B = `{tb}` ({b['env']['time']})",
        "",
        "| Case | Metric | A | B | B/A | Ranges disjoint |",
        "|---|---|---:|---:|---:|---|",
    ]
    cases_b = {c["case"]["name"]: c for c in b["cases"]}
    for ca in a["cases"]:
        cb = cases_b.get(ca["case"]["name"])
        if not cb:
            continue
        for k, (sa, sb) in enumerate(zip(ca["summary"]["steps"], cb["summary"]["steps"], strict=False)):
            name = ca["case"]["name"] + (f" #{k + 1}" if len(ca["summary"]["steps"]) > 1 else "")
            for key, label in LABELS.items():
                va, vb = sa["metrics"].get(key), sb["metrics"].get(key)
                if not va or not vb:
                    continue
                ratio = vb["median"] / va["median"] if va["median"] else None
                apart = va["max"] < vb["min"] or vb["max"] < va["min"]
                out.append(
                    f"| {name} | {label} | {num(va['median'])} | {num(vb['median'])} | "
                    f"{num(ratio)} | {'yes' if apart else 'no'} |"
                )
    return "\n".join(out) + "\n"
