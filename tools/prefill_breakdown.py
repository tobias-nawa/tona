#!/usr/bin/env python3
"""Prefill time per kernel family from a rocprofv3 rocpd database, against the ceiling of each family.

Usage: tools/prefill_breakdown.py TRACE.db --tokens 1933,29913,129769 [--gap 1.5]

The trace holds several requests of one server start. A request is a run of kernels with no gap longer than
`--gap` seconds around it; `--tokens` lists the prompt tokens of the last runs in order (the runs before them,
the model load, are skipped). Per family the tool prints time, share of the window, calls, useful work, achieved
rate, the time at the ceiling and the gap. Ceiling = FLOP / measured WMMA peak (docs/backends/reference-platform.md: f16
93, fp8 186 TFLOPS). Bandwidth-bound families have no ceiling column (estimated by hand in the measurement file).
The work per token comes from the model's tensor list (see WEIGHTS below); it is the useful work, padding of
partial tiles is not counted, so a low rate includes that waste.
"""

import argparse
import re
import sqlite3
import sys
from collections import defaultdict

PEAK_F16, PEAK_FP8 = 93e12, 186e12  # docs/backends/reference-platform.md, measured

# Weights (values) per token that a family multiplies, from `tools/gguf_info.py --tensors` of the Coder model: 36
# DeltaNet and 12 attention layers, d 2560, 10 of 256 routed experts of 3 x 2560 x 640. FLOP = 2 x weights x tokens.
WEIGHTS = {
    # fp8 route (rows >= 512): attn_qkv + attn_gate (36 x 41.9 M), attn q, k, v (12 x 34.1 M),
    # shared expert (48 x 4.9 M)
    "dense fp8 GEMM": (36 * 41.9e6 + 12 * 34.1e6 + 48 * 4.9e6, PEAK_FP8),
    # f16 route: ssm_out, attn_output (48 x 15.7 M), hc down (48 x 2 x 3.3 M), router 31 M, indexer 19 M,
    # alpha/beta 9 M
    "dense f16 GEMM": (36 * 15.7e6 + 12 * 15.7e6 + 48 * 2 * 3.3e6 + 31e6 + 19e6 + 9e6, PEAK_F16),
    "hc up GEMM + mix": (48 * 2 * 3.3e6, PEAK_F16),
    "MoE expert GEMM": (48 * 10 * 3 * 2560 * 640, PEAK_F16),
}
FAMILIES = [
    ("dense fp8 GEMM", r"k_gemm8b"),
    ("fp8 weight/input conversion", r"k_dequant_fp8|k_to_fp8"),
    ("dense f16 GEMM", r"k_gemm2<\d+, false>|k_gemm<"),
    ("hc up GEMM + mix", r"k_gemm2<\d+, true>"),
    ("f16 weight unpack", r"k_dequant<"),
    ("MoE expert GEMM", r"k_gemm_q2"),
    ("MoE expert reformat", r"k_q8_to_q2"),
    ("attention (QSA)", r"k_attn_wmma"),
    ("indexer score + top-k", r"k_idx_score|k_topk"),
    ("QSA prep (norm, rope, cache)", r"k_qsa_prep|k_ring_store"),
    ("DeltaNet recurrence", r"k_dn_wave|k_dn_recur"),
    ("DeltaNet conv, norm", r"k_dn_conv|k_dn_norm"),
    ("residual combine, norm", r"k_combine|k_norm16|k_hc_mean|k_mix|k_act16|k_to16|k_bcast"),
    ("MoE sum, swiglu, scatter, route", r"k_moe_sum|k_swiglu16|k_scatter|k_route|k_moe_check"),
    ("PLE, MTP, other", r"k_ple|k_mtp|k_fill|k_mega|rocclr"),
]


def family_of(name: str) -> str:
    for fam, pat in FAMILIES:
        if re.search(pat, name):
            return fam
    return "unclassified: " + re.sub(r"\(.*", "", name)[:60]


def runs(kernels: list, gap_ns: float) -> list:
    """Kernel index ranges separated by gaps longer than gap_ns."""
    out, start = [], 0
    for i in range(1, len(kernels)):
        if kernels[i][0] - kernels[i - 1][1] > gap_ns:
            out.append((start, i))
            start = i
    out.append((start, len(kernels)))
    return out


def busy_union(spans: list) -> float:
    """Time covered by at least one kernel (spans sorted by start), in ns."""
    total, end = 0.0, None
    for s, e in spans:
        if end is None or s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total


def report(kernels: list, copies: list, tokens: int) -> list:
    window = kernels[-1][1] - kernels[0][0]
    busy = busy_union([(k[0], k[1]) for k in kernels])
    fam_ns, fam_calls = defaultdict(float), defaultdict(int)
    for s, e, name in kernels:
        f = family_of(name)
        fam_ns[f] += e - s
        fam_calls[f] += 1
    lines = [
        f"window {window / 1e9:.2f} s, {tokens} tokens: {tokens / (window / 1e9):.0f} tok/s inside the trace",
        f"kernels {len(kernels)}, busy (union) {busy / 1e9:.2f} s, idle between kernels {(window - busy) / 1e9:.2f} s "
        f"({100 * (window - busy) / window:.1f} %), sum of kernel times {sum(fam_ns.values()) / 1e9:.2f} s",
        f"copies {len(copies)}, {sum(c[2] for c in copies) / 1e9:.1f} GB, "
        f"copy time {sum(c[1] - c[0] for c in copies) / 1e9:.2f} s",
        "",
        "| family | ms | % window | calls | useful TFLOP | TFLOPS | ceiling ms | gap ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for f, ns in sorted(fam_ns.items(), key=lambda kv: -kv[1]):
        row = f"| {f} | {ns / 1e6:.0f} | {100 * ns / window:.1f} | {fam_calls[f]} | "
        if f in WEIGHTS:
            w, peak = WEIGHTS[f]
            flop = 2 * w * tokens
            ceil_ms = flop / peak * 1e3
            row += f"{flop / 1e12:.0f} | {flop / (ns / 1e9) / 1e12:.0f} | {ceil_ms:.0f} | {ns / 1e6 - ceil_ms:.0f} |"
        else:
            row += "| | | |"
        lines.append(row)
    lines.append(f"| idle gaps | {(window - busy) / 1e6:.0f} | {100 * (window - busy) / window:.1f} | | | | | |")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("db")
    ap.add_argument("--tokens", required=True, help="prompt tokens of the runs after the load, comma separated")
    ap.add_argument("--gap", type=float, default=1.5, help="seconds of idle that separate two requests")
    a = ap.parse_args()
    tokens = [int(t) for t in a.tokens.split(",")]
    con = sqlite3.connect(a.db)
    kernels = con.execute("select start, end, name from kernels order by start").fetchall()
    copies = con.execute("select start, end, size from memory_copies order by start").fetchall()
    spans = runs(kernels, a.gap * 1e9)
    if len(spans) <= len(tokens):
        sys.exit(f"{len(spans)} runs found (the first is the load), {len(tokens)} token counts given")
    for (lo, hi), n in zip(spans[-len(tokens) :], tokens, strict=True):
        ks = kernels[lo:hi]
        cs = [c for c in copies if ks[0][0] <= c[0] <= ks[-1][1]]
        print(f"## {n} tokens\n")
        print("\n".join(report(ks, cs, n)), "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
