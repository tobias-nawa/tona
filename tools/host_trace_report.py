#!/usr/bin/python3 -I
"""Per-segment means of a TONA_HOST_TRACE file (engine/backend/host_trace.h).

host_trace_report.py TRACE [--skip N]

A decode cycle runs from `cycle_begin` to `cycle_end`. Every interval between two consecutive marks is named
"<from> -> <to>"; the table gives, per cycle, the mean time and the count of such intervals, so the rows sum to the
cycle. Router intervals (`selected -> router_arrived` is the GPU computing up to the next router plus the mailbox
round trip) are summed over the 48 layers in one row. --skip N drops the first N cycles (warm-up).
"""

import argparse
import itertools
import statistics
from collections import defaultdict


def read_cycles(path: str):
    cycles, cur = [], None
    with open(path) as f:
        for line in f:
            ns, what, arg = line.split()
            if what == "cycle_begin":
                cur = []
            if cur is not None:
                cur.append((int(ns), what, int(arg)))
            if what == "cycle_end" and cur:
                cycles.append(cur)
                cur = None
    return cycles


def segments(cycle):
    rows = defaultdict(lambda: [0.0, 0])
    for (t0, a, _), (t1, b, _) in itertools.pairwise(cycle):
        r = rows[f"{a} -> {b}"]
        r[0] += (t1 - t0) / 1e6
        r[1] += 1
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--skip", type=int, default=5)
    args = ap.parse_args()
    cycles = read_cycles(args.trace)[args.skip :]
    if not cycles:
        raise SystemExit("no complete cycle in the trace")
    totals = [(c[-1][0] - c[0][0]) / 1e6 for c in cycles]
    acc = defaultdict(list)
    count = defaultdict(int)
    for c in cycles:
        seen = segments(c)
        for k, (ms, n) in seen.items():
            acc[k].append(ms)
            count[k] += n
    n_cyc = len(cycles)
    print(f"{n_cyc} cycles, mean {statistics.mean(totals):.3f} ms (sd {statistics.pstdev(totals):.3f})")
    print(f"{'segment':44s} {'ms/cycle':>9s} {'per cycle':>9s} {'us each':>9s}")
    rows = sorted(((sum(v) / n_cyc, k) for k, v in acc.items()), reverse=True)
    for ms, k in rows:
        n = count[k] / n_cyc
        print(f"{k:44s} {ms:9.3f} {n:9.2f} {1000 * ms / max(n, 1e-9):9.1f}")
    print(f"{'sum':44s} {sum(ms for ms, _ in rows):9.3f}")


if __name__ == "__main__":
    main()
