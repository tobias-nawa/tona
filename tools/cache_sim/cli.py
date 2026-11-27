"""python3 tools/cache_sim/cli.py --slots 90 --grid trace1.bin trace2.bin

Prints the hit rate (%) of each policy on each trace. With --grid, ClassLRU settings are added and the tuning is
read crosswise: the best setting on one trace, its number on the other."""

import argparse
import itertools
import multiprocessing

from route_trace import load_passes, set_experts
from sim import hit_rate, make_factory

BASE = ["lru", "lrfu:3", "lrfu:10", "lrfu:30"]
CLASSES = (4, 8)
HALF_LIVES = (16, 32, 64, 128, 256, 512)


def grid_specs() -> list[str]:
    specs = [f"lru-admit:{hl}" for hl in HALF_LIVES]
    for kind, n, hl in itertools.product(("class", "class-admit"), CLASSES, HALF_LIVES):
        specs.append(f"{kind}:{n}:{hl}")
    return specs


_STATE: dict = {}


def _run(job):
    trace, spec = job
    passes, slots = _STATE["passes"][trace], _STATE["slots"]
    return trace, spec, hit_rate(passes, make_factory(spec, slots), server_fill=_STATE["server_fill"]) * 100


def crosswise(results: dict, traces: list[str], specs: list[str]) -> list[str]:
    """For each trace the best of `specs` on it and that setting's number on every other trace."""
    lines = []
    for tune in traces:
        best = max(specs, key=lambda s: results[tune, s])
        others = ", ".join(f"{t}: {results[t, best]:.2f}" for t in traces if t != tune)
        lines.append(f"tuned on {tune}: {best} {results[tune, best]:.2f} -> held out {others}")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--slots", type=int, default=90)
    ap.add_argument("--policy", action="append", default=[], help="policy spec, see sim.make_factory")
    ap.add_argument("--grid", action="store_true", help="add the reference policies and the ClassLRU grid")
    ap.add_argument("--server-fill", action="store_true", help="misses enter only while slots are free (TONA_FILL=0)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument(
        "--experts", type=int, default=256, help="experts per layer of the traced model (rows of the trace)"
    )
    args = ap.parse_args(argv)
    specs = args.policy + (BASE + grid_specs() if args.grid else [])
    _STATE["slots"] = args.slots
    _STATE["server_fill"] = args.server_fill
    set_experts(args.experts)
    _STATE["passes"] = {t: load_passes(t) for t in args.traces}
    jobs = list(itertools.product(args.traces, specs))
    with multiprocessing.get_context("fork").Pool(args.jobs) as pool:
        results = {(t, s): r for t, s, r in pool.map(_run, jobs, chunksize=1)}
    for s in specs:
        print(f"{s:20s}", "  ".join(f"{results[t, s]:6.2f}" for t in args.traces))
    if args.grid:
        print("\n".join(crosswise(results, args.traces, [s for s in specs if s.startswith("class")])))
        print("\n".join(crosswise(results, args.traces, [s for s in specs if s.startswith("lru-admit")])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
