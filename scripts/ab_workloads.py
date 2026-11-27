"""The named workloads of scripts/ab.py (docs/contribute/measurement.md, "Workloads") and the jobs they turn into.

A job is one server start (one `--ctx`, some environment) with the requests that run in it. Workloads that share
`--ctx` and environment share the start, in the fixed order short, mid, long, so that `long` always follows `mid` on
the same server as in the runs behind bench/noise.toml.
"""

from dataclasses import dataclass, field

# kind "sum": summarize the code from the start of the corpus; "write": write code from a slice further in
SUM, WRITE = "sum", "write"
WRITE_OFFSET = 1_000_000  # characters into the corpus: other text than the "sum" prompt, so no shared prefix
# kinds of the prose corpus (scripts/make_prose_corpus.py, the knowledge workload of Bolt): "prose" answers in prose
# about the start of it, "longdoc" about a long text that starts further in (no shared prefix with "prose")
PROSE, LONGDOC = "prose", "longdoc"
PROSE_KINDS = (PROSE, LONGDOC)
LONGDOC_OFFSET = 90_000  # characters into the prose corpus: more than the longest "prose" prompt (16000 tokens)


@dataclass(frozen=True)
class Request:
    kind: str
    tokens: int  # prompt
    decode: int

    @property
    def name(self) -> str:
        """Part of the metric name: 2000 for a summary, 2000:write for the write request, 2000:prose."""
        return str(self.tokens) + ("" if self.kind == SUM else f":{self.kind}")


@dataclass(frozen=True)
class Spec:
    requests: tuple = ()
    ctxs: tuple = (140000,)
    env: dict = field(default_factory=dict)
    bench_set: bool = False


SHORT = (Request(SUM, 2000, 400), Request(WRITE, 2000, 400))
NAMED = {
    "short": Spec(SHORT),
    "ctx": Spec(SHORT, (140000, 262144)),
    "mid": Spec((Request(SUM, 30000, 200),)),
    "long": Spec((Request(SUM, 130000, 200),)),
    "set": Spec(ctxs=(32768,), bench_set=True),
    "prose": Spec((Request(PROSE, 2000, 400), Request(PROSE, 16000, 400)), (32768,)),
    "longdoc": Spec((Request(LONGDOC, 65536, 200),), (98304,)),
    "huge": Spec((Request(SUM, 512000, 200), Request(SUM, 1000000, 200)), (1048576,), {"TONA_SEQS": "1"}),
}
ORDER = ("short", "ctx", "mid", "long", "prose", "longdoc", "set", "huge")
CUSTOM = ("server", "bench-set")  # explicit --targets / --ctx


@dataclass
class Job:
    ctx: int
    env: dict
    requests: list = field(default_factory=list)
    bench_set: bool = False


def spec_of(name: str, targets: list, ctxs: list) -> Spec:
    """targets: [(prompt tokens, decode tokens)] and ctxs from --targets / --ctx, for the custom workloads."""
    if name in NAMED:
        return NAMED[name]
    if name == "server":
        return Spec(tuple(Request(SUM, t, n) for t, n in targets), tuple(ctxs))
    return Spec(ctxs=tuple(ctxs), bench_set=True)


def jobs_for(names: list, targets: list, ctxs: list) -> list:
    """Jobs in run order; a request that two workloads share (`short` inside `ctx`) runs once per start."""
    unknown = [n for n in names if n not in NAMED and n not in CUSTOM]
    if unknown:
        raise SystemExit(f"unknown workload {unknown}; choose from {', '.join(ORDER + CUSTOM)}")
    jobs: dict = {}
    for name in sorted(set(names), key=(ORDER + CUSTOM).index):
        spec = spec_of(name, targets, ctxs)
        for ctx in spec.ctxs:
            job = jobs.setdefault((ctx, tuple(sorted(spec.env.items()))), Job(ctx, dict(spec.env)))
            job.requests += [r for r in spec.requests if r not in job.requests]
            job.bench_set |= spec.bench_set
    return list(jobs.values())


def suffixes(jobs: list) -> dict:
    """{base name: True} for every request or set that runs under more than one --ctx (its metrics get /ctx<N>)."""
    bases: dict = {}
    for job in jobs:
        for base in [r.name for r in job.requests] + (["set"] if job.bench_set else []):
            bases.setdefault(base, set()).add(job.ctx)
    return {base: len(ctxs) > 1 for base, ctxs in bases.items()}


def metric_names(jobs: list) -> list:
    """All metric names the run reports, in the order of the jobs."""
    several = suffixes(jobs)
    names = []
    for job in jobs:
        for base in [r.name for r in job.requests] + (["set"] if job.bench_set else []):
            suffix = f"/ctx{job.ctx}" if several[base] else ""
            if base == "set":
                names += [f"decode-code{suffix}", f"decode-text{suffix}"]
            else:
                names += [f"prefill@{base}{suffix}", f"decode@{base}{suffix}"]
    return names


def tokens_by_target(jobs: list) -> dict:
    return {r.tokens: r.decode for job in jobs for r in job.requests}


def check_ctx(jobs: list) -> None:
    for job in jobs:
        need = max((r.tokens + r.decode for r in job.requests), default=0)
        if job.ctx <= need:
            raise SystemExit(f"--ctx {job.ctx} must exceed the longest prompt plus its decode tokens ({need})")


def describe(jobs: list) -> str:
    """One line for the plan: '2000:400 2000:write:400 at ctx 140000; set at ctx 32768'."""
    parts = []
    for job in jobs:
        items = [f"{r.name}:{r.decode}" for r in job.requests] + (["bench-set"] if job.bench_set else [])
        env = "".join(f", {k}={v}" for k, v in job.env.items())
        parts.append(f"{' '.join(items)} at --ctx {job.ctx}{env}")
    return "; ".join(parts)
