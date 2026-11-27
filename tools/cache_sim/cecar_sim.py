"""Offline comparison of Tona's expert cache policies with the reuse-distance predictor and the cache-aware routing of
CECAR (arXiv 2611.20119), on a TONA_ROUTE_TRACE_P file (router logits per token, layer and pass).

Nothing here changes the engine's routing. Part 1 (`policies`): hit rate of LRU, LRFU, a learned reuse predictor and
Belady (oracle) on the original routing, demand only (no preloading, to isolate the eviction policy). Part 2
(`routing`): CECAR's bonus and pruning applied to the recorded logits: hit rate gained and how much the routing
changes (selected experts replaced, router weight mass dropped). Replay caveat: the token stream is the recorded one;
a changed routing would change the later tokens, which this cannot show.

Usage: tools/cache_sim/cecar_sim.py TRACE [--slots 90] [--k 10] [--tau 4 6 8] [--layers 0 12 24 36 47]
"""

import argparse
import sys
from dataclasses import dataclass

import numpy as np

N_EXPERT = 256  # the Coder
HORIZONS = (4.0, 16.0, 64.0)  # half-lives of the frequency features, in passes
LABEL_PASSES = 8  # "reused soon": the expert is used again within this many passes


def read_router(path: str, layers: set[int], n_expert: int = N_EXPERT) -> dict[int, list[np.ndarray]]:
    """Per layer the router logits of each pass as an array [tokens, experts] (the prediction half is skipped)."""
    row = n_expert * 4
    out: dict[int, list[np.ndarray]] = {ly: [] for ly in layers}
    with open(path, "rb") as f:
        while head := f.read(2):
            layer, nb = head[0], head[1]
            body = f.read(2 * nb * row)
            if len(body) != 2 * nb * row:
                break
            if layer in layers:
                out[layer].append(np.frombuffer(body[: nb * row], dtype="<f4").reshape(nb, n_expert))
    return out


def topk_sets(logits: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k largest logits per row, best first."""
    idx = np.argpartition(-logits, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(logits, idx, axis=1), axis=1, kind="stable")
    return np.take_along_axis(idx, order, axis=1)


@dataclass
class Features:
    """Recency and decayed frequency of every expert of one layer, advanced once per pass."""

    last: np.ndarray
    freq: np.ndarray  # [len(HORIZONS), experts]
    count: np.ndarray  # uses so far, for the paper's f / max f
    now: int = 0

    @classmethod
    def new(cls, n: int = N_EXPERT) -> "Features":
        return cls(np.full(n, -1000.0), np.zeros((len(HORIZONS), n)), np.zeros(n))

    def vector(self) -> np.ndarray:
        """Feature matrix [experts, 3 + len(HORIZONS)] at the current pass: log recency, log decayed frequencies, then
        the two inputs of the paper (1 / recency, uses / largest use count)."""
        dt = self.now - self.last
        decay = np.array([2.0 ** (-dt / h) for h in HORIZONS])
        paper = [1.0 / (dt + 1.0), self.count / max(self.count.max(), 1.0)]
        return np.column_stack([np.log1p(dt), np.log1p(self.freq * decay).T, *paper])

    def use(self, used: np.ndarray) -> None:
        for i, h in enumerate(HORIZONS):
            self.freq[i][used] = self.freq[i][used] * 2.0 ** (-(self.now - self.last[used]) / h) + 1.0
        self.last[used] = self.now
        self.count[used] += 1
        self.now += 1


def expand(x: np.ndarray) -> np.ndarray:
    """Linear and pairwise product terms, plus a bias: a logistic regression on these is the 'predictor'."""
    n = x.shape[1]
    cols = [np.ones(len(x))] + [x[:, i] for i in range(n)] + [x[:, i] * x[:, j] for i in range(n) for j in range(i, n)]
    return np.column_stack(cols)


def fit_logistic(x: np.ndarray, y: np.ndarray, steps: int = 300) -> np.ndarray:
    """Newton-free batch gradient descent with feature standardisation folded into the returned weights."""
    mu, sd = x[:, 1:].mean(axis=0), x[:, 1:].std(axis=0) + 1e-6
    z = np.column_stack([x[:, :1], (x[:, 1:] - mu) / sd])
    w = np.zeros(z.shape[1])
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-np.clip(z @ w, -30, 30)))
        w -= 1.5 * (z.T @ (p - y)) / len(y)
    w[1:] /= sd
    w[0] -= float(w[1:] @ mu)
    return w


def used_matrix(passes: list[np.ndarray], k: int) -> np.ndarray:
    """Bool [passes, experts]: the union of the top-k of the tokens of each pass (the original routing)."""
    u = np.zeros((len(passes), N_EXPERT), bool)
    for t, lg in enumerate(passes):
        u[t, topk_sets(lg, k).ravel()] = True
    return u


def next_table(used: np.ndarray) -> np.ndarray:
    """[t, e]: the first pass after t that uses e (1e9: never again)."""
    nxt = np.full(used.shape, 1e9)
    seen = np.full(N_EXPERT, 1e9)
    for t in range(len(used) - 1, -1, -1):
        nxt[t] = seen
        seen = np.where(used[t], t, seen)
    return nxt


def training_set(used: np.ndarray, stride: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Features at every stride-th pass, label: used again within LABEL_PASSES passes."""
    ft, xs, ys = Features.new(), [], []
    nxt = next_table(used)
    for t in range(len(used)):
        if t % stride == 0 and t > 20:
            xs.append(expand(ft.vector()))
            ys.append((nxt[t] - t <= LABEL_PASSES).astype(float))
        ft.use(np.flatnonzero(used[t]))
    return np.concatenate(xs), np.concatenate(ys)


class Layer:
    """One layer's slot cache. policy: lru | lrfu | learned | belady. score(e) high = keep."""

    def __init__(self, slots: int, policy: str, weights: np.ndarray | None = None, used: np.ndarray | None = None):
        self.slots, self.policy, self.w = slots, policy, weights
        self.nxt = next_table(used) if used is not None else None
        self.res = np.zeros(N_EXPERT, bool)
        self.ft = Features.new()
        self.t = 0

    def keep_score(self) -> np.ndarray:
        x = self.ft.vector()
        if self.policy == "lru":
            return -x[:, 0]
        if self.policy == "lrfu":
            return x[:, 2]  # decayed frequency, half-life 16 passes
        if self.policy == "learned":
            return expand(x) @ self.w
        return -self.next_use()

    def next_use(self) -> np.ndarray:
        return self.nxt[self.t]

    def admit(self, missed: np.ndarray, protect: np.ndarray) -> None:
        """Misses enter; when full the lowest keep score among the unprotected leaves (Belady: bypass if worse)."""
        keep = self.keep_score()
        for e in missed:
            if self.res.sum() >= self.slots:
                sc = np.where(self.res & ~protect, keep, np.inf)
                v = int(sc.argmin())
                if not np.isfinite(sc[v]) or (self.policy == "belady" and keep[e] < sc[v]):
                    continue
                self.res[v] = False
            self.res[e] = True

    def end_pass(self, used: np.ndarray) -> None:
        self.ft.use(used)
        self.t += 1


def run_policy(passes: list[np.ndarray], slots: int, policy: str, k: int, w=None) -> tuple[float, float]:
    """Hit rate over the unions of the original routing, counted from 20 % of the run, and misses per layer pass."""
    used = used_matrix(passes, k)
    layer = Layer(slots, policy, w, used)
    first, hits, total = int(len(passes) * 0.2), 0, 0
    for t in range(len(passes)):
        u = np.flatnonzero(used[t])
        if t >= first:
            hits += int(layer.res[u].sum())
            total += len(u)
        layer.admit(u[~layer.res[u]], used[t])
        layer.end_pass(u)
    return hits / max(total, 1), (total - hits) / max(len(passes) - first, 1)


@dataclass
class RouteStats:
    hits: int = 0
    total: int = 0
    replaced: int = 0  # original top-k experts not in the final selection (of the token)
    selections: int = 0
    mass_lost: float = 0.0  # original top-k softmax weight of those
    tokens: int = 0
    passes: int = 0

    def line(self, name: str) -> str:
        n = max(self.selections, 1)
        return (
            f"{name:28s} hit {100 * self.hits / max(self.total, 1):6.2f} %  replaced {100 * self.replaced / n:5.2f} %  "
            f"weight mass lost {100 * self.mass_lost / max(self.tokens, 1):5.2f} %  "
            f"misses/pass {(self.total - self.hits) / max(self.passes, 1):5.2f}"
        )


@dataclass
class Route:
    k: int
    tau: int  # ranks 0..tau stay even when not cached (CECAR's j <= tau, with 1-based j in the paper)
    margin: int  # the m of the margin z[1] - z[m]
    bonus: bool


def select(lg: np.ndarray, resident: np.ndarray, pr: np.ndarray, r: Route) -> np.ndarray:
    """CECAR selection for one token: bonus on resident experts, top-k on the adjusted logits, prune non-resident
    experts below rank tau. pr: reuse probability of each expert from the predictor (bonus scale)."""
    z = lg.copy()
    if r.bonus:
        s = np.sort(lg)[::-1]
        delta = s[0] - s[r.margin - 1]
        p = pr[resident]
        d = (pr - p.min()) / max(p.max() - p.min(), 1e-9) if p.size else np.zeros(N_EXPERT)
        z = z + np.where(resident, delta * np.clip(d, 0.0, 1.0), 0.0)
    cand = np.argsort(-z, kind="stable")[: r.k]
    keep = [e for j, e in enumerate(cand) if resident[e] or j < r.tau]
    return np.array(keep, dtype=int)


def run_routing(passes: list[np.ndarray], slots: int, r: Route, w: np.ndarray) -> RouteStats:
    layer = Layer(slots, "learned", w)
    st, first = RouteStats(), int(len(passes) * 0.2)
    for t, lg in enumerate(passes):
        pr = 1.0 / (1.0 + np.exp(-np.clip(expand(layer.ft.vector()) @ w, -30, 30)))
        orig = topk_sets(lg, r.k)
        union: set[int] = set()
        for i in range(len(lg)):
            sel = select(lg[i], layer.res, pr, r)
            union.update(sel.tolist())
            if t >= first:
                e = np.exp(lg[i] - lg[i].max())
                e /= e[orig[i]].sum()
                lost = np.setdiff1d(orig[i], sel)
                st.replaced += len(lost)
                st.selections += r.k
                st.mass_lost += float(e[lost].sum())
                st.tokens += 1
        u = np.array(sorted(union), dtype=int)
        if t >= first:
            st.hits += int(layer.res[u].sum())
            st.total += len(u)
            st.passes += 1
        protect = np.zeros(N_EXPERT, bool)
        protect[u] = True
        layer.admit(u[~layer.res[u]], protect)
        layer.end_pass(u)
    return st


DEFAULT_CONFIGS = ["prune:0:4", "prune:0:6", "prune:0:8"] + [
    f"bonus:{m}:{t}" for m in (2, 3, 5, 10) for t in (4, 6, 10)
]


def parse_config(spec: str, k: int) -> Route:
    """Spec: prune:_:TAU (pruning only) or bonus:M:TAU (CECAR bonus with margin index M, then pruning)."""
    kind, m, tau = spec.split(":")
    return Route(k, int(tau), int(m) or k, kind == "bonus")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trace")
    ap.add_argument("--slots", type=int, default=90)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--config", nargs="+", default=DEFAULT_CONFIGS, help="prune:0:TAU or bonus:M:TAU")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 12, 24, 36, 47])
    ap.add_argument("--train", help="fit the predictor on all passes of this trace, test on all of TRACE")
    args = ap.parse_args(argv)
    data = read_router(args.trace, set(args.layers))
    if args.train:
        fit = read_router(args.train, set(args.layers))
        halves = {ly: (fit[ly], p) for ly, p in data.items()}
    else:
        # fit on the first half of the passes of all chosen layers, test on the second half (sessions are not marked)
        halves = {ly: (p[: len(p) // 2], p[len(p) // 2 :]) for ly, p in data.items()}
    xs, ys = zip(*(training_set(used_matrix(h[0], args.k)) for h in halves.values()), strict=True)
    w = fit_logistic(np.concatenate(xs), np.concatenate(ys))
    print(f"passes per layer {len(data[args.layers[0]])}, slots {args.slots}, k {args.k}, test = second half")
    names = ["lru", "lrfu", "learned", "belady"]
    rows = {n: [] for n in names}
    rows.update({c: [] for c in args.config})
    for ly, (_, test) in halves.items():
        for n in names:
            rows[n].append(run_policy(test, args.slots, n, args.k, w))
        for c in args.config:
            rows[c].append(run_routing(test, args.slots, parse_config(c, args.k), w))
        print(f"layer {ly} done", file=sys.stderr)
    for n in names:
        hit, miss = np.mean(rows[n], axis=0)
        print(f"{n:28s} hit {100 * hit:6.2f} %  (exact routing)  misses/pass {miss:5.2f}")
    for c in args.config:
        tot = RouteStats(*(sum(getattr(s, f) for s in rows[c]) for f in RouteStats.__dataclass_fields__))
        print(tot.line(c))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
