"""Count tables, predictors and the evaluation of the co-activation replay. Everything works on token rows of a
coact_trace.Trace; a predictor returns a score per expert (n, 256) for the target layer, the evaluation ranks them.
"""

from dataclasses import dataclass

import numpy as np
from coact_trace import N_EXPERT, N_LAYER, N_USED, Trace

HOT_SLOTS = 85  # VRAM slots per layer on the 16 GB card (docs/architecture/expert-tiers.md)
KS = (4, 6, 8, 10, 12, 16, 20, 24, 32, 48)
MAX_D = 24  # the tables reach 24 layers ahead (20 asked for, 4 for multi-source chains)
FIELDS = 13  # hit, pred, truth per token; the same cold; the same per pass union, all and cold; rows


@dataclass
class Counts:
    """Counting set: pair[(l, d)] = X_l^T X_{l+d} (256, 256), within[l] = X_l^T X_l, freq[l] uses per expert."""

    pair: dict[tuple[int, int], np.ndarray]
    within: list[np.ndarray]
    freq: np.ndarray  # (L, 256)
    n: int

    def hot(self, layer: int) -> np.ndarray:
        """Boolean mask of the HOT_SLOTS most used experts of a layer."""
        mask = np.zeros(N_EXPERT, bool)
        mask[np.argsort(-self.freq[layer], kind="stable")[:HOT_SLOTS]] = True
        return mask


def count(trace: Trace, rows: np.ndarray, max_d: int = MAX_D) -> Counts:
    ind = [trace.indicator(layer)[rows] for layer in range(N_LAYER)]
    pair = {(layer, d): ind[layer].T @ ind[layer + d] for d in range(1, max_d + 1) for layer in range(N_LAYER - d)}
    return Counts(pair, [x.T @ x for x in ind], np.stack([x.sum(0) for x in ind]), len(rows))


def conditional(c: np.ndarray, uses: np.ndarray) -> np.ndarray:
    """P(Y | X) from a pair count matrix and the uses of X."""
    return c / np.maximum(uses, 1.0)[:, None]


def tiebreak(counts: Counts, layer: int) -> np.ndarray:
    """A tiny score by frequency so that equal scores (zeros) rank by use, not by expert number."""
    return 1e-6 * counts.freq[layer] / max(counts.n, 1)


def marginal(counts: Counts, layer: int, n: int) -> np.ndarray:
    return np.tile(counts.freq[layer] / max(counts.n, 1), (n, 1))


def lookahead(trace: Trace, rows: np.ndarray, layer: int, counts: Counts) -> np.ndarray:
    """The server's prediction for layer + 1 made at `layer`: rank score 64 - rank for the top-32, the rest by use."""
    s = np.zeros((len(rows), N_EXPERT), np.float32)
    np.put_along_axis(s, trace.look[rows, layer].astype(np.int64), 64.0 - np.arange(32, dtype=np.float32), axis=1)
    return s + tiebreak(counts, layer + 1)


def chain(src: np.ndarray, counts: Counts, layer: int, d: int) -> np.ndarray:
    """Mean over the token's routed experts X at `layer` of P(Y at layer + d | X)."""
    cond = conditional(counts.pair[(layer, d)], counts.freq[layer])
    return src @ cond / N_USED + tiebreak(counts, layer + d)


def multi_source(srcs: list[np.ndarray], counts: Counts, layer: int, d: int, weights: tuple[float, ...]) -> np.ndarray:
    """Path evidence: chains from the layers layer, layer - 1, ... to the same target, weighted (srcs[j] is the set j
    layers before `layer`)."""
    total = np.zeros_like(srcs[0])
    for j, w in enumerate(weights):
        if layer - j < 0:
            break
        total += w * (srcs[j] @ conditional(counts.pair[(layer - j, d + j)], counts.freq[layer - j])) / N_USED
    return total + tiebreak(counts, layer + d)


def partners(seed_scores: np.ndarray, counts: Counts, target: int, m: int) -> np.ndarray:
    """Completion score inside the target layer: the sum of P(Y | S) over the m best seeds S of `seed_scores`."""
    seeds = np.argsort(-seed_scores, axis=1, kind="stable")[:, :m]
    ind = np.zeros_like(seed_scores)
    np.put_along_axis(ind, seeds, 1.0, axis=1)
    return ind @ conditional(counts.within[target], counts.freq[target]) / m


def sparse(cond: np.ndarray, top_n: int) -> np.ndarray:
    """The table kept as the top_n partners per row (the rest zero), as a shipped file would hold it."""
    out = np.zeros_like(cond)
    idx = np.argsort(-cond, axis=1, kind="stable")[:, :top_n]
    np.put_along_axis(out, idx, np.take_along_axis(cond, idx, axis=1), axis=1)
    return out


def _ranks(score: np.ndarray) -> np.ndarray:
    order = np.argsort(-score, axis=1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(N_EXPERT), order.shape), axis=1)
    return ranks


def evaluate(score, truth, hot, bounds, ks=KS) -> dict[tuple[str, int], np.ndarray]:
    """Counts for the top-K of `score` against `truth` (n, 256) over all rows. Mode "all": the top-K over all
    experts, the cold part counted separately (the Mac's recall of routed non-resident experts). Mode "cold": the
    top-K among the cold experts only. `bounds` are the first rows of the passes (for the union numbers)."""
    out = {}
    for mode, sc in (("all", score), ("cold", np.where(hot, -1e9, score))):
        ranks = _ranks(sc)
        for k in ks:
            out[(mode, k)] = set_counts((ranks < k).astype(np.float32), truth, hot, bounds)
    return out


def set_counts(pred, truth, hot, bounds) -> np.ndarray:
    """FIELDS numbers for one predicted 0/1 set per row: token level (all, cold), pass-union level (all, cold), rows."""
    cold = pred * (~hot)
    cold_truth = truth * (~hot)
    up, ut = np.maximum.reduceat(pred, bounds, axis=0), np.maximum.reduceat(truth, bounds, axis=0)
    uc, uct = up * (~hot), ut * (~hot)
    return np.array(
        [
            (pred * truth).sum(), pred.sum(), truth.sum(),
            (cold * truth).sum(), cold.sum(), cold_truth.sum(),
            (up * ut).sum(), up.sum(), ut.sum(),
            (uc * ut).sum(), uc.sum(), uct.sum(),
            len(truth),
        ]
    )  # fmt: skip


def rrf(*scores: np.ndarray, c: float = 10.0) -> np.ndarray:
    """Reciprocal rank fusion of several scores (no weights, nothing fitted)."""
    return sum(1.0 / (c + _ranks(s)) for s in scores)


class Acc:
    """Sums of the evaluation counts per (name, d, mode, K) over layers; the ratios are taken at the end."""

    def __init__(self):
        self.sums: dict[tuple, np.ndarray] = {}

    def add(self, name: str, d: int, res: dict) -> None:
        for (mode, k), v in res.items():
            key = (name, d, mode, k)
            self.sums[key] = self.sums.get(key, 0) + v

    def row(self, name: str, d: int, mode: str, k: int) -> dict[str, float]:
        v = self.sums[(name, d, mode, k)]
        r = lambda a, b: float(a / b) if b else float("nan")  # noqa: E731
        return {
            "recall": r(v[0], v[2]), "precision": r(v[0], v[1]),
            "cold_recall": r(v[3], v[5]), "cold_precision": r(v[3], v[4]),
            "union_recall": r(v[6], v[8]), "union_precision": r(v[6], v[7]),
            "union_cold_recall": r(v[9], v[11]), "union_cold_precision": r(v[9], v[10]),
            "set_size": r(v[1], v[12]),
        }  # fmt: skip


def pass_bounds(trace: Trace, rows: np.ndarray) -> np.ndarray:
    """First index (within `rows`) of each pass; the split is by request, so passes stay whole."""
    p = trace.pass_of[rows]
    return np.concatenate([[0], np.flatnonzero(np.diff(p)) + 1])
