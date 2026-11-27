"""Predictors of the co-activation replay beyond the plain count chain, each adapted from another field
(docs in research/55-expert-coactivation.md). All return a score per expert of the target layer, (n, 256).
"""

import numpy as np
from coact_tables import Counts, chain, conditional, tiebreak
from coact_trace import N_EXPERT, N_LAYER, N_USED, Split, Trace

EPS = 0.5  # additive smoothing of the pair counts in PMI


def pmi_matrix(c: np.ndarray, fx: np.ndarray, fy: np.ndarray, n: int, normalize: bool) -> np.ndarray:
    """Market-basket lift in log form: PMI(X, Y) = log P(XY) / (P(X) P(Y)), smoothed; NPMI divides by -log P(XY)."""
    pxy = (c + EPS) / (n + EPS * N_EXPERT * N_EXPERT)
    px, py = (fx + EPS) / (n + EPS * N_EXPERT), (fy + EPS) / (n + EPS * N_EXPERT)
    pmi = np.log(pxy / (px[:, None] * py[None, :]))
    return pmi / -np.log(pxy) if normalize else pmi


def pmi_score(src: np.ndarray, counts: Counts, layer: int, d: int, normalize: bool = False) -> np.ndarray:
    """Naive Bayes in log form: the sum of the PMI of Y with every routed X (vertical)."""
    m = pmi_matrix(counts.pair[(layer, d)], counts.freq[layer], counts.freq[layer + d], counts.n, normalize)
    return src @ m / N_USED


def cosine_score(src: np.ndarray, counts: Counts, layer: int, d: int) -> np.ndarray:
    """Item-item collaborative filtering: cosine similarity of the co-occurrence vectors of expert X (layer) and Y
    (layer + d) over the tokens, summed over the routed X."""
    fx, fy = counts.freq[layer], counts.freq[layer + d]
    cos = counts.pair[(layer, d)] / np.sqrt(np.maximum(fx, 1)[:, None] * np.maximum(fy, 1)[None, :])
    return src @ cos / N_USED + tiebreak(counts, layer + d)


def rule_prediction(src: np.ndarray, counts: Counts, layer: int, d: int, theta: float, support: int = 20):
    """Association rules X -> Y with confidence >= theta and support >= `support` tokens; the predicted set is every Y
    named by a rule whose X the token routed. Returns a 0/1 matrix (n, 256)."""
    c = counts.pair[(layer, d)]
    rules = (conditional(c, counts.freq[layer]) >= theta) & (c >= support)
    return ((src @ rules.astype(np.float32)) > 0).astype(np.float32)


def lift_prune(score: np.ndarray, counts: Counts, target: int, floor: float, seeds: int) -> np.ndarray:
    """Horizontal pruning: experts whose lift with the best `seeds` experts of the score is below `floor` lose their
    score (they almost never come with the seed)."""
    f = counts.freq[target]
    lift = counts.within[target] * counts.n / np.maximum(np.outer(f, f), 1.0)
    best = np.argsort(-score, axis=1, kind="stable")[:, :seeds]
    ind = np.zeros_like(score)
    np.put_along_axis(ind, best, 1.0, axis=1)
    mean_lift = (ind @ lift) / seeds
    return np.where(mean_lift < floor, score * 0.0, score)


def knn_scores(sp: Split, layer: int, ds: tuple[int, ...], hist: int = 2, k: int = 32) -> dict[int, np.ndarray]:
    """Memory-based CF / temporal streaming: the signature of a token is the routed sets of the last `hist` layers; the
    k most similar training tokens vote with the set they routed d layers later."""
    trace, train, test = sp.trace, sp.train, sp.test
    ds = tuple(d for d in ds if layer + d < N_LAYER)
    cols = [trace.indicator(layer - j) for j in range(min(hist, layer + 1))]
    sig = np.concatenate(cols, axis=1)
    sim = sig[test] @ sig[train].T
    nbr = np.argpartition(-sim, k, axis=1)[:, :k]
    out = {}
    for d in ds:
        tgt = trace.indicator(layer + d)[train]
        out[d] = tgt[nbr].mean(axis=1)
    return out


def _context_ids(trace: Trace, rows: np.ndarray, layer: int, order: int) -> np.ndarray | None:
    """Context id of each row: order 1 = top-1 of this layer, 2 = top-1 and top-2, 3 = top-1 of this and the layer
    before, 4 = top-1 of the last three layers."""
    t = trace.top.astype(np.int64)
    if order == 1:
        return t[rows, layer, 0]
    if order == 2:
        return t[rows, layer, 0] * N_EXPERT + t[rows, layer, 1]
    need = 1 if order == 3 else 2
    if layer < need:
        return None
    ids = t[rows, layer, 0]
    for j in range(1, need + 1):
        ids = ids * N_EXPERT + t[rows, layer - j, 0]
    return ids


def ppm_score(sp: Split, layer: int, d: int, base: np.ndarray, tau: float = 4.0) -> np.ndarray:
    """PPM / variable-order Markov (TAGE-like longest match): the score of the shorter context is replaced by
    lambda p_ctx + (1 - lambda) score with lambda = n / (n + tau), n the training tokens seen with this context.
    Order 0 is `base` (the count chain)."""
    trace, train, test = sp.trace, sp.train, sp.test
    tgt = trace.indicator(layer + d)[train]
    score = base.copy()
    for order in (1, 2, 3, 4):
        tr, te = _context_ids(trace, train, layer, order), _context_ids(trace, test, layer, order)
        if tr is None:
            continue
        uniq, inv = np.unique(tr, return_inverse=True)
        sums = np.zeros((len(uniq), N_EXPERT), np.float32)
        np.add.at(sums, inv, tgt)
        n = np.bincount(inv, minlength=len(uniq)).astype(np.float32)
        pos = np.clip(np.searchsorted(uniq, te), 0, len(uniq) - 1)
        seen = uniq[pos] == te
        lam = np.where(seen, n[pos] / (n[pos] + tau), 0.0)[:, None]
        score = lam * (sums[pos] / n[pos, None]) + (1 - lam) * score
    return score


def markov_score(trace: Trace, counts_rows: np.ndarray, test: np.ndarray, layer: int, d: int, top_n: int):
    """Correlation prefetcher: the successors of the top-1 expert, the table keeping only the top_n per entry."""
    t1 = trace.top[:, layer, 0].astype(np.int64)
    tgt = trace.indicator(layer + d)[counts_rows]
    sums = np.zeros((N_EXPERT, N_EXPERT), np.float32)
    np.add.at(sums, t1[counts_rows], tgt)
    n = np.bincount(t1[counts_rows], minlength=N_EXPERT).astype(np.float32)
    cond = sums / np.maximum(n, 1)[:, None]
    keep = np.zeros_like(cond)
    idx = np.argsort(-cond, axis=1, kind="stable")[:, :top_n]
    np.put_along_axis(keep, idx, np.take_along_axis(cond, idx, axis=1), axis=1)
    return keep[t1[test]]


def history_features(trace: Trace, rows: np.ndarray, layer: int, hist: int, logits: bool = False) -> np.ndarray:
    """Feature matrix for the linear predictors: routed-set indicators of the last `hist` layers (or the router
    logits of this layer, centered per token) and a bias."""
    if logits:
        x = trace.logits[rows, layer].astype(np.float32)
        parts = [(x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)]
    else:
        parts = [trace.indicator(layer - j)[rows] for j in range(min(hist, layer + 1))]
    return np.concatenate([*parts, np.ones((len(rows), 1), np.float32)], axis=1)


def ridge_scores(sp: Split, layer: int, ds, hist: int = 1, lam: float = 100.0, logits: bool = False):
    """Perceptron-style linear predictor per target expert (batch ridge, the same function class): one solve per
    layer gives the weights for every distance in `ds`."""
    trace, train, test = sp.trace, sp.train, sp.test
    ds = [d for d in ds if layer + d < N_LAYER]
    if not ds:
        return {}
    f_tr, f_te = history_features(trace, train, layer, hist, logits), history_features(trace, test, layer, hist, logits)
    gram = f_tr.T @ f_tr + lam * np.eye(f_tr.shape[1], dtype=np.float32)
    y = np.concatenate([trace.indicator(layer + d)[train] for d in ds], axis=1)
    w = np.linalg.solve(gram, f_tr.T @ y)
    pred = f_te @ w
    return {d: pred[:, i * N_EXPERT : (i + 1) * N_EXPERT] for i, d in enumerate(ds)}


def chain_for_rows(trace: Trace, rows: np.ndarray, counts: Counts, layer: int, d: int) -> np.ndarray:
    return chain(trace.indicator(layer)[rows], counts, layer, d)
