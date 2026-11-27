"""A fitted mixer of the lookahead and the co-activation predictors at d = 1: per (token, candidate expert) a
logistic regression over the features below, one weight vector for all layers, fitted on the counting set only with
cross-fitting (the tables and the ridge that give a training row's features never saw that row).
"""

import numpy as np
from coact_models import ridge_scores
from coact_tables import Acc, count, evaluate, lookahead, partners, pass_bounds
from coact_trace import N_USED, Split, Trace

FEATURE_SETS = {
    "look+ridge": (0, 1),
    "look+ridge+chain": (0, 1, 2),
    "look+ridge+chain+partner": (0, 1, 2, 3),
}


def features(sp: Split, layer: int, counts) -> np.ndarray:
    """(n, 256, 4): lookahead rank score, ridge on the router logits, count chain, partner score of the lookahead's
    top-3, all for layer + 1, from the tables `counts` and a ridge fitted on `sp.train` (the rows of `counts`)."""
    trace, rows = sp.trace, sp.test
    look = lookahead(trace, rows, layer, counts)
    ridge = ridge_scores(sp, layer, (1,), logits=True)[1]
    cond = counts.pair[(layer, 1)] / np.maximum(counts.freq[layer], 1.0)[:, None]
    chain = trace.indicator(layer)[rows] @ cond / N_USED
    partner = partners(look, counts, layer + 1, 3)
    return np.stack([np.maximum(look, 0) / 64.0, ridge, chain, partner], axis=2)


def fit_logistic(x: np.ndarray, y: np.ndarray, steps: int = 150, lr: float = 0.5) -> np.ndarray:
    """Weights (last = bias) of a logistic regression by full-batch gradient descent on standardized features."""
    mu, sd = x.mean(0), x.std(0) + 1e-6
    z = np.concatenate([(x - mu) / sd, np.ones((len(x), 1), np.float32)], axis=1)
    w = np.zeros(z.shape[1], np.float32)
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(z @ w)))
        w -= lr * (z.T @ (p - y)) / len(y)
    return np.concatenate([w[:-1] / sd, [w[-1] - (w[:-1] * mu / sd).sum()]]).astype(np.float32)


def training_matrix(trace: Trace, train: np.ndarray, layers) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fitted features of the counting rows (two halves by pass blocks) and the truth bits."""
    half = len(train) // 2
    parts = (train[:half], train[half:])
    tables = [count(trace, p, max_d=1) for p in parts]
    xs, ys = [], []
    for layer in layers:
        truth = trace.indicator(layer + 1)
        for a, b in ((0, 1), (1, 0)):
            f = features(Split(trace, parts[a], parts[b]), layer, tables[a])
            sub = np.random.default_rng(layer).choice(len(parts[b]), min(len(parts[b]), 600), replace=False)
            xs.append(f[sub].reshape(-1, f.shape[2]))
            ys.append(truth[parts[b]][sub].reshape(-1))
    return np.concatenate(xs), np.concatenate(ys)


def mixer_experiment(trace: Trace, train: np.ndarray, test: np.ndarray, layers) -> Acc:
    layers = [layer for layer in layers if layer + 1 < 48]
    x, y = training_matrix(trace, train, layers)
    weights = {name: fit_logistic(x[:, list(cols)], y) for name, cols in FEATURE_SETS.items()}
    acc = Acc()
    bounds = pass_bounds(trace, test)
    counts = count(trace, train, max_d=1)
    for layer in layers:
        f = features(Split(trace, train, test), layer, counts)
        truth, hot = trace.indicator(layer + 1)[test], counts.hot(layer + 1)
        for name, cols in FEATURE_SETS.items():
            w = weights[name]
            score = f[:, :, list(cols)] @ w[:-1] + w[-1]
            acc.add(name, 1, evaluate(score, truth, hot, bounds))
    acc.weights = {n: w.tolist() for n, w in weights.items()}  # type: ignore[attr-defined]
    return acc
