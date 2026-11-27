"""The experiment suite of the co-activation replay: stability across workloads, tables updated online with
decay, and table sizes. The plan with the questions and thresholds is in
research/measurements/2026-11-15-expert-coactivation.md.
"""

import itertools

import numpy as np
from coact_tables import Acc, count, evaluate, pass_bounds
from coact_trace import N_EXPERT, N_LAYER, N_USED, Split, Trace

HOT_SLOTS = 85


def stability_plan(trace: Trace) -> dict[str, tuple[list[str], list[str], bool]]:
    """name -> (count groups, test groups, split within). Different prompts always; 'within' = even vs odd requests."""
    short = [g for g in sorted(set(trace.group)) if g != "long"]
    plan = {
        "codeA->codeB": (["codeA"], ["codeB"], False),
        "code->text": (["codeA", "codeB"], ["text"], False),
        "text->code": (["text"], ["codeA", "codeB"], False),
        "short->long": (short, ["long"], False),
        "code within": (["codeA", "codeB"], ["codeA", "codeB"], True),
        "text within": (["text"], ["text"], True),
    }
    have = set(trace.group)
    return {k: v for k, v in plan.items() if set(v[0]) | set(v[1]) <= have}


def online(sp: Split, layer: int, d: int, halflife: float | None, start: str):
    """Chain table updated while the test rows stream by (blocks of 8 passes, predicted before the update).
    start = "shipped" (counts of `train`) or "empty"; halflife in tokens (None = no decay). Returns an Acc."""
    trace, train, test = sp.trace, sp.train, sp.test
    src_all, tgt_all = trace.indicator(layer), trace.indicator(layer + d)
    pair = src_all[train].T @ tgt_all[train] if start == "shipped" else np.zeros((N_EXPERT, N_EXPERT), np.float32)
    fx = src_all[train].sum(0) if start == "shipped" else np.zeros(N_EXPERT, np.float32)
    ft = tgt_all[train].sum(0) if start == "shipped" else np.zeros(N_EXPERT, np.float32)
    keep = 1.0 if halflife is None else 0.5 ** (1.0 / halflife)
    acc = Acc()
    bounds_all = pass_bounds(trace, test)
    for b0 in range(0, len(bounds_all), 8):
        lo = bounds_all[b0]
        hi = bounds_all[b0 + 8] if b0 + 8 < len(bounds_all) else len(test)
        rows = test[lo:hi]
        src, truth = src_all[rows], tgt_all[rows]
        hot = np.zeros(N_EXPERT, bool)
        hot[np.argsort(-ft, kind="stable")[:HOT_SLOTS]] = True
        score = src @ (pair / np.maximum(fx, 1.0)[:, None]) / N_USED + 1e-6 * ft
        bounds = pass_bounds(trace, rows)
        acc.add("online", d, evaluate(score, truth, hot, bounds, ks=(10,)))
        # control: the most used experts of the target layer in the decayed window, no conditioning
        acc.add("online_marginal", d, evaluate(np.tile(ft, (len(rows), 1)), truth, hot, bounds, ks=(10,)))
        decay = keep ** len(rows)
        pair, fx, ft = pair * decay + src.T @ truth, fx * decay + src.sum(0), ft * decay + truth.sum(0)
    return acc


def online_suite(trace: Trace, train: np.ndarray, test: np.ndarray, layers: range) -> dict:
    out: dict = {}
    sp = Split(trace, train, test)
    for start in ("shipped", "empty"):
        for half in (None, 5000.0, 500.0):
            total = Acc()
            for d in (1, 5, 10):
                for layer in layers:
                    if layer + d < N_LAYER:
                        a = online(sp, layer, d, half, start)
                        for k, v in a.sums.items():
                            total.sums[k] = total.sums.get(k, 0) + v
            out[f"{start} halflife={half}"] = {
                name: {
                    str(d): total.row(name, d, "all", 10) | {"cold": total.row(name, d, "cold", 10)} for d in (1, 5, 10)
                }
                for name in ("online", "online_marginal")
            }
    return out


def table_sizes(trace: Trace, train: np.ndarray) -> dict:
    """Bytes of the dense and sparse tables and the multiply-adds per token and layer (counted from the algorithm)."""
    counts = count(trace, train, max_d=4)
    n_pairs = sum(N_LAYER - d for d in range(1, 21))  # (l, d) pairs for d = 1..20
    dense = n_pairs * N_EXPERT * N_EXPERT * 2
    out = {
        "pairs d=1..20": n_pairs,
        "dense_bytes_16bit": dense,
        "within_dense_bytes": N_LAYER * N_EXPERT * N_EXPERT * 2,
    }
    for top_n in (8, 16, 32):
        out[f"sparse_top{top_n}_bytes"] = n_pairs * N_EXPERT * top_n * 2  # one id byte and one probability byte
    live = counts.pair[(10, 1)]
    out["nonzero_share_d1"] = float((live > 0).mean())
    out["macs_per_token_layer"] = {
        "chain/pmi/cosine (10 routed x 256 targets, per distance)": N_USED * N_EXPERT,
        "partner completion (m seeds x 256)": 8 * N_EXPERT,
        "ridge_set (256 inputs x 256 targets, per distance)": N_EXPERT * N_EXPERT,
        "perceptron2": 2 * N_EXPERT * N_EXPERT,
        "knn (signature 512 against all counted tokens)": 2 * N_EXPERT * len(train),
        "ppm (4 contexts x 256)": 4 * N_EXPERT,
        "markov1 (top-10 successors)": 10,
    }
    return out


def code_text_boundary(trace: Trace) -> int:
    """For a trace of code requests followed by text requests without a request list: the pass where the routing turns
    from code-like to text-like (log-likelihood ratio of the first and the last third of the passes at every 4th
    layer, smoothed over 9 passes, best single change point)."""
    n = len(trace.pass_start)
    pof = trace.pass_of
    code, text = np.flatnonzero(pof < n // 3), np.flatnonzero(pof >= 2 * n // 3)
    llr = np.zeros(n)
    for layer in range(4, N_LAYER - 4, 4):
        x = trace.indicator(layer)
        w = np.log(((x[code].sum(0) + 0.5) / len(code)) / ((x[text].sum(0) + 0.5) / len(text)))
        llr += np.bincount(pof, weights=x @ w, minlength=n)
    sign = (np.convolve(llr, np.ones(9) / 9, mode="same") > 0).astype(int)
    return int(max(range(n // 4, 7 * n // 8), key=lambda b: (sign[:b] == 1).sum() + (sign[b:] == 0).sum()))


LIFT_BINS = (0.0, 0.2, 0.5, 1.0, 2.0, 5.0, 1e9)


def partner_stats(trace: Trace, train: np.ndarray, test: np.ndarray, layers) -> dict:
    """Horizontal structure of a layer (market-basket view): the lift P(XY)/(P(X)P(Y)) of the expert pairs inside a
    layer from the counting rows, binned, and how the same pairs behave on the test rows. Only pairs with at least 20
    expected co-occurrences in the counting rows. 'confidence' = P(Y given X) of the best partner Y of X."""
    pairs = np.zeros(len(LIFT_BINS) - 1)
    test_lift_sum = np.zeros(len(LIFT_BINS) - 1)
    top1, top1_test, n_layers = 0.0, 0.0, 0
    for layer in layers:
        x_tr, x_te = trace.indicator(layer)[train], trace.indicator(layer)[test]
        c_tr, c_te = x_tr.T @ x_tr, x_te.T @ x_te
        f_tr, f_te = x_tr.sum(0), x_te.sum(0)
        exp_tr, exp_te = np.outer(f_tr, f_tr) / len(train), np.outer(f_te, f_te) / len(test)
        ok = (exp_tr >= 20) & (exp_te >= 2) & ~np.eye(N_EXPERT, dtype=bool)
        lift_tr, lift_te = c_tr[ok] / exp_tr[ok], c_te[ok] / exp_te[ok]
        which = np.digitize(lift_tr, LIFT_BINS[1:-1])
        pairs += np.bincount(which, minlength=len(pairs))
        test_lift_sum += np.bincount(which, weights=lift_te, minlength=len(pairs))
        cond_tr = c_tr / np.maximum(f_tr, 1)[:, None]
        np.fill_diagonal(cond_tr, 0)
        best = cond_tr.argmax(1)
        cond_te = c_te / np.maximum(f_te, 1)[:, None]
        top1 += float(cond_tr.max(1).mean())
        top1_test += float(cond_te[np.arange(N_EXPERT), best].mean())
        n_layers += 1
    return {
        "bins": [f"{lo:g}-{hi:g}" for lo, hi in itertools.pairwise(LIFT_BINS)],
        "pair_share": (pairs / pairs.sum()).tolist(),
        "mean_test_lift": (test_lift_sum / np.maximum(pairs, 1)).tolist(),
        "best_partner_confidence_train": top1 / n_layers,
        "same_partner_confidence_test": top1_test / n_layers,
    }
