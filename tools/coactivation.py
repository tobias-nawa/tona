#!/usr/bin/env python3
"""Offline replay of expert co-activation predictors over a route trace (research/55-expert-coactivation.md).

Reads TONA_ROUTE_TRACE_P (and, to cut requests into workload groups, TONA_ROUTE_LOG with the request list of
tools/coactivation_drive.py), counts on one set of requests, predicts the experts of layer l + d for the requests of
another set, and prints recall and precision per predictor, distance and K as JSON (--json) or tables.

Usage: nice -n 19 tools/coactivation.py TRACE [--log ROUTE_LOG --requests REQ.jsonl] [--count GROUPS --test GROUPS]
                                         [--layers-step 3] [--out result.json]
"""

import argparse
import json
import sys

import numpy as np
from coact_mixer import mixer_experiment
from coact_models import (
    cosine_score,
    knn_scores,
    lift_prune,
    markov_score,
    pmi_score,
    ppm_score,
    ridge_scores,
    rule_prediction,
)
from coact_suite import code_text_boundary, online_suite, stability_plan, table_sizes
from coact_tables import (
    Acc,
    Counts,
    chain,
    count,
    evaluate,
    lookahead,
    marginal,
    multi_source,
    partners,
    pass_bounds,
    rrf,
    set_counts,
    tiebreak,
)
from coact_trace import N_LAYER, Split, Trace, cut_requests, read_trace

DS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20)  # distances of the count predictors
HEAVY_DS = (1, 2, 5, 10, 20)  # distances of the learned and memory-based predictors
THETAS = (0.5, 0.7, 0.9)


def request_rows(trace: Trace, groups: list[str], request_ids: set[int] | None = None) -> np.ndarray:
    """Rows of the requests of the given groups (all rows if the trace has no request list)."""
    if not len(trace.request):
        return np.arange(trace.rows)
    sel = np.isin(trace.request, [i for i, g in enumerate(trace.group) if g in groups])
    if request_ids is not None:
        sel &= np.isin(trace.request, list(request_ids))
    return np.flatnonzero(sel)


def split_requests(trace: Trace, groups: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Within one workload: even requests count, odd requests test (other prompts, the upper bound of a table)."""
    ids = [i for i, g in enumerate(trace.group) if g in groups]
    return request_rows(trace, groups, set(ids[0::2])), request_rows(trace, groups, set(ids[1::2]))


def count_predictors(trace, rows, counts: Counts, layer: int, d: int) -> dict[str, np.ndarray]:
    src = trace.indicator(layer)[rows]
    base = chain(src, counts, layer, d)
    out = {
        "marginal": marginal(counts, layer + d, len(rows)) + tiebreak(counts, layer + d),
        "chain": base,
        "pmi": pmi_score(src, counts, layer, d),
        "npmi": pmi_score(src, counts, layer, d, normalize=True),
        "cosine": cosine_score(src, counts, layer, d),
    }
    if layer >= 2:
        srcs = [trace.indicator(layer - j)[rows] for j in range(3)]
        out["path3"] = multi_source(srcs, counts, layer, d, (1.0, 0.5, 0.25))
    for m in (3, 5):
        comp = partners(base, counts, layer + d, m)
        seeds = np.argsort(-base, axis=1, kind="stable")[:, :m]
        seed_mask = np.zeros_like(base)
        np.put_along_axis(seed_mask, seeds, 1.0, axis=1)
        out[f"chain+partner{m}"] = np.where(seed_mask > 0, 1e3 + base, comp + tiebreak(counts, layer + d))
    return out


def one_layer(trace, train, test, counts, layer, acc: Acc) -> None:
    bounds = pass_bounds(trace, test)
    sp = Split(trace, train, test)
    heavy = {}
    for name, fn in (
        ("ridge_set", lambda: ridge_scores(sp, layer, HEAVY_DS, hist=1)),
        ("perceptron2", lambda: ridge_scores(sp, layer, HEAVY_DS, hist=2)),
        ("ridge_logit", lambda: ridge_scores(sp, layer, HEAVY_DS, logits=True)),
        ("knn", lambda: knn_scores(sp, layer, HEAVY_DS)),
    ):
        heavy[name] = fn()
    for d in DS:
        t = layer + d
        if t >= N_LAYER:
            continue
        truth, hot = trace.indicator(t)[test], counts.hot(t)
        scores = count_predictors(trace, test, counts, layer, d)
        if d == 1:
            scores["lookahead"] = lookahead(trace, test, layer, counts)
            _horizontal(scores, counts, t)
        if d in HEAVY_DS:
            scores.update({k: v[d] for k, v in heavy.items() if d in v})
            scores["ppm"] = ppm_score(sp, layer, d, scores["chain"])
            scores["markov1"] = markov_score(trace, train, test, layer, d, top_n=10)
            scores["mix"] = rrf(scores["chain"], scores["ridge_set"], scores["ppm"])
            if d == 1:
                scores["look+ridge_logit"] = rrf(scores["lookahead"], scores["ridge_logit"])
                scores["look+chain"] = rrf(scores["lookahead"], scores["chain"])
                scores["look+chain+ridge"] = rrf(scores["lookahead"], scores["chain"], scores["ridge_set"])
        for name, sc in scores.items():
            acc.add(name, d, evaluate(sc, truth, hot, bounds))
        src = trace.indicator(layer)[test]
        for theta in THETAS:
            pred = rule_prediction(src, counts, layer, d, theta)
            acc.add(f"rule{theta}", d, {("set", 0): set_counts(pred, truth, hot, bounds)})


def _horizontal(scores: dict, counts: Counts, target: int) -> None:
    look = scores["lookahead"]
    for m in (3, 5, 8):
        comp = partners(look, counts, target, m)
        seeds = np.argsort(-look, axis=1, kind="stable")[:, :m]
        mask = np.zeros_like(look)
        np.put_along_axis(mask, seeds, 1.0, axis=1)
        scores[f"look+partner{m}"] = np.where(mask > 0, 1e3 + look, comp + tiebreak(counts, target))
        scores[f"look*partner{m}"] = rrf(look, comp + tiebreak(counts, target))
    scores["look+prune"] = lift_prune(look, counts, target, floor=0.5, seeds=3)


def run(trace: Trace, train: np.ndarray, test: np.ndarray, layers: range) -> Acc:
    counts = count(trace, train)
    acc = Acc()
    for layer in layers:
        one_layer(trace, train, test, counts, layer, acc)
    return acc


def at_precision(acc: Acc, name: str, d: int, mode: str, key: str, target: float) -> float:
    """Recall (`key` = recall, cold_recall, union_recall, ...) at precision `target`, interpolated over K."""
    ks = sorted(k for (n, dd, m, k) in acc.sums if (n, dd, m) == (name, d, mode))
    pts = [(acc.row(name, d, mode, k)[key.replace("recall", "precision")], acc.row(name, d, mode, k)[key]) for k in ks]
    pts = sorted(pts)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if not xs or target < xs[0] or target > xs[-1]:
        return float("nan")
    return float(np.interp(target, xs, ys))


def to_json(acc: Acc) -> dict:
    out: dict = {}
    for name, d, mode, k in sorted(acc.sums, key=str):
        out.setdefault(name, {}).setdefault(str(d), {}).setdefault(mode, {})[str(k)] = acc.row(name, d, mode, k)
    return out


def label_by_pass(trace: Trace, boundary: int) -> None:
    """Workload labels without a route log: a request per pass block (codeA, codeB, text), for the traces recorded
    before the driver existed. The boundary comes from code_text_boundary."""
    n = len(trace.pass_start)
    starts = [0, boundary // 2, boundary, (boundary + n) // 2]
    pass_req = np.searchsorted(starts, np.arange(len(trace.pass_start)), side="right") - 1
    sizes = np.diff(np.append(trace.pass_start, trace.rows))
    trace.request = np.repeat(pass_req.astype(np.int32), sizes)
    trace.group = ["codeA", "codeB", "text", "text"]


def run_suite(trace: Trace, args) -> int:
    layers = range(1, N_LAYER - 1, args.layers_step)
    result: dict = {"experiments": {}}
    stability = {} if args.only_mixer else stability_plan(trace)
    for name, (cg, tg, within) in stability.items():
        train, test = split_requests(trace, cg) if within else (request_rows(trace, cg), request_rows(trace, tg))
        if not len(train) or not len(test):
            continue
        print(f"{name}: count rows {len(train)}, test rows {len(test)}", file=sys.stderr)
        result["experiments"][name] = {"count_rows": len(train), "test_rows": len(test)}
        result["experiments"][name]["result"] = to_json(run(trace, train, test, layers))
    for name, groups in (("codeA->codeB", (["codeA"], ["codeB"])), ("code->text", (["codeA", "codeB"], ["text"]))):
        acc = mixer_experiment(trace, request_rows(trace, groups[0]), request_rows(trace, groups[1]), layers)
        result["mixer " + name] = {"result": to_json(acc), "weights": acc.weights}
    train, test = split_requests(trace, ["codeA", "codeB"])
    result["online"] = online_suite(trace, train, test, layers)
    result["sizes"] = table_sizes(trace, train)
    with open(args.out, "w") as f:
        json.dump(result, f)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--log")
    ap.add_argument("--requests")
    ap.add_argument("--count", default="codeA", help="groups to count on, comma separated")
    ap.add_argument("--test", default="codeB", help="groups to test on")
    ap.add_argument("--within", action="store_true", help="count on even and test on odd requests of --count")
    ap.add_argument("--suite", action="store_true", help="stability experiments, online tables, sizes (needs --log)")
    ap.add_argument("--devsplit", help="no request list: pass number N (or auto) where code turns into text")
    ap.add_argument("--only-mixer", action="store_true", help="with --suite: only the mixer, online and sizes")
    ap.add_argument("--layers-step", type=int, default=3)
    ap.add_argument("--out")
    args = ap.parse_args()
    trace = read_trace(args.trace)
    if args.log:
        cut_requests(trace, args.log, args.requests)
    if args.devsplit:
        label_by_pass(trace, code_text_boundary(trace) if args.devsplit == "auto" else int(args.devsplit))
    if args.within:
        train, test = split_requests(trace, args.count.split(","))
    elif args.log or args.devsplit:
        train, test = request_rows(trace, args.count.split(",")), request_rows(trace, args.test.split(","))
    else:  # no request list: the first 2/3 of the passes count, the rest tests (development only)
        cut = int(trace.pass_start[int(len(trace.pass_start) * 2 / 3)])
        train, test = np.arange(cut), np.arange(cut, trace.rows)
    if args.suite:
        return run_suite(trace, args)
    print(f"count rows {len(train)}, test rows {len(test)}", file=sys.stderr)
    acc = run(trace, train, test, range(1, N_LAYER - 1, args.layers_step))
    res = to_json(acc)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f)
    else:
        print(json.dumps({n: res[n] for n in ("marginal", "lookahead", "chain") if n in res}, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
