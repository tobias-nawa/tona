"""The co-activation replay on a synthetic trace with a planted dependency between layers."""

import json
import struct

import numpy as np
from coact_tables import chain, count, evaluate, lookahead, marginal, pass_bounds, set_counts
from coact_trace import N_EXPERT, N_LAYER, cut_requests, read_trace

RNG = np.random.default_rng(1)


def write_trace(path, passes: int, nb: int = 2) -> list[list[list[list[int]]]]:
    """Layer l + 1 of a token takes experts (e + 1) mod 256 of the 10 experts of layer l: a perfect chain. The logits of
    the lookahead name the true next experts. Returns routes[pass][layer][token] = experts."""
    routes = []
    with open(path, "wb") as f:
        for _ in range(passes):
            layers = []
            first = [list(RNG.choice(N_EXPERT, 10, replace=False)) for _ in range(nb)]
            for layer in range(N_LAYER):
                cur = [[(e + layer) % N_EXPERT for e in toks] for toks in first]
                nxt = [[(e + layer + 1) % N_EXPERT for e in toks] for toks in first]
                layers.append(cur)
                router, look = np.zeros((nb, N_EXPERT), np.float32), np.zeros((nb, N_EXPERT), np.float32)
                for b in range(nb):
                    router[b, cur[b]] = 5.0 - 0.1 * np.arange(10)
                    look[b, nxt[b]] = 5.0 - 0.1 * np.arange(10)
                f.write(struct.pack("BB", layer, nb) + router.tobytes() + look.tobytes())
            routes.append(layers)
    return routes


def test_read_trace_and_chain(tmp_path):
    routes = write_trace(tmp_path / "t.bin", 400)
    trace = read_trace(str(tmp_path / "t.bin"))
    assert trace.rows == 800 and len(trace.pass_start) == 400
    assert sorted(trace.top[0, 3]) == sorted(routes[0][3][0])
    train, test = np.arange(0, 600), np.arange(600, 800)
    counts = count(trace, train, max_d=3)
    bounds = pass_bounds(trace, test)
    assert len(bounds) == 100
    truth, hot = trace.indicator(4)[test], counts.hot(4)
    look = evaluate(lookahead(trace, test, 3, counts), truth, hot, bounds)[("all", 10)]
    assert look[0] / look[2] > 0.99  # the planted lookahead is exact
    sc = chain(trace.indicator(3)[test], counts, 3, 1)
    got = evaluate(sc, truth, hot, bounds)[("all", 10)]
    assert got[0] / got[2] > 0.9  # the table finds the shifted experts
    marg = evaluate(marginal(counts, 4, len(test)), truth, hot, bounds)[("all", 10)]
    assert marg[0] / marg[2] < 0.3


def test_set_counts_union():
    pred = np.zeros((2, N_EXPERT), np.float32)
    truth = np.zeros((2, N_EXPERT), np.float32)
    pred[0, [1, 2]] = 1
    pred[1, [2, 3]] = 1
    truth[0, [1, 9]] = 1
    truth[1, [3, 9]] = 1
    hot = np.zeros(N_EXPERT, bool)
    v = set_counts(pred, truth, hot, np.array([0]))  # both rows one pass
    assert v[0] == 2 and v[1] == 4 and v[2] == 4  # tokens
    assert v[6] == 2 and v[7] == 3 and v[8] == 3  # union: predicted {1,2,3}, truth {1,3,9}


def test_cut_requests(tmp_path):
    write_trace(tmp_path / "t.bin", 6, nb=1)
    trace = read_trace(str(tmp_path / "t.bin"))
    pos = [100, 101, 102, 5, 6, 7]  # requests start at passes 0 and 3
    (tmp_path / "log").write_text("".join(f"{p} 0 1 2\n{p} 1 3 4\n" for p in pos))
    (tmp_path / "req").write_text("".join(json.dumps({"group": g}) + "\n" for g in ("codeA", "text")))
    # the layers of a pass are written one after the other; layer 0 lines carry the positions
    cut_requests(trace, str(tmp_path / "log"), str(tmp_path / "req"))
    assert list(trace.request) == [0, 0, 0, 1, 1, 1] and trace.group == ["codeA", "text"]
