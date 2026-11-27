"""Reader for TONA_ROUTE_TRACE_P files (engine/gpu/mega_select.cpp).

A record is one layer of one pass: layer (u8), nb (u8, tokens), then per token the router logits of this layer and the
logits of the prediction for the next layer (float32 each): 256 of each for the Coder, 512 for the 512-expert model
(set_experts, the file does not say). The simulations need per layer-pass the union of the
chosen experts and the preload candidates.
"""

import heapq
from array import array
from dataclasses import dataclass

N_EXPERT = 256  # the Coder; the default of set_experts
_STATE = {"n": N_EXPERT}
N_USED = 10  # experts chosen per token
N_PREDICT = 16  # experts taken per token from the prediction for the next layer


def set_experts(n: int) -> None:
    """Experts per layer of the traced model (the rows of the trace have this many logits)."""
    _STATE["n"] = n


def n_experts() -> int:
    return _STATE["n"]


@dataclass(frozen=True)
class LayerPass:
    layer: int
    ntok: int
    union: tuple[int, ...]  # chosen experts of all tokens, first seen first
    cand: tuple[int, ...]  # preload candidates for layer + 1, best first


def top_k(values, k: int) -> list[int]:
    """Indices of the k largest values, ties by lower index (a stable descending sort)."""
    return heapq.nlargest(k, range(len(values)), key=values.__getitem__)


def layer_pass(layer: int, router: list[array], pred: list[array]) -> LayerPass:
    """Union by the top-10 of the router logits (softmax keeps the order), candidates by how many tokens name an
    expert in their top-16 of the prediction, ties by the best logit."""
    union = dict.fromkeys(e for row in router for e in top_k(row, N_USED))
    count: dict[int, int] = {}
    best: dict[int, float] = {}
    for row in pred:
        for e in top_k(row, N_PREDICT):
            count[e] = count.get(e, 0) + 1
            best[e] = max(best.get(e, -1e30), row[e])
    cand = sorted(count, key=lambda e: (-count[e], -best[e]))
    return LayerPass(layer, len(router), tuple(union), tuple(cand))


def _rows(buf: bytes, n: int) -> list[array]:
    rows, row_bytes = [], n_experts() * 4
    for i in range(n):
        row = array("f")
        row.frombytes(buf[i * row_bytes : (i + 1) * row_bytes])
        rows.append(row)
    return rows


def read_layer_passes(path: str):
    """Yields the LayerPass of each record in file order."""
    row_bytes = n_experts() * 4
    with open(path, "rb") as f:
        while head := f.read(2):
            layer, nb = head[0], head[1]
            body = f.read(2 * nb * row_bytes)
            if len(body) != 2 * nb * row_bytes:
                raise ValueError(f"{path}: truncated record at layer {layer}")
            yield layer_pass(layer, _rows(body[: nb * row_bytes], nb), _rows(body[nb * row_bytes :], nb))


def load_passes(path: str) -> list[list[LayerPass]]:
    """The records grouped into passes: a new pass starts at each record of layer 0."""
    passes: list[list[LayerPass]] = []
    for lp in read_layer_passes(path):
        if lp.layer == 0 or not passes:
            passes.append([])
        passes[-1].append(lp)
    return passes
