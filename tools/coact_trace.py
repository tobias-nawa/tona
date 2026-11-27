"""Reader for the co-activation replay: TONA_ROUTE_TRACE_P (router logits and the lookahead per token and layer)
and, if given, TONA_ROUTE_LOG plus the request list of tools/coactivation_drive.py (to cut the tokens into requests).

A record of the logit trace is one layer of one pass: layer (u8), nb (u8), then nb rows of 256 router logits and nb rows
of 256 lookahead logits (the router of layer + 1 on the input of this layer), float32. Passes hold 1 to 8 token rows
(verify passes include the rejected drafts, the rows the pass really routed).
"""

import json
from dataclasses import dataclass, field

import numpy as np

N_LAYER, N_EXPERT, N_USED, N_LOOK = 48, 256, 10, 32


@dataclass(frozen=True)
class Split:
    """A trace with the rows that count (train) and the rows that are predicted (test)."""

    trace: "Trace"
    train: np.ndarray
    test: np.ndarray


@dataclass
class Trace:
    """Per token row and layer: the router logits (float16), the top-10 chosen experts, the lookahead's top-32."""

    logits: np.ndarray  # (T, L, 256) float16
    top: np.ndarray  # (T, L, 10) uint8
    look: np.ndarray  # (T, L, 32) uint8, best first: prediction for layer l + 1
    pass_start: np.ndarray  # (P,) first row of each pass
    request: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))  # (T,) request index
    group: list[str] = field(default_factory=list)  # group name per request

    @property
    def rows(self) -> int:
        return self.top.shape[0]

    @property
    def pass_of(self) -> np.ndarray:
        """Pass index of every row."""
        out = np.zeros(self.rows, np.int64)
        out[self.pass_start[1:]] = 1
        return np.cumsum(out)

    def indicator(self, layer: int) -> np.ndarray:
        """(T, 256) float32, 1 where the token chose the expert in this layer."""
        x = np.zeros((self.rows, N_EXPERT), np.float32)
        np.put_along_axis(x, self.top[:, layer].astype(np.int64), 1.0, axis=1)
        return x


def _top(rows: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-rows, axis=1, kind="stable")[:, :k].astype(np.uint8)


def read_trace(path: str) -> Trace:
    buf = np.memmap(path, dtype=np.uint8, mode="r")
    i, n = 0, len(buf)
    chunks: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = []  # layer, nb, logits, top, look
    while i + 2 <= n:
        layer, nb = int(buf[i]), int(buf[i + 1])
        size = nb * N_EXPERT * 4
        if i + 2 + 2 * size > n:
            break  # a truncated last record (server stopped)
        r = np.frombuffer(buf[i + 2 : i + 2 + size], np.float32).reshape(nb, N_EXPERT)
        p = np.frombuffer(buf[i + 2 + size : i + 2 + 2 * size], np.float32).reshape(nb, N_EXPERT)
        i += 2 + 2 * size
        chunks.append((layer, nb, r.astype(np.float16), _top(r, N_USED), _top(p, N_LOOK)))
    return _assemble(chunks[: len(chunks) // N_LAYER * N_LAYER])  # whole passes only


def _assemble(chunks) -> Trace:
    starts, total = [], 0
    for layer, nb, *_ in chunks:
        if layer == 0:
            starts.append(total)
            total += nb
    logits = np.zeros((total, N_LAYER, N_EXPERT), np.float16)
    top = np.zeros((total, N_LAYER, N_USED), np.uint8)
    look = np.zeros((total, N_LAYER, N_LOOK), np.uint8)
    base, passes = -1, 0
    for layer, nb, r, t, p in chunks:
        if layer == 0:
            base = starts[passes]
            passes += 1
        logits[base : base + nb, layer], top[base : base + nb, layer], look[base : base + nb, layer] = r, t, p
    return Trace(logits, top, look, np.array(starts, np.int64))


def cut_requests(trace: Trace, route_log: str, requests: str) -> None:
    """Fills trace.request and trace.group: the first position of each pass starts a new request when it does not
    continue the previous pass (an accepted run of 1..nb tokens), checked against the request list."""
    first_pos: list[int] = []
    with open(route_log) as f:
        want = {int(s) for s in trace.pass_start}
        row = 0
        for line in f:
            parts = line.split(maxsplit=2)
            if int(parts[1]) != 0:
                continue
            if row in want:
                first_pos.append(int(parts[0]))
            row += 1
    if row != trace.rows:
        raise ValueError(f"{row} token rows in the route log, {trace.rows} in the logit trace")
    with open(requests) as f:
        groups = [json.loads(line)["group"] for line in f]
    sizes = np.diff(np.append(trace.pass_start, trace.rows))
    req = [0]
    for p in range(1, len(first_pos)):
        step = first_pos[p] - first_pos[p - 1]
        req.append(req[-1] + (0 if 1 <= step <= sizes[p - 1] else 1))
    if len(groups) != req[-1] + 1:
        raise ValueError(f"{req[-1] + 1} requests found in the route log, {len(groups)} in {requests}")
    per_row = np.repeat(np.array(req, np.int32), sizes)
    trace.request, trace.group = per_row, groups
