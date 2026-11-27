"""Which experts a smaller variant keeps: score each (layer, expert) on coding traffic, keep the best per layer.

Score, stage 1 of the REAP criterion (arXiv 2510.13999): the summed gate weight a token gives an expert (softmax over
its top-10 router logits, as the model mixes them), from TONA_ROUTE_TRACE_P files. Stage 2 would multiply by the
norm of the expert's output, which no trace records. Without traces: use counts of TONA_EXPERT_PROFILE.
"""

import json
import math
import struct
from array import array

N_USED = 10  # experts mixed per token (expert_used_count of the Coder)
PROFILE_MAGIC = 0x50584549  # engine/gpu/mega_experts.cpp, load_profile


def gate_weights(row, k: int = N_USED) -> dict[int, float]:
    """Mixing weight of the top-k experts of one router row: softmax over the k logits (ties: lower index)."""
    top = sorted(range(len(row)), key=lambda e: (-row[e], e))[:k]
    m = row[top[0]]
    w = {e: math.exp(row[e] - m) for e in top}
    s = sum(w.values())
    return {e: v / s for e, v in w.items()}


def scores_from_trace(paths: list[str], n_layer: int, n_expert: int) -> list[list[float]]:
    """Summed gate weights per layer and expert over all tokens of the TONA_ROUTE_TRACE_P files."""
    score = [[0.0] * n_expert for _ in range(n_layer)]
    row_bytes = 4 * n_expert
    for path in paths:
        with open(path, "rb") as f:
            while head := f.read(2):
                layer, nb = head[0], head[1]
                body = f.read(2 * nb * row_bytes)
                if len(body) != 2 * nb * row_bytes:
                    raise ValueError(f"{path}: truncated record at layer {layer}")
                for t in range(nb):  # the router rows; the prediction rows that follow are not used
                    row = array("f")
                    row.frombytes(body[t * row_bytes : (t + 1) * row_bytes])
                    for e, w in gate_weights(row).items():
                        score[layer][e] += w
    return score


def scores_from_profile(path: str) -> list[list[float]]:
    with open(path, "rb") as f:
        magic, n_layer, n_expert = struct.unpack("<3I", f.read(12))
        if magic != PROFILE_MAGIC:
            raise ValueError(f"{path}: not an expert profile")
        counts = struct.unpack(f"<{n_layer * n_expert}I", f.read(4 * n_layer * n_expert))
    return [[float(c) for c in counts[i * n_expert : (i + 1) * n_expert]] for i in range(n_layer)]


def ranked(scores: list[float]) -> list[int]:
    """Experts best first; ties by lower index, so the result is deterministic."""
    return sorted(range(len(scores)), key=lambda e: (-scores[e], e))


def keep_equal(score: list[list[float]], n_keep: int) -> list[list[int]]:
    return [sorted(ranked(s)[:n_keep]) for s in score]


def keep_adaptive(score: list[list[float]], n_keep: int, floor: int) -> list[list[int]]:
    """The same total as keep_equal, taken globally by each expert's share of its layer's score mass; every layer
    keeps at least `floor`. Unequal counts have no loader yet (slicing.equal_count), so this is for analysis."""
    n_layer, n_expert = len(score), len(score[0])
    if not N_USED <= floor <= n_keep <= n_expert:
        raise ValueError(f"need {N_USED} <= floor {floor} <= n_keep {n_keep} <= {n_expert}")
    order = [ranked(s) for s in score]
    keep = [o[:floor] for o in order]
    mass = [sum(s) or 1.0 for s in score]
    rest = [(score[y][e] / mass[y], -y, -e, y, e) for y in range(n_layer) for e in order[y][floor:]]
    rest.sort(reverse=True)
    for *_, y, e in rest[: n_layer * (n_keep - floor)]:
        keep[y].append(e)
    return [sorted(k) for k in keep]


def dropped_mass(score: list[list[float]], keep: list[list[int]]) -> list[float]:
    """Per layer: share of the score mass on the experts not kept."""
    out = []
    for s, k in zip(score, keep, strict=True):
        total = sum(s)
        out.append(0.0 if total == 0 else 1.0 - sum(s[e] for e in k) / total)
    return out


def write_keep(path: str, keep: list[list[int]], score: list[list[float]], source: str) -> None:
    doc = {
        "n_expert_in": len(score[0]),
        "source": source,
        "dropped_mass": [round(x, 6) for x in dropped_mass(score, keep)],
        "keep": keep,
    }
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)


def read_keep(path: str) -> dict[int, list[int]]:
    with open(path) as f:
        doc = json.load(f)
    return {layer: list(k) for layer, k in enumerate(doc["keep"])}
