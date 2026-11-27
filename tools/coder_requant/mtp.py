"""The MTP head cut to a list of experts: a byte copy of the kept experts and router rows, expert_count set.

The head has one layer with experts (blk.<n_layer>, 512 experts in the ggml-org file). The keep list comes from the
routing statistics of the head itself on coding load (counted by the Metal backend), one JSON file:
a flat list of expert indices, or {"keep": [indices]} (also {"keep": [[indices]]}).
"""

import json
from pathlib import Path

from coder_requant import slicing, sources
from coder_requant.gguf_layout import Layout


def read_keep_list(path: str) -> list[int]:
    doc = json.loads(Path(path).read_text())
    keep = doc["keep"] if isinstance(doc, dict) else doc
    if keep and isinstance(keep[0], list):
        if len(keep) != 1:
            raise ValueError(f"{path}: the MTP head has one layer, the file has {len(keep)} lists")
        keep = keep[0]
    if not all(isinstance(e, int) and not isinstance(e, bool) for e in keep):
        raise ValueError(f"{path}: the keep list holds something other than integers")
    return sorted(keep)


def expert_layer(layout: Layout) -> int:
    """The one layer whose tensors carry experts; the cut needs exactly one."""
    layers = {slicing.per_expert_layer(t.name) for t in layout.tensors} - {None}
    if len(layers) != 1:
        raise ValueError(f"expected one layer with experts, found {sorted(layers)}")
    return layers.pop()


def keep_for(layout: Layout, keep: list[int]) -> dict[int, list[int]]:
    """The keep dict of slicing for this file; rejects duplicates, out-of-range indices and an empty list."""
    n = max(t.dims[-1] for t in layout.tensors if slicing.per_expert_layer(t.name) is not None)
    if not keep or len(set(keep)) != len(keep) or keep[0] < 0 or keep[-1] >= n:
        raise ValueError(f"keep list must hold distinct indices in 0..{n - 1}")
    return {expert_layer(layout): keep}


def check_expert_bytes(src: str, final: str, keep: dict[int, list[int]]) -> None:
    """Every kept expert of every expert tensor is byte-equal to the source expert it claims to be."""
    sr, fr = sources.Reader(Path(src)), sources.Reader(Path(final))
    sl, fl = sr.layout(), fr.layout()
    for t in sl.tensors:
        layer = slicing.per_expert_layer(t.name)
        if layer is None:
            continue
        per = t.nbytes // t.dims[-1]
        ft = fl.by_name[t.name]
        for j, e in enumerate(keep[layer]):
            want = sr.read(sl.data_start + t.offset + e * per, per)
            if want != fr.read(fl.data_start + ft.offset + j * per, per):
                raise SystemExit(f"check: {t.name} expert {j} is not source expert {e}")
