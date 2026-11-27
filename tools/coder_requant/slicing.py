"""Plans and writes GGUF files as byte ranges of source files: expert slices, imatrix slices, the final assembly.

Every per-expert tensor keeps its experts on the slowest axis (ggml ne[last]), so one expert is one contiguous byte
range and a slice is a list of ranges. The same holds for the router (ffn_gate_inp, one row per expert) and for the
imatrix entries of the expert tensors (in_sum2 [ncols, E], counts [1, E]).
"""

import re
import struct
from collections.abc import Callable
from dataclasses import dataclass

from mini_model.gguf_io import KV

from coder_requant.gguf_layout import Layout, arch_key, with_scalar

EXPERT_WEIGHT = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
ROUTER = re.compile(r"^blk\.(\d+)\.ffn_gate_inp\.weight$")
IMATRIX_EXPERT = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight\.(in_sum2|in_sum|counts)$")
PIECE = 64 << 20  # copy granularity: bounded memory for tensors of up to ~1 GB


@dataclass
class Chunk:
    src: str  # key of the source file
    offset: int  # absolute file offset
    nbytes: int


@dataclass
class OutTensor:
    name: str
    dims: list[int]
    type: int
    chunks: list[Chunk]

    @property
    def nbytes(self) -> int:
        return sum(c.nbytes for c in self.chunks)


def per_expert_layer(name: str) -> int | None:
    """The layer of a tensor with one slice per expert on its last axis, else None."""
    for pattern in (EXPERT_WEIGHT, ROUTER, IMATRIX_EXPERT):
        if m := pattern.match(name):
            return int(m.group(1))
    return None


def whole(layout: Layout, src: str, name: str) -> OutTensor:
    t = layout.by_name[name]
    return OutTensor(t.name, list(t.dims), t.type, [Chunk(src, layout.data_start + t.offset, t.nbytes)])


def sliced(layout: Layout, src: str, name: str, keep: list[int]) -> OutTensor:
    """The experts `keep` of a per-expert tensor, in that order; consecutive experts become one range."""
    t = layout.by_name[name]
    n = t.dims[-1]
    if t.nbytes % n:
        raise ValueError(f"{name}: {t.nbytes} bytes do not split into {n} experts")
    if not keep or min(keep) < 0 or max(keep) >= n or len(set(keep)) != len(keep):
        raise ValueError(f"{name}: keep list out of range 0..{n - 1} or with duplicates")
    per = t.nbytes // n
    base = layout.data_start + t.offset
    chunks: list[Chunk] = []
    for e in keep:
        if chunks and chunks[-1].offset + chunks[-1].nbytes == base + e * per:
            chunks[-1].nbytes += per
        else:
            chunks.append(Chunk(src, base + e * per, per))
    return OutTensor(t.name, [*t.dims[:-1], len(keep)], t.type, chunks)


def plan_slice(layout: Layout, src: str, keep: dict[int, list[int]]) -> list[OutTensor]:
    """All tensors of one file in file order; the per-expert ones cut to keep[layer]. All per-expert tensors must
    have the same last axis: a tensor laid out otherwise (e.g. flattened) would be cut silently wrong."""
    axes = {t.dims[-1] for t in layout.tensors if per_expert_layer(t.name) is not None}
    if len(axes) > 1:
        raise ValueError(f"per-expert tensors with different last axes {sorted(axes)}")
    out = []
    for t in layout.tensors:
        layer = per_expert_layer(t.name)
        if layer is None:
            out.append(whole(layout, src, t.name))
        elif layer not in keep:
            raise ValueError(f"{t.name}: no keep list for layer {layer}")
        else:
            out.append(sliced(layout, src, t.name, keep[layer]))
    return out


def plan_assemble(coder: Layout, quant: Layout, keep: dict[int, list[int]]) -> list[OutTensor]:
    """The Coder file with its expert weights taken from `quant` (already cut to keep) and its router cut to keep;
    every other tensor is a byte copy. Sources are named "coder" and "quant"."""
    out = []
    for t in coder.tensors:
        if EXPERT_WEIGHT.match(t.name):
            q = quant.by_name.get(t.name)
            layer = int(EXPERT_WEIGHT.match(t.name).group(1))
            if q is None or q.dims != [*t.dims[:-1], len(keep[layer])]:
                raise ValueError(f"{t.name}: quantized file has {q and q.dims}, expected {t.dims[:-1]} x E")
            out.append(whole(quant, "quant", t.name))
        elif m := ROUTER.match(t.name):
            out.append(sliced(coder, "coder", t.name, keep[int(m.group(1))]))
        else:
            out.append(whole(coder, "coder", t.name))
    return out


def equal_count(keep: dict[int, list[int]]) -> int:
    """GGUF has one expert_count for all layers; unequal lists have no loader (llama.cpp, Tona)."""
    counts = {len(v) for v in keep.values()}
    if len(counts) != 1:
        raise ValueError(f"keep lists of unequal length {sorted(counts)}: no loader reads per-layer expert counts")
    return counts.pop()


def model_kvs(layout: Layout, n_expert: int) -> list[KV]:
    return with_scalar(layout.kvs, arch_key(layout.kvs, "expert_count"), n_expert)


def drop_split_kvs(kvs: list[KV]) -> list[KV]:
    """A single output file from split sources carries no split.* keys."""
    return [kv for kv in kvs if not kv.key.startswith("split.")]


def _string(s: str) -> bytes:
    raw = s.encode()
    return struct.pack("<Q", len(raw)) + raw


def header(kvs: list[KV], tensors: list[OutTensor], alignment: int) -> bytes:
    head = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)))
    for kv in kvs:
        head += _string(kv.key) + struct.pack("<I", kv.vtype) + kv.raw
    offset = 0
    for t in tensors:
        head += _string(t.name) + struct.pack("<I", len(t.dims)) + struct.pack(f"<{len(t.dims)}Q", *t.dims)
        head += struct.pack("<IQ", t.type, offset)
        offset += t.nbytes + (-t.nbytes % alignment)
    return bytes(head + bytes(-len(head) % alignment))


@dataclass
class Hooks:
    """Called before the first and after the last tensor of each source, for shard-wise download and deletion."""

    path: Callable[[str], str]  # source key -> file path
    before: Callable[[str], None] = lambda src: None
    after: Callable[[str], None] = lambda src: None


def _copy(dst, path: str, chunk: Chunk) -> None:
    with open(path, "rb") as f:
        f.seek(chunk.offset)
        left = chunk.nbytes
        while left:
            data = f.read(min(PIECE, left))
            if not data:
                raise ValueError(f"{chunk.src}: file ends at {f.tell()}, inside a tensor")
            dst.write(data)
            left -= len(data)


def write(path: str, kvs: list[KV], tensors: list[OutTensor], hooks: Hooks, alignment: int = 32) -> None:
    """Writes the file; tensors are copied in order, hooks.before runs before a source's first chunk and hooks.after
    after its last."""
    last = {c.src: i for i, t in enumerate(tensors) for c in t.chunks}
    seen: set[str] = set()
    with open(path, "wb") as dst:
        dst.write(header(kvs, tensors, alignment))
        for i, t in enumerate(tensors):
            for c in t.chunks:
                if c.src not in seen:
                    hooks.before(c.src)
                    seen.add(c.src)
                _copy(dst, hooks.path(c.src), c)
            dst.write(bytes(-t.nbytes % alignment))
            for src in [s for s, at in last.items() if at == i]:
                hooks.after(src)
