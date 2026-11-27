"""GGUF header with tensor offsets and sizes, standard library only.

The header parser of tools/mini_model keeps no offsets; slicing and merging need them, and the data start, to copy
byte ranges without loading tensors.
"""

import struct
from dataclasses import dataclass, field

from mini_model.gguf_io import ARR, KV, SCALAR_FMT, _skip_value

DEFAULT_ALIGNMENT = 32  # GGUF_DEFAULT_ALIGNMENT, ggml/include/gguf.h

# ggml type: (elements per block, bytes per block), from ggml.c type_traits (b11822)
TYPE_SIZE = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 6: (32, 22), 8: (32, 34), 10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176),
    14: (256, 210), 16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110),
    22: (256, 82), 23: (256, 136), 29: (256, 56), 30: (1, 2), 42: (64, 18),
    1018: (512, 196),  # TONA_IQ3_XXS_R, Tona's own (docs/use/tona-types.md)
    1019: (256, 210),  # TONA_Q6_K_T, Tona's own, 8 rows interleaved (docs/use/tona-types.md)
}  # fmt: skip


class Truncated(Exception):
    """The buffer ends inside the header; read more bytes and parse again."""


@dataclass
class Tensor:
    name: str
    dims: list[int]
    type: int
    offset: int  # relative to the data start
    nbytes: int
    type_pos: int = -1  # file position of the tensor's uint32 type field (repack.py patches it)


@dataclass
class Layout:
    kvs: list[KV]
    tensors: list[Tensor]
    data_start: int
    alignment: int
    by_name: dict[str, Tensor] = field(default_factory=dict)

    def __post_init__(self):
        self.by_name = {t.name: t for t in self.tensors}


def tensor_bytes(ttype: int, dims: list[int]) -> int:
    if ttype not in TYPE_SIZE:
        raise ValueError(f"ggml type {ttype} has no size entry")
    per_block, block_bytes = TYPE_SIZE[ttype]
    n = 1
    for d in dims:
        n *= d
    if n % per_block:
        raise ValueError(f"{n} elements are no whole number of type {ttype} blocks")
    return n // per_block * block_bytes


def scalar(kvs: list[KV], key: str):
    for kv in kvs:
        if kv.key == key:
            return struct.unpack_from(SCALAR_FMT[kv.vtype], kv.raw)[0]
    return None


def with_scalar(kvs: list[KV], key: str, value) -> list[KV]:
    """A copy of kvs with key set to value in the key's own value type; the key must exist."""
    out, found = [], False
    for kv in kvs:
        if kv.key == key:
            kv = KV(kv.key, kv.vtype, kv.elem, struct.pack(SCALAR_FMT[kv.vtype], value))
            found = True
        out.append(kv)
    if not found:
        raise KeyError(key)
    return out


def arch_key(kvs: list[KV], suffix: str) -> str:
    """The architecture-prefixed key, e.g. qwen4exp.expert_count for suffix expert_count."""
    for kv in kvs:
        if kv.key == "general.architecture":
            return f"{kv.raw[8:].decode()}.{suffix}"
    raise KeyError("general.architecture")


def _need(buf, pos: int, n: int) -> None:
    if pos + n > len(buf):
        raise Truncated


def _read_kvs(buf, pos: int, n_kv: int) -> tuple[list[KV], int]:
    kvs = []
    for _ in range(n_kv):
        _need(buf, pos, 8)
        klen = struct.unpack_from("<Q", buf, pos)[0]
        _need(buf, pos, 12 + klen)
        key = bytes(buf[pos + 8 : pos + 8 + klen]).decode()
        pos += 8 + klen
        vtype = struct.unpack_from("<I", buf, pos)[0]
        pos += 4
        try:
            end = _skip_value(buf, pos, vtype)
            elem = struct.unpack_from("<I", buf, pos)[0] if vtype == ARR else -1
        except struct.error as e:
            raise Truncated from e
        _need(buf, pos, end - pos)
        kvs.append(KV(key, vtype, elem, bytes(buf[pos:end])))
        pos = end
    return kvs, pos


def _read_tensors(buf, pos: int, n_tensors: int) -> tuple[list[Tensor], int]:
    tensors = []
    for _ in range(n_tensors):
        _need(buf, pos, 8)
        nlen = struct.unpack_from("<Q", buf, pos)[0]
        _need(buf, pos, 12 + nlen)
        name = bytes(buf[pos + 8 : pos + 8 + nlen]).decode()
        pos += 8 + nlen
        nd = struct.unpack_from("<I", buf, pos)[0]
        _need(buf, pos, 4 + 8 * nd + 12)
        dims = list(struct.unpack_from(f"<{nd}Q", buf, pos + 4))
        pos += 4 + 8 * nd
        ttype, offset = struct.unpack_from("<IQ", buf, pos)
        tensors.append(Tensor(name, dims, ttype, offset, tensor_bytes(ttype, dims), pos))
        pos += 12
    return tensors, pos


def parse(buf) -> Layout:
    """Parses a header from the start of a file (bytes, mmap); raises Truncated if buf ends inside it."""
    _need(buf, 0, 24)
    if bytes(buf[:4]) != b"GGUF":
        raise ValueError("not a GGUF file")
    version, n_tensors, n_kv = struct.unpack_from("<IQQ", buf, 4)
    if version != 3:
        raise ValueError(f"GGUF version {version}, expected 3")
    kvs, pos = _read_kvs(buf, 24, n_kv)
    tensors, pos = _read_tensors(buf, pos, n_tensors)
    alignment = scalar(kvs, "general.alignment") or DEFAULT_ALIGNMENT
    return Layout(kvs, tensors, pos + (-pos % alignment), alignment)


def expert_count(layout: Layout) -> int:
    return scalar(layout.kvs, arch_key(layout.kvs, "expert_count"))
