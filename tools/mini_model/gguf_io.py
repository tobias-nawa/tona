"""GGUF reading (raw key-value spans) and writing, standard library only.

The mini model copies the tokenizer and chat template byte for byte from the real model, so values are kept as raw
bytes; only the model keys are re-encoded, in the value type the source file uses.
"""

import mmap
import struct
from dataclasses import dataclass

U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
SCALAR_FMT = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i", F32: "<f", BOOL: "<?", U64: "<Q",
              I64: "<q", F64: "<d"}  # fmt: skip
ALIGNMENT = 32


@dataclass
class KV:
    key: str
    vtype: int
    elem: int  # element type for arrays, else -1
    raw: bytes  # value bytes without the type field


@dataclass
class SourceTensor:
    name: str
    dims: list[int]
    type: int


def _skip_value(buf, pos: int, vtype: int) -> int:
    if vtype in SCALAR_FMT:
        return pos + struct.calcsize(SCALAR_FMT[vtype])
    if vtype == STR:
        return pos + 8 + struct.unpack_from("<Q", buf, pos)[0]
    if vtype == ARR:
        elem, n = struct.unpack_from("<IQ", buf, pos)
        pos += 12
        if elem in SCALAR_FMT:
            return pos + n * struct.calcsize(SCALAR_FMT[elem])
        for _ in range(n):
            pos = _skip_value(buf, pos, elem)
        return pos
    raise ValueError(f"unknown GGUF value type {vtype}")


def parse_header(buf) -> tuple[list[KV], list[SourceTensor]]:
    """Key-value pairs and tensor directory of a GGUF file or of its header alone (the data is never read)."""
    if buf[:4] != b"GGUF":
        raise ValueError("not a GGUF file")
    _, n_tensors, n_kv = struct.unpack_from("<IQQ", buf, 4)
    pos = 24
    kvs = []
    for _ in range(n_kv):
        klen = struct.unpack_from("<Q", buf, pos)[0]
        key = bytes(buf[pos + 8 : pos + 8 + klen]).decode()
        pos += 8 + klen
        vtype = struct.unpack_from("<I", buf, pos)[0]
        pos += 4
        end = _skip_value(buf, pos, vtype)
        elem = struct.unpack_from("<I", buf, pos)[0] if vtype == ARR else -1
        kvs.append(KV(key, vtype, elem, bytes(buf[pos:end])))
        pos = end
    tensors = []
    for _ in range(n_tensors):
        nlen = struct.unpack_from("<Q", buf, pos)[0]
        name = bytes(buf[pos + 8 : pos + 8 + nlen]).decode()
        pos += 8 + nlen
        nd = struct.unpack_from("<I", buf, pos)[0]
        dims = list(struct.unpack_from(f"<{nd}Q", buf, pos + 4))
        pos += 4 + 8 * nd
        ttype = struct.unpack_from("<I", buf, pos)[0]
        pos += 12  # type and offset
        tensors.append(SourceTensor(name, dims, ttype))
    return kvs, tensors


def read_header(path: str) -> tuple[list[KV], list[SourceTensor]]:
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as buf:
        try:
            return parse_header(buf)
        except ValueError as e:
            raise ValueError(f"{path}: {e}") from None


def header_bytes(kvs: list[KV], tensors: list[SourceTensor]) -> bytes:
    """A GGUF file without tensor data: every offset is 0, the loaders of this repository never follow them."""
    head = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)))
    for kv in kvs:
        head += _string(kv.key) + struct.pack("<I", kv.vtype) + kv.raw
    for t in tensors:
        head += _string(t.name) + struct.pack("<I", len(t.dims)) + struct.pack(f"<{len(t.dims)}Q", *t.dims)
        head += struct.pack("<IQ", t.type, 0)
    return bytes(head)


def encode_scalar(vtype: int, value) -> bytes:
    if vtype == STR:
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw
    return struct.pack(SCALAR_FMT[vtype], value)


def encode_value(src: KV, value) -> bytes:
    """Re-encodes a value in the type the source file uses for this key."""
    if src.vtype != ARR:
        return encode_scalar(src.vtype, value)
    items = b"".join(encode_scalar(src.elem, v) for v in value)
    return struct.pack("<IQ", src.elem, len(value)) + items


def _string(s: str) -> bytes:
    raw = s.encode()
    return struct.pack("<Q", len(raw)) + raw


@dataclass
class OutTensor:
    name: str
    dims: list[int]
    type: int
    nbytes: int
    make: object  # callable returning the tensor's bytes


def write_gguf(path: str, kvs: list[KV], tensors: list[OutTensor]) -> None:
    head = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)))
    for kv in kvs:
        head += _string(kv.key) + struct.pack("<I", kv.vtype) + kv.raw
    offset = 0
    offsets = []
    for t in tensors:
        offsets.append(offset)
        offset += -(-t.nbytes // ALIGNMENT) * ALIGNMENT
        head += _string(t.name) + struct.pack("<I", len(t.dims))
        head += struct.pack(f"<{len(t.dims)}Q", *t.dims) + struct.pack("<IQ", t.type, offsets[-1])
    head += bytes(-len(head) % ALIGNMENT)
    with open(path, "wb") as f:
        f.write(head)
        for t in tensors:
            data = t.make()
            if len(data) != t.nbytes:
                raise ValueError(f"{t.name}: {len(data)} bytes, expected {t.nbytes}")
            f.write(data + bytes(-len(data) % ALIGNMENT))
