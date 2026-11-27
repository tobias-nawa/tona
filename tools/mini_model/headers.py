"""Committed headers of the real models: what the mini model needs from them, xz-compressed (data/mini-models/).

A header is the key-value block (tokenizer, chat template, metadata) and the tensor directory of the real model, as a
GGUF file without tensor data. A split model's header lists the tensors of every part. The keys that name the
license and the origin (`general.license*`, `general.repo_url`, `general.quantized_by`) stay, because the licenses ask
that notices of origin travel with copies. The quantizer's build paths (`quantize.*`, `split.*`) go: they are no
notice and name people and machines.
"""

import lzma

from . import gguf_io

SUFFIX = ".header.gguf.xz"
BUILD_KEYS = ("quantize.", "split.")
# The preset is fixed so that an export of the same model gives the same bytes on every machine with the same liblzma
PRESET = 9 | lzma.PRESET_EXTREME


def export_header(kvs: list[gguf_io.KV], tensors: list[gguf_io.SourceTensor], dest: str) -> int:
    """Writes the header file and returns its size in bytes."""
    kept = [kv for kv in kvs if not kv.key.startswith(BUILD_KEYS)]
    packed = lzma.compress(gguf_io.header_bytes(kept, tensors), format=lzma.FORMAT_XZ, preset=PRESET)
    with open(dest, "wb") as f:
        f.write(packed)
    return len(packed)


def load_header(path: str) -> tuple[list[gguf_io.KV], list[gguf_io.SourceTensor]]:
    with lzma.open(path, "rb") as f:
        try:
            return gguf_io.parse_header(f.read())
        except ValueError as e:
            raise ValueError(f"{path}: {e}") from None
