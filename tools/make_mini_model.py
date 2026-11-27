#!/usr/bin/env python3
"""Writes the mini model: the architecture and tensor names of the reference, random weights, a few layers and
experts. Tokenizer and chat template are copied byte for byte from the real model, the quant type of every tensor
role from its tensor directory. Same seed, same bytes. For host logic, determinism, stash and API tests, never for
speed or quality (docs/contribute/testing.md).

Usage: tools/make_mini_model.py --from-header data/mini-models/ID.header.gguf.xz [--out DIR] [--seed N]
           ID: coder (writes mini and mini-mtp, reads coder-mtp next to it), ornith, bolt; needs no big model
       tools/make_mini_model.py --export-header ID [--source MODEL.gguf]   writes data/mini-models/ID.header.gguf.xz
           ID: coder, coder-mtp, ornith, bolt; needs the big model in work/models/
       tools/make_mini_model.py [--source TARGET.gguf] [--mtp-source MTP.gguf] [--out DIR] [--seed N]
       tools/make_mini_model.py --arch qwen35moe [--source ORNITH.gguf] [--out DIR] [--seed N]   (mini Ornith)
       tools/make_mini_model.py --arch nemotron_h_moe [--source BOLT.gguf] [--out DIR] [--seed N]   (mini Bolt)
Standard library only.
"""

import argparse
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mini_model import gguf_io, headers, nemotron_h, qwen35moe, shapes, tensors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "work", "models")
DEFAULT_SOURCE = os.path.join(MODELS, "coder-iq1_m", "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf")
DEFAULT_MTP = os.path.join(MODELS, "draft", "mtp-Qwen3.8-Flash-Next-Q8_0.gguf")
DEFAULT_BOLT = os.path.join(MODELS, "nemotron-bolt-74b-a9b-24gb", "NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf")
DEFAULT_ORNITH = os.path.join(MODELS, "ornith-1.5-35b-a3b", "Ornith-1.5-35B-Q4_K_M.gguf")
DATA = os.path.join(ROOT, "data", "mini-models")
SOURCES = {"coder": DEFAULT_SOURCE, "coder-mtp": DEFAULT_MTP, "ornith": DEFAULT_ORNITH, "bolt": DEFAULT_BOLT}
ARCH_OF_ID = {"coder": "qwen4exp", "ornith": "qwen35moe", "bolt": "nemotron_h_moe"}
MTP_EXPERTS = 128  # differs from the target's 64 on purpose: the drafter has its own expert count


def build_kvs(source: list[gguf_io.KV], values: dict) -> list[gguf_io.KV]:
    missing = set(values) - {kv.key for kv in source}
    if missing:
        raise ValueError(f"keys missing in the source file: {sorted(missing)}")
    out = []
    for kv in source:
        if kv.key.startswith(shapes.DROPPED_PREFIXES):
            continue
        raw = gguf_io.encode_value(kv, values[kv.key]) if kv.key in values else kv.raw
        out.append(gguf_io.KV(kv.key, kv.vtype, kv.elem, raw))
    return out


def read_source(source: str) -> tuple[list[gguf_io.KV], list[gguf_io.SourceTensor]]:
    if source.endswith(headers.SUFFIX):
        return headers.load_header(source)
    kvs, src_tensors = gguf_io.read_header(source)
    if "-00001-of-0000" in source:  # split model: the PLE table lives in the later parts
        n = int(source.rsplit("-of-0000", 1)[1].split(".", maxsplit=1)[0])
        for part in range(2, n + 1):
            src_tensors += gguf_io.read_header(source.replace("-00001-of-", f"-0000{part}-of-"))[1]
    return kvs, src_tensors


def write_model(path: str, source: str, mtp: bool, shape: shapes.Shape, seed: int) -> None:
    kvs, src_tensors = read_source(source)
    types = tensors.role_types(src_tensors)
    n_blocks = shape.n_layer + (1 if mtp else 0)
    if mtp:
        planned = tensors.mtp_tensors(shape, types, seed, shape.n_expert)
    else:
        planned = tensors.target_tensors(shape, types, seed)
    gguf_io.write_gguf(path, build_kvs(kvs, shapes.overrides(shape, n_blocks, mtp)), planned)


def write_ornith(path: str, source: str, shape: qwen35moe.Shape, seed: int) -> None:
    kvs, src_tensors = read_source(source)
    types = tensors.role_types(src_tensors)
    gguf_io.write_gguf(path, build_kvs(kvs, qwen35moe.overrides(shape)), qwen35moe.model_tensors(shape, types, seed))


def write_bolt(path: str, source: str, shape: nemotron_h.Shape, seed: int, f32: bool = False) -> None:
    kvs, src_tensors = read_source(source)
    types = tensors.role_types(src_tensors)
    if f32:  # every tensor F32: the other engine then has no activation quantization to differ in
        types = {role: [0] for role in types}
    gguf_io.write_gguf(path, build_kvs(kvs, nemotron_h.overrides(shape)), nemotron_h.model_tensors(shape, types, seed))


def export_header(model_id: str, source: str | None, out: str | None) -> int:
    if model_id not in SOURCES:
        print(f"unknown model {model_id}; one of: {', '.join(SOURCES)}", file=sys.stderr)
        return 2
    dest = os.path.join(out or DATA, model_id + headers.SUFFIX)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    size = headers.export_header(*read_source(source or SOURCES[model_id]), dest)
    print(f"{dest}: {size} bytes")
    return 0


def header_id(path: str) -> str:
    name = os.path.basename(path)
    if not name.endswith(headers.SUFFIX) or name[: -len(headers.SUFFIX)] not in ARCH_OF_ID:
        raise SystemExit(f"{path}: expected one of {', '.join(i + headers.SUFFIX for i in ARCH_OF_ID)}")
    return name[: -len(headers.SUFFIX)]


def build_from_header(path: str, out: str | None, seed: int, f32: bool) -> int:
    """Builds the mini model of one header; the Coder also gets its MTP head from the header next to it."""
    model_id = header_id(path)
    if model_id == "ornith":
        out = out or os.path.join(MODELS, "mini-ornith")
        os.makedirs(out, exist_ok=True)
        write_ornith(os.path.join(out, "mini-ornith.gguf"), path, qwen35moe.Shape(), seed)
    elif model_id == "bolt":
        out = out or os.path.join(MODELS, "mini-bolt")
        os.makedirs(out, exist_ok=True)
        name = "mini-bolt-f32.gguf" if f32 else "mini-bolt.gguf"
        write_bolt(os.path.join(out, name), path, nemotron_h.Shape(), seed, f32)
    else:
        out = out or os.path.join(MODELS, "mini")
        mtp = os.path.join(os.path.dirname(path), "coder-mtp" + headers.SUFFIX)
        write_coder(out, path, mtp, seed)
    return 0


def write_coder(out: str, source: str, mtp_source: str, seed: int) -> None:
    os.makedirs(out, exist_ok=True)
    shape = shapes.Shape()
    write_model(os.path.join(out, "mini.gguf"), source, False, shape, seed)
    write_model(os.path.join(out, "mini-mtp.gguf"), mtp_source, True,
                dataclasses.replace(shape, n_expert=MTP_EXPERTS), seed)  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", choices=("qwen4exp", "qwen35moe", "nemotron_h_moe"), default="qwen4exp")
    ap.add_argument("--source", default=None, help="header of the real model (default: the architecture's)")
    ap.add_argument("--mtp-source", default=DEFAULT_MTP)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--export-header", metavar="ID", help="write data/mini-models/ID.header.gguf.xz (--out: folder)")
    ap.add_argument("--from-header", metavar="FILE", help="build from a committed header instead of the big model")
    ap.add_argument(
        "--f32", action="store_true", help="mini Bolt with F32 weights only (cross-checks against another engine)"
    )
    a = ap.parse_args()
    if a.export_header:
        return export_header(a.export_header, a.source, a.out)
    if a.from_header:
        return build_from_header(a.from_header, a.out, a.seed, a.f32)
    if a.arch == "qwen35moe":
        out = a.out or os.path.join(MODELS, "mini-ornith")
        os.makedirs(out, exist_ok=True)
        write_ornith(os.path.join(out, "mini-ornith.gguf"), a.source or DEFAULT_ORNITH, qwen35moe.Shape(), a.seed)
        return 0
    if a.arch == "nemotron_h_moe":
        out = a.out or os.path.join(MODELS, "mini-bolt")
        os.makedirs(out, exist_ok=True)
        name = "mini-bolt-f32.gguf" if a.f32 else "mini-bolt.gguf"
        write_bolt(os.path.join(out, name), a.source or DEFAULT_BOLT, nemotron_h.Shape(), a.seed, a.f32)
        return 0
    write_coder(a.out or os.path.join(MODELS, "mini"), a.source or DEFAULT_SOURCE, a.mtp_source, a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
