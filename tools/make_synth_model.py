#!/usr/bin/env python3
"""Writes a synthetic model in full size: the shapes of the Coder (Qwen3.8-Flash-Next) with random weights, for speed
measurements of a backend before the real model of a variant exists. Types per role as in the Coder, except the
routed experts, which take the types of the variant. Differences to the real model, to keep it resident beside other
work: --experts (per-token bytes stay the same, 10 experts per token), and a small n-gram table (the real one has 320 M
rows, about 29 GB; a token reads 16 rows of it either way). Random weights route uniformly over the experts.

Usage: tools/make_synth_model.py [--source TARGET.gguf] [--experts N] [--variant a] [--out DIR] [--seed N]
Standard library only; about 13 GB for 128 experts. Never for quality (docs/contribute/testing.md).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from make_mini_model import DEFAULT_SOURCE, MODELS, build_kvs, read_source
from mini_model import blocks, fast, gguf_io, shapes, tensors

# Expert types of the compression variants (research/42): gate and up share theirs
VARIANTS = {"a": {"ffn_gate_exps.weight": [16], "ffn_down_exps.weight": [42]}}  # IQ2_XXS, Q2_0


def primes_below(limit: int, count: int) -> tuple[int, ...]:
    out: list[int] = []
    n = limit - 1
    while len(out) < count:
        if all(n % p for p in range(2, int(n**0.5) + 1)):
            out.append(n)
        n -= 1
    return tuple(out)


def coder_shape(experts: int, variant: str) -> shapes.Shape:
    """The Coder's header values (qwen4exp.*); the rest of Shape's defaults already match it."""
    return shapes.Shape(n_layer=48, d=2560, hc_rank=320, dn_k=16, dn_v=48, idx_top_k=2048, n_expert=experts,
                        expert_ff=640, shexp_ff=640, ple_dim=160, ple_per_gram=8, ple_primes=primes_below(65536, 16),
                        name=f"Synthetic model, variant {variant.upper()} (random weights)",
                        size_label=f"synth-{variant}-e{experts}")  # fmt: skip


class FastBuilder(tensors.Builder):
    @staticmethod
    def _data(ttype: int, n: int, fan_in: int, gen: tuple, rng) -> bytes:
        if gen[0] in ("g", "s"):
            sigma = blocks.sigma_of(fan_in, gen[1]) if gen[0] == "g" else gen[1]
            if ttype not in (blocks.F32, blocks.F16, blocks.BF16):
                return fast.quant_bytes(ttype, n, rng, sigma)
            if ttype == blocks.BF16 and n >= 1 << 16:
                return fast.bf16_bytes(rng, n, sigma)
        return tensors.Builder._data(ttype, n, fan_in, gen, rng)


def write_model(path: str, source: str, shape: shapes.Shape, variant: str, seed: int) -> int:
    kvs, src_tensors = read_source(source)
    types = tensors.role_types(src_tensors) | VARIANTS[variant]
    planned = tensors.target_tensors(shape, types, seed, FastBuilder)
    gguf_io.write_gguf(path, build_kvs(kvs, shapes.overrides(shape, shape.n_layer, False)), planned)
    return sum(t.nbytes for t in planned)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--variant", default="a", choices=sorted(VARIANTS))
    ap.add_argument("--out", default=os.path.join(MODELS, "synth"))
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    shape = coder_shape(a.experts, a.variant)
    path = os.path.join(a.out, f"{shape.size_label}.gguf")
    size = write_model(path, a.source, shape, a.variant, a.seed)
    print(f"{path}: {size / 1e9:.2f} GB of tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
