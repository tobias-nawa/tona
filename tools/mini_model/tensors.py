"""Tensor list of the mini model: names and shape rules of engine/model/qwen4exp.cpp, types from the source file.

A tensor is described by its generator: ("g", gain) is a matrix with weights of rms gain / sqrt(fan-in), so that
a unit-rms input gives an output of rms gain; ("s", sigma) fixes the rms; ("norm",) a norm weight around 1;
("uni", lo, hi) uniform values. The gains keep activations in a range where no kernel overflows and the router and
the logits still separate (checked on the CPU reference, docs/contribute/testing.md).
"""

import math
import re
from collections import Counter

from . import blocks
from .gguf_io import OutTensor, SourceTensor
from .shapes import Shape, ple_rows

GAIN = {
    "router": 4.0, "exps_in": 2.0, "exps_out": 1.0, "shexp_in": 2.0, "shexp_out": 1.0, "gate_inp_shexp": 1.0,
    "proj_in": 1.0, "proj_out": 1.0, "attn_out": 2.0, "hc": 1.0, "output": 15.0, "ple": 1.0,
}  # fmt: skip


def role_types(tensors: list[SourceTensor]) -> dict[str, list[int]]:
    """Types per role (name without block number), most frequent first, so a layer index can pick one."""
    count: dict[str, Counter] = {}
    for t in tensors:
        count.setdefault(re.sub(r"^blk\.\d+\.", "", t.name), Counter())[t.type] += 1
    return {r: [ty for ty, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))] for r, c in count.items()}


class Builder:
    def __init__(self, types: dict[str, list[int]], seed: int):
        self.types = types
        self.seed = seed
        self.out: list[OutTensor] = []
        self.chosen: dict[tuple[str, int], int] = {}
        self.used: Counter = Counter()  # tensors per role so far: the types cycle over the layers that have the role

    def add(self, name: str, dims: list[int], gen: tuple, layer: int = 0, role: str | None = None) -> None:
        role = role or re.sub(r"^blk\.\d+\.", "", name)
        elements = math.prod(dims)
        # a type whose block does not divide the row cannot be used here (down projections have 128 inputs)
        usable = [t for t in self.types[role] if dims[0] % blocks.TYPES[t][1] == 0]
        if not usable:
            raise ValueError(f"{name}: no type of {role} fits a row of {dims[0]}")
        if (role, layer) not in self.chosen:  # gate and up of a layer share their role and so their type
            self.chosen[(role, layer)] = usable[self.used[role] % len(usable)]
            self.used[role] += 1
        ttype = self.chosen[(role, layer)]
        rng = blocks.seeded(self.seed, name)
        self.out.append(OutTensor(name, dims, ttype, blocks.nbytes(ttype, elements),
                                  lambda: self._data(ttype, elements, dims[0], gen, rng)))  # fmt: skip

    @staticmethod
    def _data(ttype: int, n: int, fan_in: int, gen: tuple, rng) -> bytes:
        if gen[0] == "norm":
            values = [1.0 + v for v in blocks.gaussian(rng, n, 0.02)]
        elif gen[0] == "uni":
            values = blocks.uniform(rng, n, gen[1], gen[2])
        else:
            sigma = blocks.sigma_of(fan_in, gen[1]) if gen[0] == "g" else gen[1]
            if ttype in (blocks.F32, blocks.F16, blocks.BF16):
                values = blocks.gaussian(rng, n, sigma)
            else:
                return blocks.quant_bytes(ttype, n, rng, sigma)
        return blocks.float_bytes(ttype, values)


def gated_residual(b: Builder, s: Shape, pre: str, kind: str, layer: int, name: str | None = None) -> None:
    n = name or f"{pre}hc_{kind}_"
    b.add(n + "norm.weight", [s.hc], ("norm",), layer)
    b.add(n + "down.weight", [s.hc, s.hc_rank], ("g", GAIN["hc"]), layer)
    b.add(n + "up.weight", [s.hc_rank, s.hc], ("g", GAIN["hc"]), layer)
    if kind != "head":
        b.add(n + "inject.weight", [s.hc, s.n_hc], ("g", GAIN["hc"]), layer)


def qsa(b: Builder, s: Shape, p: str, layer: int) -> None:
    d, hd = s.d, s.head_dim
    b.add(p + "attn_q.weight", [d, 2 * s.n_head * hd], ("g", GAIN["proj_in"]), layer)
    b.add(p + "attn_k.weight", [d, s.n_kv * hd], ("g", GAIN["proj_in"]), layer)
    b.add(p + "attn_v.weight", [d, s.n_kv * hd], ("g", GAIN["proj_in"]), layer)
    b.add(p + "attn_output.weight", [s.n_head * hd, d], ("g", GAIN["attn_out"]), layer)
    b.add(p + "attn_q_norm.weight", [hd], ("norm",), layer)
    b.add(p + "attn_k_norm.weight", [hd], ("norm",), layer)
    b.add(p + "indexer.q_proj.weight", [d, s.idx_heads * s.idx_dim], ("g", GAIN["proj_in"]), layer)
    b.add(p + "indexer.k_proj.weight", [d, s.idx_dim], ("g", GAIN["proj_in"]), layer)
    b.add(p + "indexer.q_norm.weight", [s.idx_dim], ("norm",), layer)
    b.add(p + "indexer.k_norm.weight", [s.idx_dim], ("norm",), layer)


def deltanet(b: Builder, s: Shape, p: str, layer: int) -> None:
    d, g = s.d, ("g", GAIN["proj_in"])
    b.add(p + "attn_qkv.weight", [d, 2 * s.dn_kd + s.dn_vd], g, layer)
    b.add(p + "attn_gate.weight", [d, s.dn_vd], g, layer)
    b.add(p + "ssm_beta.weight", [d, s.dn_v], g, layer)
    b.add(p + "ssm_alpha.weight", [d, s.dn_v], g, layer)
    b.add(p + "ssm_conv1d.weight", [s.dn_conv, 2 * s.dn_kd + s.dn_vd], ("s", 0.5), layer)
    b.add(p + "ssm_dt.bias", [s.dn_v], ("uni", -1.0, 0.5), layer)
    b.add(p + "ssm_a", [s.dn_v], ("uni", -3.0, -0.1), layer)  # negative: the state decays
    b.add(p + "ssm_norm.weight", [s.dn_head], ("norm",), layer)
    b.add(p + "ssm_out.weight", [s.dn_vd, d], ("g", GAIN["proj_out"]), layer)


def moe(b: Builder, s: Shape, p: str, layer: int, n_expert: int) -> None:
    d, ff, sff = s.d, s.expert_ff, s.shexp_ff
    b.add(p + "ffn_gate_inp.weight", [d, n_expert], ("g", GAIN["router"]), layer)
    b.add(p + "ffn_gate_exps.weight", [d, ff, n_expert], ("g", GAIN["exps_in"]), layer)
    # gate and up share the type of a layer, as in the real model
    b.add(p + "ffn_up_exps.weight", [d, ff, n_expert], ("g", GAIN["exps_in"]), layer, "ffn_gate_exps.weight")
    b.add(p + "ffn_down_exps.weight", [ff, d, n_expert], ("g", GAIN["exps_out"]), layer)
    b.add(p + "ffn_gate_inp_shexp.weight", [d], ("g", GAIN["gate_inp_shexp"]), layer)
    b.add(p + "ffn_gate_shexp.weight", [d, sff], ("g", GAIN["shexp_in"]), layer)
    b.add(p + "ffn_up_shexp.weight", [d, sff], ("g", GAIN["shexp_in"]), layer)
    b.add(p + "ffn_down_shexp.weight", [sff, d], ("g", GAIN["shexp_out"]), layer)


def ple(b: Builder, s: Shape, p: str, layer: int) -> None:
    d, hc = s.d, s.hc
    b.add(p + "ple_key.weight", [d, hc], ("g", GAIN["ple"]), layer)
    b.add(p + "ple_value.weight", [d, d], ("g", GAIN["ple"]), layer)
    b.add(p + "ple_conv1d.weight", [s.ple_conv, hc], ("s", 0.5), layer)
    for n in ("key", "query", "conv"):
        b.add(p + f"ple_norm_{n}.weight", [hc], ("norm",), layer)


def block(b: Builder, s: Shape, layer: int, n_expert: int) -> None:
    p = f"blk.{layer}."
    gated_residual(b, s, p, "attn", layer)
    gated_residual(b, s, p, "ffn", layer)
    (qsa if s.is_qsa(layer) else deltanet)(b, s, p, layer)
    moe(b, s, p, layer, n_expert)
    if layer == s.ple_layer:
        ple(b, s, p, layer)


def target_tensors(s: Shape, types: dict[str, list[int]], seed: int, builder: type = Builder) -> list[OutTensor]:
    b = builder(types, seed)
    b.add("token_embd.weight", [s.d, s.vocab], ("s", 1.0))
    b.add("output.weight", [s.d, s.vocab], ("g", GAIN["output"]))
    gated_residual(b, s, "", "head", 0, "output_hc_")
    b.add("per_layer_token_embd.weight", [s.ple_dim, ple_rows(s)], ("s", 1.0))
    for layer in range(s.n_layer):
        block(b, s, layer, s.n_expert)
    return b.out


def mtp_tensors(s: Shape, types: dict[str, list[int]], seed: int, n_expert: int) -> list[OutTensor]:
    """The MTP block; embedding and LM head of the target are used, so the file carries none."""
    b = Builder(types, seed)
    layer, p = s.n_layer, f"blk.{s.n_layer}."
    gated_residual(b, s, p, "attn", layer)
    gated_residual(b, s, p, "ffn", layer)
    qsa(b, s, p, layer)
    moe(b, s, p, layer, n_expert)
    b.add(p + "nextn.eh_proj.weight", [2 * s.d, s.d], ("g", GAIN["proj_in"]), layer)
    b.add(p + "nextn.enorm.weight", [s.d], ("norm",), layer)
    b.add(p + "nextn.hnorm.weight", [s.hc], ("norm",), layer)
    gated_residual(b, s, p, "head", layer, p + "nextn.hc_head_")
    return b.out
