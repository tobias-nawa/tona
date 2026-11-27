"""The mini Ornith (GGUF architecture qwen35moe): shapes, key overrides and tensor list.

Same idea as the Coder's mini model (shapes.py, tensors.py): the real file's tokenizer, chat template and quant type per
tensor role, random weights, few layers and experts. Ornith has one residual stream, so there is no gated residual and
no PLE; the MTP layer lives in the same file (blk.<n_layer>). Names and shape rules: engine/model/qwen35moe.cpp.
The expert width is 256 (the real 512): the Q4_K and Q6_K blocks of the down projection need rows of 256 inputs.
"""

from dataclasses import dataclass

from . import tensors
from .gguf_io import OutTensor
from .shapes import DROPPED_PREFIXES  # noqa: F401  (re-exported: the same keys are dropped)

PREFIX = "qwen35moe."


@dataclass(frozen=True)
class Shape:
    vocab: int = 248320  # as the reference: tokenizer and chat tokens stay the same
    n_layer: int = 8  # trunk layers; the MTP layer comes on top
    d: int = 256
    n_head: int = 16
    n_kv: int = 2
    head_dim: int = 256
    dn_k: int = 2
    dn_v: int = 4
    dn_head: int = 128
    dn_conv: int = 4
    full_interval: int = 4
    n_expert: int = 32
    n_used: int = 8
    expert_ff: int = 256
    shexp_ff: int = 256
    name: str = "Mini Ornith (random weights)"
    size_label: str = "mini"

    @property
    def dn_kd(self) -> int:
        return self.dn_k * self.dn_head

    @property
    def dn_vd(self) -> int:
        return self.dn_v * self.dn_head

    def is_attn(self, layer: int) -> bool:
        return (layer + 1) % self.full_interval == 0


def overrides(s: Shape) -> dict:
    """Values that replace those of the source file; the block count includes the MTP layer."""
    return {
        PREFIX + "block_count": s.n_layer + 1,
        PREFIX + "embedding_length": s.d,
        PREFIX + "attention.head_count": s.n_head,
        PREFIX + "attention.head_count_kv": s.n_kv,
        PREFIX + "expert_count": s.n_expert,
        PREFIX + "expert_used_count": s.n_used,
        PREFIX + "expert_feed_forward_length": s.expert_ff,
        PREFIX + "expert_shared_feed_forward_length": s.shexp_ff,
        PREFIX + "ssm.group_count": s.dn_k,
        PREFIX + "ssm.time_step_rank": s.dn_v,
        PREFIX + "ssm.inner_size": s.dn_vd,
        "general.name": s.name,
        "general.size_label": s.size_label,
    }


def attention(b: tensors.Builder, s: Shape, p: str, layer: int) -> None:
    d, hd, gain = s.d, s.head_dim, tensors.GAIN
    b.add(p + "attn_q.weight", [d, 2 * s.n_head * hd], ("g", gain["proj_in"]), layer)
    b.add(p + "attn_k.weight", [d, s.n_kv * hd], ("g", gain["proj_in"]), layer)
    b.add(p + "attn_v.weight", [d, s.n_kv * hd], ("g", gain["proj_in"]), layer)
    b.add(p + "attn_output.weight", [s.n_head * hd, d], ("g", gain["attn_out"]), layer)
    b.add(p + "attn_q_norm.weight", [hd], ("norm",), layer)
    b.add(p + "attn_k_norm.weight", [hd], ("norm",), layer)


def block_head(b: tensors.Builder, s: Shape, p: str, layer: int) -> None:
    b.add(p + "attn_norm.weight", [s.d], ("norm",), layer)
    b.add(p + "post_attention_norm.weight", [s.d], ("norm",), layer)


def block(b: tensors.Builder, s: Shape, layer: int) -> None:
    p = f"blk.{layer}."
    block_head(b, s, p, layer)
    (attention if s.is_attn(layer) else tensors.deltanet)(b, s, p, layer)
    tensors.moe(b, s, p, layer, s.n_expert)


def mtp_block(b: tensors.Builder, s: Shape) -> None:
    """blk.<n_layer>: a gated attention layer with MoE, preceded by eh_proj over [enorm(e) | hnorm(h)]."""
    layer, p = s.n_layer, f"blk.{s.n_layer}."
    block_head(b, s, p, layer)
    attention(b, s, p, layer)
    tensors.moe(b, s, p, layer, s.n_expert)
    b.add(p + "nextn.eh_proj.weight", [2 * s.d, s.d], ("g", tensors.GAIN["proj_in"]), layer)
    for name in ("enorm", "hnorm", "shared_head_norm"):
        b.add(p + f"nextn.{name}.weight", [s.d], ("norm",), layer)


def model_tensors(s: Shape, types: dict[str, list[int]], seed: int) -> list[OutTensor]:
    b = tensors.Builder(types, seed)
    b.add("token_embd.weight", [s.d, s.vocab], ("s", 1.0))
    b.add("output_norm.weight", [s.d], ("norm",))
    b.add("output.weight", [s.d, s.vocab], ("g", tensors.GAIN["output"]))
    for layer in range(s.n_layer):
        block(b, s, layer)
    mtp_block(b, s)
    return b.out
