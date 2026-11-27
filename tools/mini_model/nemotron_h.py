"""The mini Bolt (GGUF architecture nemotron_h_moe): shapes, key overrides and tensor list.

Same idea as the other mini models: the real file's tokenizer, chat template and quant type per tensor role (Mamba-2
projections NVFP4, experts IQ2_XXS, attention and latent projections Q8_0), random weights, few and uneven layers:
the MoE blocks differ in expert count, width and top-k, as in the real file, so a reader that assumes one geometry
for all layers fails on it. Names and shape rules: engine/model/nemotron_h_moe.cpp. Widths are the smallest that the
block sizes allow (IQ2_XXS rows of 256, NVFP4 rows of 64, Q8_0 rows of 32).
"""

from dataclasses import dataclass, field

from . import tensors
from .gguf_io import OutTensor
from .shapes import DROPPED_PREFIXES  # noqa: F401  (re-exported: the same keys are dropped)

PREFIX = "nemotron_h_moe."


@dataclass(frozen=True)
class Shape:
    vocab: int = 131072  # as the reference: tokenizer and chat tokens stay the same
    d: int = 256
    # One letter per block: m Mamba-2, e latent MoE, a attention (the real file alternates them like this)
    pattern: str = "memeamem"
    n_head: int = 4
    n_kv: int = 2
    head_dim: int = 128
    ssm_heads: int = 8
    ssm_head_dim: int = 64
    ssm_groups: int = 2
    ssm_state: int = 128
    ssm_conv: int = 4
    latent: int = 256
    shexp_ff: int = 512
    # Per MoE block, in order: experts, top-k, expert width
    experts: tuple[int, ...] = field(default=(32, 24, 40))
    used: tuple[int, ...] = field(default=(4, 3, 5))
    expert_ff: tuple[int, ...] = field(default=(256, 512, 256))
    scale: float = 5.0
    name: str = "Mini Bolt (random weights)"
    size_label: str = "mini"

    @property
    def n_layer(self) -> int:
        return len(self.pattern)

    @property
    def inner(self) -> int:
        return self.ssm_heads * self.ssm_head_dim

    @property
    def conv_dim(self) -> int:
        return self.inner + 2 * self.ssm_groups * self.ssm_state

    @property
    def in_proj(self) -> int:
        return 2 * self.inner + 2 * self.ssm_groups * self.ssm_state + self.ssm_heads

    def moe_index(self, layer: int) -> int:
        return self.pattern[:layer].count("e")


def _per_layer(s: Shape, values: tuple[int, ...], kind: str, fill: int = 0) -> list[int]:
    return [values[s.moe_index(i)] if k == kind else fill for i, k in enumerate(s.pattern)]


def overrides(s: Shape) -> dict:
    """Values that replace those of the source file."""
    return {
        PREFIX + "block_count": s.n_layer,
        PREFIX + "embedding_length": s.d,
        PREFIX + "attention.head_count": s.n_head,
        PREFIX + "attention.head_count_kv": [s.n_kv if k == "a" else 0 for k in s.pattern],
        PREFIX + "feed_forward_length": _per_layer(s, s.expert_ff, "e"),
        PREFIX + "expert_feed_forward_length": _per_layer(s, s.expert_ff, "e"),
        PREFIX + "expert_used_count": _per_layer(s, s.used, "e"),
        PREFIX + "expert_count": max(s.experts),
        PREFIX + "expert_shared_feed_forward_length": s.shexp_ff,
        PREFIX + "moe_latent_size": s.latent,
        PREFIX + "ssm.inner_size": s.inner,
        PREFIX + "ssm.time_step_rank": s.ssm_heads,
        PREFIX + "ssm.group_count": s.ssm_groups,
        "general.name": s.name,
        "general.size_label": s.size_label,
    }


def mamba(b: tensors.Builder, s: Shape, p: str, layer: int) -> None:
    d, g = s.d, ("g", tensors.GAIN["proj_in"])
    b.add(p + "ssm_in.weight", [d, s.in_proj], g, layer)
    b.add(p + "ssm_in.scale", [1], ("uni", 0.5, 1.5), layer)
    b.add(p + "ssm_out.weight", [s.inner, d], ("g", tensors.GAIN["proj_out"]), layer)
    b.add(p + "ssm_out.scale", [1], ("uni", 0.5, 1.5), layer)
    b.add(p + "ssm_conv1d.weight", [s.ssm_conv, s.conv_dim], ("s", 0.5), layer)
    b.add(p + "ssm_conv1d.bias", [s.conv_dim], ("s", 0.1), layer)
    b.add(p + "ssm_dt.bias", [s.ssm_heads], ("uni", -2.0, 0.5), layer)
    b.add(p + "ssm_a", [1, s.ssm_heads], ("uni", -3.0, -0.1), layer)  # negative: the state decays
    b.add(p + "ssm_d", [1, s.ssm_heads], ("uni", 0.5, 1.5), layer)
    b.add(p + "ssm_norm.weight", [s.inner // s.ssm_groups, s.ssm_groups], ("norm",), layer)


def attention(b: tensors.Builder, s: Shape, p: str, layer: int) -> None:
    d, hd, gain = s.d, s.head_dim, tensors.GAIN
    b.add(p + "attn_q.weight", [d, s.n_head * hd], ("g", gain["proj_in"]), layer)
    b.add(p + "attn_k.weight", [d, s.n_kv * hd], ("g", gain["proj_in"]), layer)
    b.add(p + "attn_v.weight", [d, s.n_kv * hd], ("g", gain["proj_in"]), layer)
    b.add(p + "attn_output.weight", [s.n_head * hd, d], ("g", gain["attn_out"]), layer)


def moe(b: tensors.Builder, s: Shape, p: str, layer: int) -> None:
    i = s.moe_index(layer)
    n, ff, d, gain = s.experts[i], s.expert_ff[i], s.d, tensors.GAIN
    b.add(p + "ffn_gate_inp.weight", [d, n], ("g", gain["router"]), layer)
    b.add(p + "exp_probs_b.bias", [n], ("uni", -0.1, 0.1), layer)
    b.add(p + "ffn_up_exps.weight", [s.latent, ff, n], ("g", gain["exps_in"]), layer)
    b.add(p + "ffn_down_exps.weight", [ff, s.latent, n], ("g", gain["exps_out"]), layer)
    b.add(p + "ffn_latent_down.weight", [d, s.latent], ("g", gain["proj_in"]), layer)
    b.add(p + "ffn_latent_up.weight", [s.latent, d], ("g", gain["proj_out"]), layer)
    b.add(p + "ffn_up_shexp.weight", [d, s.shexp_ff], ("g", gain["shexp_in"]), layer)
    b.add(p + "ffn_down_shexp.weight", [s.shexp_ff, d], ("g", gain["shexp_out"]), layer)


def block(b: tensors.Builder, s: Shape, layer: int) -> None:
    p = f"blk.{layer}."
    b.add(p + "attn_norm.weight", [s.d], ("norm",), layer)
    {"m": mamba, "a": attention, "e": moe}[s.pattern[layer]](b, s, p, layer)


def model_tensors(s: Shape, types: dict[str, list[int]], seed: int) -> list[OutTensor]:
    b = tensors.Builder(types, seed)
    b.add("token_embd.weight", [s.d, s.vocab], ("s", 1.0))
    b.add("output_norm.weight", [s.d], ("norm",))
    b.add("output.weight", [s.d, s.vocab], ("g", tensors.GAIN["output"]))
    for layer in range(s.n_layer):
        block(b, s, layer)
    return b.out
