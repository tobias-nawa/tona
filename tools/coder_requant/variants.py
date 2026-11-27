"""The variants of equal trunk size (~20 GB) and their llama-quantize arguments.

Sizes per expert (3 x 2560 x 640 weights): IQ2_XXS 2.06, IQ2_S 2.56, IQ3_XXS 3.06, Q2_0 2.25 bits per weight;
trunk = 4.45 GB dense + 48 layers x E x bytes per expert (research/measurements/2026-11-11-mac-densek.md, sizes from
the Coder header). down has 640 columns, so only 32/64-element block types fit there (Q2_0, IQ4_NL).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Variant:
    name: str
    n_expert: int
    gate_up: str  # ggml type name as llama-quantize parses it
    down: str


VARIANTS = {
    "A": Variant("A", 256, "iq2_xxs", "q2_0"),
    "B": Variant("B", 216, "iq2_s", "q2_0"),
    "C": Variant("C", 192, "iq3_xxs", "q2_0"),
}

QUANT_THREADS = 4  # the build machine stays usable for other work
NICE = 10


def tensor_types(v: Variant, ple_type: str | None = None) -> str:
    """--tensor-type-file content: regex on tensor names; other tensors keep the default type.

    ple_type: the type the source file already has for the per-layer embedding table (the Coder: iq4_nl). Without
    the entry llama-quantize converts that 51-billion-element table to the default type, which takes hours and
    writes 50 GB that assemble discards (seen 2026-11-11 on the bring-up file). A base Q8_0 source needs no entry.
    """
    text = f"ffn_gate_exps={v.gate_up}\nffn_up_exps={v.gate_up}\nffn_down_exps={v.down}\n"
    return text + (f"per_layer_token_embd={ple_type}\n" if ple_type else "")


def quantize_cmd(quantize: str, imatrix: str, types_file: str, src: str, dst: str) -> list[str]:
    """Default type Q8_0 equals the source's, so llama-quantize copies non-expert tensors unchanged
    (src/llama-quant.cpp: quantize = cur_type != new_type); assemble discards them anyway."""
    return [
        "nice", "-n", str(NICE), quantize, "--allow-requantize", "--imatrix", imatrix,
        "--tensor-type-file", types_file, src, dst, "Q8_0", str(QUANT_THREADS),
    ]  # fmt: skip
