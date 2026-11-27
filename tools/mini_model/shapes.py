"""Sizes of the mini model and the key-value overrides they imply.

Every value is bound by a fixed assumption of the kernels (engine/model/qwen4exp.cpp, engine/gpu/mega_init.cpp);
the reasons are in docs/contribute/testing.md.
"""

from dataclasses import dataclass

# Hash table sizes of the PLE heads: distinct primes, as in the real model, so that heads do not collide alike
PLE_PRIMES = (1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049)


@dataclass(frozen=True)
class Shape:
    vocab: int = 248320  # as the reference: tokenizer and chat tokens stay the same
    n_layer: int = 12
    d: int = 256
    n_hc: int = 4
    hc_rank: int = 64
    n_head: int = 24
    n_kv: int = 2
    head_dim: int = 256
    dn_k: int = 2
    dn_v: int = 6
    dn_head: int = 128
    dn_conv: int = 4
    idx_heads: int = 4
    idx_dim: int = 128
    idx_top_k: int = 64
    idx_block: int = 4
    full_interval: int = 4
    n_expert: int = 64
    n_used: int = 10
    expert_ff: int = 128
    shexp_ff: int = 128
    ple_layer: int = 1
    ple_dim: int = 32
    ple_per_gram: int = 4
    ple_ngram: int = 3
    ple_conv: int = 4
    ple_primes: tuple[int, ...] = PLE_PRIMES  # one hash table size per head
    name: str = "Mini model (random weights)"
    size_label: str = "mini"

    @property
    def hc(self) -> int:
        return self.n_hc * self.d

    @property
    def dn_kd(self) -> int:
        return self.dn_k * self.dn_head

    @property
    def dn_vd(self) -> int:
        return self.dn_v * self.dn_head

    @property
    def ple_heads(self) -> int:
        return self.ple_per_gram * (self.ple_ngram - 1)

    def is_qsa(self, layer: int) -> bool:
        return (layer + 1) % self.full_interval == 0


def ple_tables(s: Shape) -> tuple[list[int], list[int]]:
    vocab = list(s.ple_primes[: s.ple_heads])
    offsets = [sum(vocab[:i]) for i in range(len(vocab))]
    return vocab, offsets


def ple_rows(s: Shape) -> int:
    vocab, offsets = ple_tables(s)
    return offsets[-1] + vocab[-1]


def overrides(s: Shape, n_blocks: int, mtp: bool) -> dict:
    """Values that replace those of the source file; n_blocks counts the MTP block too."""
    p = "qwen4exp."
    ratios = [s.idx_block if s.is_qsa(layer) else 0 for layer in range(s.n_layer)]
    ratios += [s.idx_block] * (n_blocks - s.n_layer)
    o = {
        p + "block_count": n_blocks,
        p + "embedding_length": s.d,
        p + "expert_count": s.n_expert,
        p + "expert_feed_forward_length": s.expert_ff,
        p + "expert_shared_feed_forward_length": s.shexp_ff,
        p + "ssm.group_count": s.dn_k,
        p + "ssm.time_step_rank": s.dn_v,
        p + "ssm.inner_size": s.dn_vd,
        p + "hyper_connection.low_rank": s.hc_rank,
        p + "attention.indexer.top_k": s.idx_top_k,
        p + "attention.compress_ratios": ratios,
        "general.name": s.name,
        "general.size_label": s.size_label,
    }
    if mtp:
        o[p + "attention.recurrent_layers"] = [r == 0 for r in ratios]
    else:
        vocab, offsets = ple_tables(s)
        o.update({
            p + "ple.heads_per_ngram": s.ple_per_gram,
            p + "embedding_length_per_layer_input": s.ple_dim,
            p + "ple.layers": [s.ple_layer],
            p + "ple.head_vocab_sizes": vocab,
            p + "ple.head_offsets": offsets,
            "general.description": "Random weights, architecture of the reference; tools/make_mini_model.py",
        })  # fmt: skip
    return o


DROPPED_PREFIXES = (
    "split.",
    "quantize.",
    "general.quantized_by",
    "general.repo_url",
    "general.tags",
    "general.license",
)
