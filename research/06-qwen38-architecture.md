# Qwen3.8-Flash-Next: architecture

As of 2026-10-22. Agent research, not re-measured by us.

Goal: enough detail to write the forward pass in HIP ourselves. All shapes are in row
notation `[output] = W·x`, token axis last. "Decode" = 1 token, "prefill" = T tokens.

## Sources

| Key | Source |
|---|---|
| CFG | [config.json](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/config.json), line numbers of the raw file |
| CARD | [model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), section "Model Overview" |
| TR | [tech report](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/main/tech_report.pdf), 28 pp., section or equation |
| HF | transformers `src/transformers/models/qwen4_exp/modular_qwen4_exp.py` (main, retrieved 2026-10-22) |
| LC | llama.cpp b11822 local, `work/src/llama.cpp/…`, path:line. Model graph `src/models/qwen4exp.cpp` = **Q4X** |
| GGUF | header of the local ISTA Coder GGUF (gguf-py) and headers of unsloth UD-Q4_K_XL, UD-IQ1_M, ISTA IQ2_XS and `mtp-…-Q8_0.gguf` read via HTTP range |
| COD | local install of the pruned Coder model download (v0.1.47): its load script, MTP inventory and install log |

PRs: architecture [#27742](https://github.com/ggml-org/llama.cpp/pull/27742) (merged 2026-08-27),
MTP [#29761](https://github.com/ggml-org/llama.cpp/pull/29761) (merged 2026-10-01), indexer cache from
[#27773](https://github.com/ggml-org/llama.cpp/pull/27773) (GLM5-Next, merged 2026-09-30).

## Summary

- 48 layers, pattern 12 × (3 × Gated DeltaNet → 1 × QSA), **every** layer with MoE (512 experts, top-10 + 1 shared).
- **No LayerNorms in the block.** Instead of pre-norm, a 4-fold widened residual stream (4 × 2560) with
  "Gated Residual": elementwise, data-dependent read gates and one scalar write gate per stream.
- QSA = DeltaNet-free layer with full GQA attention, but only over **≤ 2,051 tokens**: top-512
  blocks of 4 tokens, selected by a small indexer (4 heads × 128, MQA), plus the partial block.
- N-gram table (51.2 billion parameters) at **one** layer (index 1): 16 rows of 160 per token,
  addressed by hash over the last 1–2 predecessors. Almost free computationally, I/O only.
- MTP: one additional QSA+MoE block, with its own mixer, shares embedding and LM head.
- **Active bytes per token in the Coder GGUF: 4.1 GB dense + 0.98 GB experts.** The estimate
  "~2.5 GB dense" in `03-feasibility.md` is too low (see section 10).

## 1. Hyperparameters

| Quantity | Value | Source |
|---|---|---|
| `hidden_size` d | 2560 | CFG:20, GGUF `embedding_length` |
| Layers | 48 (+1 MTP) | CFG:103, CFG:96 |
| Layer types | `linear_attention` ×3, `full_attention` ×1, repeated; QSA at index 3, 7, …, 47 | CFG:27–76, CFG:14 |
| Vocabulary | 248,320 (padded), `tie_word_embeddings` false | CFG:131, CFG:129 |
| RMS eps | 1e-6 | CFG:114 |
| **Gated Residual** streams / rank | 4 / 320 | CFG:15–16 |
| **DeltaNet** heads QK / V | 16 / 48 | CFG:79–80 |
| DeltaNet head dim k / v | 128 / 128 | CFG:78, 81 |
| DeltaNet conv kernel | 4 (causal, depthwise) | CFG:77 |
| DeltaNet state dtype | float32 | CFG:83 (`mamba_ssm_dtype`) |
| DeltaNet output gate | sigmoid (not SiLU as in Qwen3.5) | CFG:105, Q4X:657, TR eq. 11 |
| **QSA** Q heads / KV heads | 24 / 2 | CFG:100, 104 |
| QSA head dim | 256 | CFG:17 |
| RoPE | partial 0.25 → **64 of 256** dimensions, θ = 1e7, interleaved M-RoPE sections [11, 11, 10] | CFG:115–125, GGUF `rope.dimension_count` = 64 |
| RoPE variant in llama.cpp | `LLAMA_ROPE_TYPE_IMROPE` | LC `src/llama-model.cpp:3657` |
| Context | 262,144 native, YaRN ×4 up to ~1M | CFG:84, CARD "Best Practices" |
| Indexer Q heads / K heads / dim | 4 / 1 / 128 | CFG:23–25 |
| Indexer budget / block size | 2048 tokens / 4 → 512 blocks | CFG:21–22, TR §2.1.2 |
| **MoE** experts / active | 512 / 10 (+1 shared) | CFG:101–102, CARD |
| Expert intermediate | 640 (shared also 640) | CFG:86, 127 |
| Router | softmax over all, top-10, renormalized (`norm_topk_prob` true) | HF router, `configuration_qwen4_exp.py:162`, LC Q4X:1175–1188 |
| **N-gram** n / heads per order | 3 / 8 → 16 heads (8 bigram, 8 trigram) | CFG:98, 18 |
| N-gram base size per head | 20,000,000 (next primes above) | CFG:99, GGUF `ple.head_vocab_sizes` |
| N-gram row dim / total dim | 160 / 2560 (16 × 160) | GGUF `embedding_length_per_layer_input`, CFG:110 |
| N-gram table rows | 320,001,536 (padded to ×128) | GGUF tensor `per_layer_token_embd` |
| N-gram layer | `ple_layer_ids` = [2] (1-based) → **index 1** | CFG:111–113, GGUF `ple.layers` = [1] |
| PLE conv kernel / dilation | 4 / 3 (= n) | CFG:109, Q4X:1442–1444 |
| MTP | 1 layer, QSA, θ = 1e7 | CFG:87–96 |
| EOS (for n-gram reset) | 248044 `<\|endoftext\|>` | CFG:13, GGUF `ple.eos_token_id` |
| Vision | Qwen3-VL ViT, 27 layers, 1152 wide, output 2560 | CFG:136–151 |

Derived widths: `hc_dim` = 4 × 2560 = 10,240; DeltaNet `key_dim` = 16 × 128 = 2048,
`value_dim` = 48 × 128 = 6144, conv channels = 2·2048 + 6144 = 10,240 (Q4X:223–229).

## 2. Layer sequence and block structure

```
R = [E[tok]] × 4                        # 4 identical copies, [4, 2560]   (Q4X:455–461, TR eq. 35)
for l in 0..47:
    if l == 1: R = PLE(R, ngram_rows)   # n-gram injection, before the first mixer of this layer
    x, s = GR_read(R, attn_gr[l])       # x [2560], s [4]
    y = DeltaNet(x) if l % 4 != 3 else QSA(x)
    R = R + s ⊗ y                        # each stream c: R_c += s_c · y
    x, s = GR_read(R, ffn_gr[l])
    y = MoE(x) + SharedExpert(x)
    R = R + s ⊗ y
h_mtp = R                               # [4, 2560], input of the MTP head
x = GR_read(R, head_gr)                 # without write gate; replaces the final norm
logits = W_lm · x
```

- There is **no** `output_norm` and no pre-norm in the blocks: the group RMSNorm sits in the
  GR read path (Q4X:184, TR §2.2 "GR also replaces the block's pre-normalization").
- No stream-mixing operator `H_res` (unlike HC/mHC). A stream is changed only by writing
  (TR §2.2).
- Shared expert: yes, with its own sigmoid gate (Q4X:1191–1213).
- Layer type rule in llama.cpp: recurrent if `(i+1) % 4 != 0` (Q4X:149–156).

## 3. Gated Residual (GR)

Two modules per layer (before mixer, before MoE), one at the head. Weights per module: `hc_norm` [4, 2560]
(the converter folds in 1+w), `down` 10240→320, `up` 320→10240, `inject` 10240→4 (not at the head).

| Step | Operation | Shape decode | Shape prefill |
|---|---|---|---|
| 1 | `xn_c = RMSNorm(R_c) ⊙ γ_c` per stream c (group RMSNorm over 2560) | [4, 2560] | [4, 2560, T] |
| 2 | `lo = SiLU((W_down · vec(xn)) / 4)` | [320] | [320, T] |
| 3 | `G = σ(W_up · lo)` | [10240] | [10240, T] |
| 4 | `x = mean_c(G_c ⊙ xn_c)` → block input | [2560] | [2560, T] |
| 5 | `s = 2·σ((W_inject · vec(xn)) / 4)` | [4] | [4, T] |
| 6 | after the block: `R_c ← R_c + s_c · y` | [4, 2560] | [4, 2560, T] |

Sources: Q4X:325–411, HF `Qwen4ExpTextGatedResidual`, TR eq. 30–34. Notes:

- `2σ(·)` centers the write gate on 1, a zero injection is a normal residual add (Q4X:391).
- TR §2.2: residual state **can be stored in FP8**, "almost no loss in quality"; read and write each
  one fused kernel with folded group norm. llama.cpp keeps it in F32, optionally fused
  (`ggml_dsv4_hc_pre_gated`, `ggml_dsv4_hc_post`, Q4X:351–356, 396–399).
- Residual per token: 40 KB F32, 20 KB BF16, 10 KB FP8. Per layer and token it is moved ~3×
  (2× read, 1× write per module, estimate). Negligible in decode, not in prefill:
  8K chunk ≈ 8192 × 48 × 2 × 3 × 40 KB ≈ 97 GB of memory traffic in F32 (estimate, ≈ 0.3 s at 320 GB/s).

## 4. Gated DeltaNet (36 layers)

GGUF tensors per layer (Q4X:255–263): `attn_qkv` 2560→10240, `attn_gate` (z) 2560→6144,
`ssm_beta` 2560→48, `ssm_alpha` 2560→48, `ssm_conv1d` [4, 10240], `ssm_dt.bias` [48],
`ssm_a` [48] (= −exp(A_log), precomputed by the converter), `ssm_norm` [128], `ssm_out` 6144→2560.

| Step | Operation | Decode | Prefill |
|---|---|---|---|
| 1 | GR read → x | [2560] | [2560, T] |
| 2 | `qkv = W_qkv·x` (order q 2048 \| k 2048 \| v 6144), `z = W_z·x`, `b = W_β·x`, `a = W_α·x` | [10240], [6144], [48], [48] | ×T |
| 3 | Causal depthwise conv K=4 over `[conv_state(3) ; qkv]`, then SiLU; conv state ← last 3 | [10240] | [10240, T+3] → [10240, T] |
| 4 | q, k → [16, 128], **L2 norm** per head; v → [48, 128] | | |
| 5 | `β = σ(b)`, `g = ssm_a · softplus(a + dt_bias)`, `α = exp(g)` | [48] | [48, T] |
| 6 | Recurrence per V head j, K head i (see below): `S ← α S`; `u = β (v_j − Sᵀ k_i)`; `S ← S + k_i uᵀ`; `o_j = Sᵀ (q_i / √128)` | 48 × [128×128] | chunk form, see below |
| 7 | `o = RMSNorm_128(o) ⊙ w_norm ⊙ σ(z)` per head | [48, 128] | ×T |
| 8 | `y = W_out · o` | [2560] | [2560, T] |
| 9 | GR write | | |

Sources: Q4X:1046–1169, LC `src/models/delta-net-base.cpp:286–368` (autoregressive), TR eq. 1–11,
Gated DeltaNet paper [arXiv 2412.06464](https://arxiv.org/abs/2412.06464).

- **Head assignment:** HF orders V heads grouped (V head j uses K head ⌊j/3⌋). The converter reorders
  V heads in the GGUF to "tiled", then **i = j mod 16** applies (LC `conversion/qwen.py:463–501`). Whoever
  loads the GGUF must use mod 16; whoever loads HF safetensors, ⌊j/3⌋.
- **State layout in llama.cpp:** `s` = [S_v, S_v, H_v], row = V index, contiguous over the
  K index (`sum_rows(s ⊙ k)` reduces over k, delta-net-base.cpp:340–356).
- L2 norm: `x / sqrt(Σx² + eps)`, implemented as `rms_norm(x, eps/n) / √n` (LC `src/models/models.h:14–18`).
- `ssm_norm` is **not** a zero-centered norm (the converter excludes `linear_attn.norm.weight` from the +1,
  `conversion/qwen.py:411`; HF `Qwen4ExpTextRMSNormGated` multiplies directly). All other norms
  in the model: weight = 1 + w, already folded in the GGUF.
- **Prefill:** chunk form (WY/UT transformation after Yang et al., §3) with chunk size 64
  (delta-net-base.cpp:61). With `fused_gdn_ch` a dedicated ggml op `ggml_gated_delta_net`, which additionally
  writes K state snapshots for rollback (`ggml.h`: "slot s = state s tokens back").
- Qwen trains with its own kernel [FlashQLA](https://github.com/QwenLM/FlashQLA) (TileLang, TR §2.1.1).

## 5. QSA layer (12 layers)

GGUF tensors (Q4X:241–253): `attn_q` 2560→12288 (**per head [q 256 \| gate 256] interleaved**),
`attn_k` 2560→512, `attn_v` 2560→512, `attn_output` 6144→2560, `attn_q_norm`/`attn_k_norm` [256],
`indexer.q_proj` 2560→512, `indexer.k_proj` 2560→128, `indexer.q_norm`/`k_norm` [128]. In HF the
indexer is a fused projection `index_qk_proj` 2560→640 (the converter splits it, `conversion/qwen4exp.py:154–161`).

### Indexer and selection

| Step | Operation | Decode | Prefill |
|---|---|---|---|
| 1 | `k_raw = W_ik·x` → **raw** into the indexer cache (no norm, no RoPE) | [128] | [128, T] |
| 2 | If this token completes a block of 4: `k̄ = RMSNorm(mean(k_raw of the 4)) ⊙ (1+w)`, then RoPE with the **position of the first block token** | [128] per finished block | [128, ≤T/4+1] |
| 3 | `q_i = W_iq·x` → [4, 128], RMSNorm per head, RoPE with token position | [4, 128] | [4, 128, T] |
| 4 | `score_b = Σ_h ReLU(q_h · k̄_b) / √128`, only for **complete** blocks with `p_b + 3 ≤ pos` | [n_blk] | [n_blk, T] (block-causally masked) |
| 5 | Top-512 blocks → up to 2048 token indices, plus **tail** = the `(pos+1) mod 4` tokens of the partial block (incl. the current one) | ≤ 2051 indices | own list per query |

Sources: Q4X:757–898, LC `src/llama-memory-hybrid-idx.cpp:620–976` (pool lifecycle, mask l. 910,
tail l. 944), HF `Qwen4ExpTextQSAIndexer`, TR eq. 12–16. Score op `ggml_lightning_indexer` with
constant head weights 1/√128 and ReLU (Q4X:823–826, LC `ggml/src/ggml-cpu/ops.cpp:12531ff`).

- Blocks are formed in sequence order: [0..3], [4..7], … (HF forms them from the visible
  tokens, identical under a causal mask).
- **Up to 2,051 tokens of context QSA is exactly dense causal attention** (all ≤ 512 blocks + tail selected).
  Only above that does the selection take effect.
- Consequence of the definition: if the current token completes its block (`(pos+1) mod 4 == 0`), the
  tail is empty and the token sees **itself only if its block lands in the top-512**. Identical in HF
  and llama.cpp, so intended or at least reference behavior.
- Model: DSA lightning indexer from DeepSeek-V3.2 ([arXiv 2512.02556](https://arxiv.org/abs/2512.02556)),
  here compressed to blocks: indexer cost O(n²/4) instead of O(n²) (TR §2.1.2).

### Core attention

| Step | Operation | Decode | Prefill |
|---|---|---|---|
| 1 | `[q \| g] = W_q·x`, per head h: q = columns [512h, 512h+256), g = [512h+256, 512h+512) | [24, 256] + [24, 256] | ×T |
| 2 | q: RMSNorm_256 ⊙ (1+w), RoPE on dim 0..63 | [24, 256] | |
| 3 | `k = W_k·x` → [2, 256], RMSNorm, RoPE; `v = W_v·x` → [2, 256]; into KV cache | | |
| 4 | Softmax attention over the selected ≤ 2051 positions, scale 1/16, GQA 12:1 | [24, 256] | own selection per query |
| 5 | `o ⊙ σ(g)` (gated attention, [arXiv 2505.06708](https://arxiv.org/abs/2505.06708)) | [6144] | |
| 6 | `y = W_o · o` | [2560] | [2560, T] |

Sources: Q4X:959–1044, 900–957.

- **RoPE:** HF rotates `x[..., :64]` with `rotate_half` (pairs (i, i+32)), rest unchanged; inv_freq
  θ^(−2i/64), i = 0..31. For pure text tokens all three M-RoPE positions are equal, so
  IMRoPE corresponds exactly to a partial NeoX RoPE. Also holds for the indexer (64 of 128, TR eq. 14).
- llama.cpp builds the selection as an additive mask over all `n_kv` cells. With flash attention on
  CUDA/HIP the mask is compacted into an index list and **only the selected rows** are read
  (`ggml_flash_attn_ext_set_n_kv_max`, LC `ggml/src/ggml-cuda/fattn-common.cuh:1101–1141`). Building the mask
  stays O(n_kv) per token.

## 6. MoE (all 48 layers)

| Step | Operation | Shape |
|---|---|---|
| 1 | GR read → x | [2560] |
| 2 | `logits = W_r · x` (BF16/F32 in the GGUF), softmax in F32, top-10, `w /= Σw` | [512] → 10 |
| 3 | per expert e: `h = SiLU(W_g,e·x) ⊙ (W_u,e·x)`, `y += w_e · W_d,e·h` | [640] → [2560] |
| 4 | Shared: `y += σ(w_sg · x) · W_ds(SiLU(W_gs·x) ⊙ W_us·x)` | `w_sg` [2560] → scalar |
| 5 | GR write | |

- No router bias, no scaling (`expert_weights_scale` missing → 0 → no scale, LC `src/llama-graph.cpp:2349`).
- GGUF shapes: `ffn_gate_exps`/`ffn_up_exps` (2560, 640, E), `ffn_down_exps` (640, 2560, E). Expert e
  is contiguous at offset e × (bytes per expert) (COD `ref/load.py:186–208`).
- **Coder variant:** RCO-pruned to **256 experts per layer** (12,288 total), router rows
  shortened as well (GGUF `expert_count` = 256, `ffn_gate_inp` (2560, 256)). Since top-k is renormalized
  afterwards, the weights among the selected experts do not change, only the selection set.
- `ffn_down` has row length 640, not divisible by 256: **K-quants and IQ1/2/3 do not work there.**
  llama.cpp falls back (IQx → IQ4_NL, Q4_K → Q5_0, LC `src/llama-quant.cpp:414–431`); ISTA uses Q2_0
  (block 64, 2.25 bpw) or IQ4_NL, unsloth Q5_1/Q8_0.

## 7. N-gram embedding (PLE)

### Addressing

For the token at position p let t0 = tok[p], t1 = tok[p−1], t2 = tok[p−2]. Multipliers
m = [23703573157769, 20109073645365, 8052911324071] (GGUF `ple.layer_multipliers`, int64, odd).

```
mix2 = (t0·m0) XOR (t1·m1)
mix3 = (t0·m0) XOR (t1·m1) XOR (t2·m2)
head h = 0..7  (bigram):  row_h = mix2 mod P_h + off_h
head h = 8..15 (trigram): row_h = mix3 mod P_h + off_h
e = concat(table[row_0], …, table[row_15])   # [16 × 160] = [2560]
```

- P_h: 16 primes from 20,000,003 (20000003, 20000023, …, 20000171), off_h cumulative
  (0, 20000003, 40000026, …). Values in the GGUF, derivation in HF `_find_nth_prime_after`.
- No overflow: m < (2^63−1)/vocab, the product fits in int64; XOR stays non-negative. Signed (HF)
  and unsigned (llama.cpp) give the same result.
- **EOS reset:** predecessors before the sequence start or behind an EOS (248044 `<|endoftext|>`) are
  replaced by EOS. The EOS token itself keeps its context (HF `_shift_right_ignore_eos`,
  Q4X:1282–1294). Image tokens hash as `image_token_id` 248056.
- `<|im_end|>` (248046, tokenizer EOS in the GGUF) does **not** reset.
- Sources: Q4X:1221–1320, HF `Qwen4ExpTextNGramEmbedding`, TR §2.3 (model: Engram, Cheng et al. 2026,
  "Conditional memory via scalable lookup").

### Injection (layer 1, before the first GR read)

| Step | Operation | Shape |
|---|---|---|
| 1 | fetch rows, dequantize | 16 × [160] → e [2560] |
| 2 | `key = W_key·e`, group RMSNorm per stream ⊙ γ_key | [4, 2560] |
| 3 | `val = W_val·e` | [2560] |
| 4 | `query = RMSNorm(R_c) ⊙ γ_query` per stream | [4, 2560] |
| 5 | `s_c = ⟨key_c, query_c⟩ / √2560`; `gate_c = σ(sign(s_c)·√max(\|s_c\|, 1e-6))` | [4] |
| 6 | `gv_c = gate_c · val` | [4, 2560] |
| 7 | `n = RMSNorm(gv_c) ⊙ γ_conv` per stream | [4, 2560] |
| 8 | Depthwise conv, 10,240 channels, kernel 4, **dilation 3**: taps at p, p−3, p−6, p−9; then SiLU | [10240] |
| 9 | `R ← R + gv + conv` | [4, 2560] |

Sources: Q4X:1398–1486, HF `Qwen4ExpTextPLELayer`. Tensors: `ple_key` 2560→10240, `ple_value`
2560→2560, three norm weights [4, 2560], `ple_conv1d` [4, 10240].

### I/O profile

- Per token 16 rows of 90 B (IQ4_NL, 160 = 5 × 32) = 1.4 KB, spread over 28.8 GB. From NVMe: 16
  random 4K reads per token. At 50 tok/s decode that is 800 IOPS, in a prefill chunk of 8192 tokens 131K reads.
- The rows of a token are fixed as soon as the token is known. Prefetch is therefore possible already during
  layer 0, also for MTP drafts (TR §2.3.1 places the layer at index 1 for exactly this reason).
- Table: 51.2 billion parameters, BF16 102.4 GB, F32 204.8 GB, IQ4_NL 28.8 GB. All known GGUFs
  (ISTA, unsloth) use IQ4_NL, unpruned and unsearched (ISTA model card).

## 8. MTP head

Structure (Q4X:542–634, tensors Q4X:289–301; HF names `mtp.*`, COD `mtp/mtp-inventory.md`):

```
e  = RMSNorm(E[x_{t+1}]) ⊙ γ_e                    # [2560]          pre_fc_norm_embedding
hs = RMSNorm(h_t,c) ⊙ γ_h,c  per stream c         # [4, 2560]       pre_fc_norm_hidden
R_c = W_fe·e + W_fh·hs_c                          # [4, 2560]       fc_embedding, fc_hidden (each 2560×2560)
R  = one QSA block (GR → QSA → GR → MoE 512 experts + shared → GR)
h_{t+1} = R                                       # input of the next draft step
logits = W_lm · GR_read(R, mtp_head_gr)           # own head mixer, LM head shared
```

- `h_t` is the trunk residual stream **before** the final mixer, i.e. 10,240 values per token (Q4X:518–521).
- Own KV and indexer cache, no DeltaNet (LC `src/llama-model.cpp:3147–3153`).
- Size: block 2.607 billion parameters (BF16 5.21 GB, COD inventory). As a standalone GGUF with
  embedding and LM head 3.88 billion (ggml-org `mtp-…-BF16.gguf` 7.77 GB). Hence the "4B MTP" of the model card.
- **llama.cpp:** default draft length 3 (`common/common.h:329`), draft sampler top-k 10
  (`common/speculative.cpp:1499`). Verification: the target computes [token + drafts] in one batch, DeltaNet
  rolls back via state snapshots (`n_rs_seq`, one snapshot per draft position). PR #29761 reports on
  DGX Spark with 3 drafts 1.55× decode and 64% acceptance.
- **Original:** MTP with QSA, top-k indices are **reused across the draft steps** (after GLM-5,
  TR §2.1.2). Mean accepted length for "four-step speculative decoding" 4.07 (TR tab. 4).
  llama.cpp recomputes the indexer in every draft step.
- **MTP file of the Coder download:** MTP experts in Q2_0 (relative RMS error 0.42!, `mtp-q2_0.gguf.report.json`), dense
  parts Q8_0/BF16, 876 MiB VRAM. Draft LM head only over **106,299 tokens** (`draft_vocab.bin`,
  212.9 MiB instead of ~520 MB).
- With MTP a draft step costs mainly the LM head (2560 × vocabulary). The block itself reads
  ~111 MiB dense (COD layout `rt/dense.txt`) + 10 experts of 1.4 MB (Q2_0).

## 9. State per sequence

| Part | Formula | F32 | FP16/BF16 | int8 |
|---|---|---:|---:|---:|
| DeltaNet state per layer | 48 × 128 × 128 | 3.00 MiB | 1.50 MiB | – |
| DeltaNet state total | × 36 | **108 MiB** | 54 MiB | – |
| DeltaNet conv state | 36 × 3 × 10,240 | 4.2 MiB | | |
| PLE conv state | 9 × 10,240 (dilation 3 × (kernel−1)) | 360 KiB | | |
| N-gram context | 2 token IDs | 8 B | | |
| **Recurrent total** | | **112.6 MiB** | | |
| Rollback for k drafts | × (k+1) | 450 MiB at k = 3 | | |
| KV per token and QSA layer | 2 × 2 × 256 (K+V) | | 2048 B | 1024 B (q8_0: 1088 B) |
| Indexer per token and layer | 128 raw (+ 128 pooled per 4 tokens) | | 256 B + 64 B | 128 B + 32 B |
| **KV + indexer per token** | × 12 | | **≈ 28.4 KB** | **≈ 14.2 KB** |
| at 64K / 256K context | | | 1.9 / 7.4 GB | 0.9 / 3.7 GB |
| MTP layer | +1/12 | | | |

- `mamba_ssm_dtype` = float32 (CFG:83), llama.cpp keeps state and conv in F32 (`src/llama-model.cpp:3178–3190`).
  Whether FP16/BF16 state suffices: **open**.
- llama.cpp reserves in the indexer cache per cell [raw 128 | pooled 128] (`hybrid-idx.cpp:57–58`), the
  pooled half is occupied only in every 4th row. With q8_0 that is 272 B instead of the ~170 B needed per token.
- Read volume in decode at 64K context: KV of the selection 12 × 2051 × 1024 B = 25 MB, pooled indexer keys
  12 × 16,384 × 128 × 2 B = 50 MB (FP16). **The indexer scan grows linearly and overtakes the attention
  from ~32K.** Still small against ~5 GB of weights per token (< 2%, at 256K ~4%).

## 10. Weight sizes

### Parameters per part (unpruned, counted from the shapes)

| Part | Parameters | Note |
|---|---:|---|
| Experts | 120.80 billion | 48 × 512 × 3 × 2560 × 640 = 4.915 M per expert |
| DeltaNet projections | 2.085 billion | 57.9 M per layer |
| QSA projections incl. indexer | 0.617 billion | 51.4 M per layer |
| Gated Residual | 0.640 billion | 6.6 M per module, 97 modules |
| Router + shared expert | 0.299 billion | |
| PLE projections | 0.033 billion | |
| Embedding / LM head | 0.636 / 0.636 billion | not shared |
| **Trunk** | **125.74 billion** | model card "125B" |
| N-gram table | 51.20 billion | model card "51B" |
| MTP block | 2.61 billion | (+1.27 with embedding/LM head) |
| Vision | ~0.4 billion | |
| **Sum** | **~180.0 billion** | HF API: 179,999,981,459 |
| Active per token | 6.03 billion without LM head, 6.67 billion with | model card "6B" |

### Bytes per part in existing GGUFs (measured from the headers)

| | Coder ("IQ1_M", 256 exp.) | ISTA IQ2_XS (512) | unsloth UD-IQ1_M (512) | unsloth UD-Q4_K_XL (512) |
|---|---:|---:|---:|---:|
| Experts total | 25.15 GB | 35.46 GB | 41.84 GB | 77.02 GB |
| per expert (gate+up+down) | 2.046 MB | 1.443 MB | 1.702 MB | 3.134 MB |
| Types gate/up · down | IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS · IQ4_NL/Q2_0 | IQ2_S/IQ2_XXS/IQ1_M · Q2_0 | IQ1_M/IQ2_XXS · IQ4_NL | Q4_K/Q5_K · Q5_1/Q8_0 |
| DeltaNet | 1.557 GB | 1.115 GB | 1.549 GB | 2.247 GB |
| Gated Residual | 1.283 GB (BF16) | 1.283 GB (BF16) | 0.695 GB (Q8_0) | 0.695 GB (Q8_0) |
| QSA | 0.477 GB | 0.360 GB | 0.450 GB | 0.674 GB |
| LM head / embedding | 0.521 / 0.338 GB | 0.338 / 0.338 GB | 0.358 / 0.358 GB | 0.675 / 0.675 GB |
| Router + shared + PLE proj. | 0.275 GB | 0.327 GB | 0.480 GB | 0.538 GB |
| N-gram table | 28.80 GB | 28.80 GB | 28.80 GB | 28.80 GB |
| **File total** | 58.40 GB | 68.03 GB | 74.53 GB | 111.32 GB |
| **active/token dense** | **4.11 GB** | 3.42 GB | 3.53 GB | 4.83 GB |
| **active/token experts** | **0.98 GB** | 0.69 GB | 0.82 GB | 1.50 GB |

"Active dense" = all non-expert weights without embedding and table, incl. LM head. The Coder
file is named IQ1_M but contains **no** IQ1_M tensor; the experts average
3.33 bpw (ISTA card: "retained weights stored at 3.5 bpw").

### Pure quantizations (computed, not seen as a file)

`ffn_down` with the llama.cpp fallback (IQ4_NL or Q5_0), see section 6.

| | per expert | experts 512 / 256 | active/token experts |
|---|---:|---:|---:|
| IQ1_M (1.75 bpw) + down IQ4_NL | 1.638 MB | 40.3 / 20.1 GB | 786 MB |
| IQ2_XS (2.31 bpw) + down IQ4_NL | 1.869 MB | 45.9 / 23.0 GB | 897 MB |
| IQ2_XS + down Q2_0 (as ISTA) | 1.408 MB | 34.6 / 17.3 GB | 676 MB |
| Q4_K (4.5 bpw) + down Q5_0 | 2.970 MB | 73.0 / 36.5 GB | 1,425 MB |

### Consequences for the ceiling in 03-feasibility.md

Per forward pass the Coder file reads 4.11 + 0.98 = **5.1 GB**. If everything were in VRAM:
5.1 GB / 320 GB/s = 16 ms → **~63 passes/s**, not ~95. Levers without new training: GR weights from
BF16 to Q8_0 (−0.6 GB), DeltaNet from Q6_K to Q4/Q5 (−0.5 GB), draft vocabulary. With ~3.5 GB dense
the ceiling would be at ~70–75 passes/s (estimate).

## 11. FLOPs per token

Matmul FLOPs = 2 × active parameters. Unpruned, the Coder is practically the same (only the router is half).

| Part | GFLOP/token |
|---|---:|
| Routed experts (480 expert calls) | 4.72 |
| DeltaNet projections | 4.17 |
| Gated Residual (97 modules) | 1.28 |
| LM head | 1.27 |
| QSA projections incl. indexer | 1.24 |
| Shared experts | 0.47 |
| Router, PLE proj. | 0.19 |
| **Matmul total** | **13.34** |
| DeltaNet recurrence (36 × 48 × ~4 × 128²) | ~0.23 |
| QSA core attention (≤ 2051 positions) | 0.60 from 2K context |
| Indexer scoring | 0.10 at 32K, 0.20 at 64K, 0.81 at 256K |
| **Decode total** | **~14.3 (32K) to ~15.0 (256K)** |

- **Prefill:** LM head only for the last token, so ~12.1 GFLOP matmul + recurrence in chunk form
  (~0.3) + attention (up to 2K context dense causal, above that constant 0.6) + indexer (grows with position).
  On average **~13 GFLOP/token**, matches the 12 GFLOP in 03-feasibility.md.
- Prefill ceiling at 51.3 TFLOPS FP16: ~3,900 tok/s prefill.
- Experts in prefill: at T = 8192 each of the 512 experts gets on average 160 tokens per layer,
  enough for WMMA tiles.

## 12. Where llama.cpp deviates from the original

| Point | Original (TR/HF) | llama.cpp b11822 |
|---|---|---|
| MTP indices | top-k reused across draft steps | recomputed every step |
| Residual | FP8 storable, read/write one kernel each | F32, optional fusion |
| QSA selection | sparse kernel, KL loss fused (training) | additive mask over n_kv, FA-HIP compacts to an index list |
| Indexer cache | not specified | [raw \| pooled] per cell, pooled half 3/4 unused |
| Pooled-key computation | rebuilt per query (HF reference, inefficient) | once per finished block, persistent |
| N-gram hash | on GPU in torch | on the CPU (ggml has no int64/XOR), row prefetch |
| PLE conv | `nn.Conv1d` with dilation | sum of shifted copies (`ggml_conv_1d_dw` marked unreliable) |
| V head order | grouped | tiled (converter) |

No numerical deviation found: the formulas for GR, PLE, QSA, DeltaNet and MoE match
`modular_qwen4_exp.py`.

## 13. Contradictions between sources

| Topic | Source A | Source B | Assessment |
|---|---|---|---|
| MTP size | model card: 4B | MTP block 2.61 billion | the card apparently counts embedding + LM head (3.88 billion) |
| Total parameters | card 125 + 51 + 4 = 180B, HF API 180.0B | ISTA card 176.9B | ISTA without MTP and vision |
| Dense bytes per token | 03-feasibility: ~2.5 GB (estimated) | Coder GGUF: 4.11 GB | estimate too low |
| Expert bytes Coder | 03: 23 GB, 1.9 MB per expert | 25.15 GB, 2.046 MB | units: 23.4 GiB / 1.95 MiB, no real contradiction |
| EOS | GGUF tokenizer: 248046 `<\|im_end\|>` | `ple.eos_token_id` and CFG: 248044 `<\|endoftext\|>` (CFG: also BOS) | n-gram reset only at 248044 |
| Zero-centered norm | TR §2.1.1: applied to all RMSNorms | HF code and converter: DeltaNet gated norm without +1 | code is authoritative |
| MTP steps | TR tab. 4: "four-step", length 4.07 | TR fig. 6: "next_n = 4, corresponding to three MTP prediction steps" | unclear whether 3 or 4 drafts |
| File name "IQ1_M" | file name ISTA Coder | no IQ1_M tensor contained | name = ISTA convention "effective 1.89 bpw" |

## 14. Open questions

1. Is FP16/BF16 enough for the DeltaNet state, or does the recurrence need F32 (CFG says float32)?
2. MTP in the original: does step 1 use its own indexer run and the following steps its indices, or
   does MTP take over the indices of the last trunk QSA layer? TR only says "across prediction steps".
3. Which FP8 format (E4M3/E5M2, scaling) for the residual stream? TR names none.
4. How much do int8 KV and Q8 indexer keys cost in selection quality at long context? Not investigated.
5. The Coder download ships MTP experts in Q2_0 (RMS error 0.42) from the **unpruned** model. Effect on
   acceptance unknown. A Coder MTP with a pruned router does not exist.
6. Prefill kernel strategy: each query picks its own 512 blocks. Grouping for WMMA (shared
   blocks of neighboring queries) is open.
7. Self-attention at `(pos+1) mod 4 == 0` depends on the top-k (section 5). Whether this matters in practice
   is open; it is reference behavior.
8. From which context length does top-k on the GPU pay off versus partial sorting? At 256K there are 65,536 blocks per
   query and layer.
9. YaRN for > 262K: factor 4 per the card. Not checked in llama.cpp.
