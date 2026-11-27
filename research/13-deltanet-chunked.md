# 13 – Gated DeltaNet in prefill: chunked form on WMMA

**Agent research, state of 2026-10-24.** Sources in the text and at the end. Own estimates are marked
"estimate"; nothing was measured here. Open points are in section 7.

## 0. Key findings

1. **The algorithm is standard and stably documented.** All production kernels (fla-org
   `chunk_gated_delta_rule`, vLLM, SGLang, FlashQLA, the llama.cpp chunk path) compute the same steps:
   gate cumsum → Gram `KKᵀ` with decay mask → invert the lower triangular matrix (UT transform) →
   `W`, `U` → sequential state recurrence **only over chunks** (`V_new = U − W·S`, `O`, `S` update).
   Chunk size 64 is the standard (fla allows 16/32/64).
2. **A scalar gate per head makes this easy.** The decay is only a 64×64 mask
   `D[t,s] = exp(G_t − G_s)` on the Gram matrix. The expensive "secondary chunking" (chunk 16) of KDA with
   a per-channel gate is not needed.
3. **fp32 is needed for:** gate cumsum, `exp`, Gram+mask, the triangular inverse, the state `S`. **f16/bf16 is enough
   for** the big matmuls (`W·S`, `QKᵀ·V_new`, `KᵀV_new`, `T·K`, `T·V`) with an fp32 accumulator. This is
   exactly what fla does (state/accumulators fp32, operands in input dtype).
4. **Expected gain (estimate):** ~3.9 TFLOP in f16 WMMA instead of ~2.7 TFLOP fp32 elementwise,
   serial. At 20–35 % of the 93 TFLOPS plus memory traffic, `k_dn_recur` lands at roughly **0.2–0.35 s instead of
   1.1 s** for 12,000 tokens, i.e. ~3–5× faster and **~8–10 % of the prefill GPU time**. This depends strongly on
   fusion (section 5). It is an estimate, not a measurement.
5. **No AMD/RDNA precedent found.** fla/vLLM run on AMD through Triton (not specially optimized,
   no numbers found). llama.cpp-HIP inherits the CUDA kernel for the **autoregressive** path. We are
   pioneers here; the reference for testing is fla in PyTorch/Triton or the llama.cpp chunk path (CPU/ggml graph).
6. **Warning from llama.cpp:** For Qwen3.5 on an RTX 5090 the chunk path was initially *slower* than the
   fused sequential kernel; the chunk path was therefore enabled only for KDA (Kimi-Linear)
   (PR #20340). The gain appears only if the intermediate values do not constantly pass through VRAM and the
   matmuls really sit on the matrix units. Our sequential kernel is at ~2.5 TFLOPS effective
   (estimate), far below the fp32 peak of 24 TFLOPS, and takes 1.1 s by measurement, so there is room.

## 1. State of the art

### 1.1 Algorithm sources

- **Parallelizing Linear Transformers with the Delta Rule over Sequence Length**, arXiv 2406.06484
  (Jun 2024, NeurIPS 2024). WY/UT representation, chunk form
  `S_[t+1] = S_[t] + (U − W S_[t]ᵀ)ᵀ K`, `O = Q S ᵀ + (QKᵀ ⊙ M)(U − W Sᵀ)`; `T = (I + tril(diag(β)KKᵀ, −1))⁻¹ diag(β)`;
  inverse by forward substitution; FLOP `O(LCd + Ld²)`, `L/C` sequential steps; C "mostly 64 or 128";
  stability: the transition matrix `I − βkkᵀ` has eigenvalues 1 and `1 − β‖k‖²` ∈ [0,1] for L2-normalized k, β ∈ (0,1),
  so it is non-expanding. The chunk form was ~2× faster there than the original recurrence kernel and
  grows with length/head dimension (no numbers in the text). <https://arxiv.org/html/2406.06484v6>
- **Gated Delta Networks**, arXiv 2412.06464 (Dec 2024). Scalar gate α_t plus delta rule; same WY chunk form
  with decay mask. <https://arxiv.org/abs/2412.06464>
- **Kimi Linear / KDA**, arXiv 2510.26692 (Oct 2025, v2). Per-channel gate, DPLR special case; kernel ~2× faster
  than general DPLR according to secondary sources. Only abstract/summaries seen
  (PDF not readable). <https://arxiv.org/abs/2510.26692>
- **Gated DeltaNet-2**, arXiv 2605.22791 (2026-05-21). Separates erase and write gate per channel, same
  WY chunk structure, C = 64, fused Triton. Numerics: decay in fp32 before the kernel, state fp32,
  **the triangular solve is the most precision-critical step** (precision flag). Throughput H100 38.0 → 36.1 Kt/s
  (1.3B hybrid, whole models, not kernel). <https://arxiv.org/html/2605.22791v1>
- **Tiled Flash Linear Attention (TFLA)**, arXiv 2503.14376 (Mar 2025, v3 Dec 2025, NeurIPS 2025). For mLSTM, not
  delta rule: a second level of parallelization *inside* the chunk allows larger chunks, fewer
  stored intermediate states, higher arithmetic intensity. Transferability to the delta rule unchecked
  (the triangular inverse couples tokens in the chunk). <https://arxiv.org/abs/2503.14376>
- **Kronecker Delta Rule** (KronLSTM), arXiv 2610.26834 (2026-10-19): new update rule, no kernel statement;
  listed for completeness only. <https://arxiv.org/abs/2610.26834>

### 1.2 Implementations

| Project | What | Source |
|---|---|---|
| **fla-org** `chunk_gated_delta_rule` | Triton. Steps: optional L2 norm → `gdn_gate_chunk_cumsum`/`chunk_local_cumsum` (scaling `RCP_LN2`, rest in base 2) → `chunk_gated_delta_rule_fwd_intra` (fused `kkt` + `solve_tril` + `recompute_w_u`) → `chunk_gated_delta_rule_fwd_h` (state per chunk) → `chunk_fwd_o`. Chunk 16/32/64, default 64. | <https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py> (read 2026-10-24) |
| fla `solve_tril` | (I+A)⁻¹: 16×16 diagonal blocks by forward substitution in fp32, off-diagonal blocks by matmul (`Ai21 = −Ai22·A21·Ai11`), four blocks merged for 64; output fp32 by default. Precision of the dot products via `FLA_TRIL_PRECISION` (`ieee`/`tf32`/`tf32x3`) | <https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/solve_tril.py> |
| fla `recompute_w_u_fwd` | `u = A (v⊙β)`, `w = A (k⊙β⊙2^g)`; operands cast to input dtype beforehand; TF32 off for `u` | `fla/ops/gated_delta_rule/wy_fast.py` |
| fla `chunk_delta_h` | State fp32 in registers, K in slices of 64, `v_new = u − w·h` (h is cast to w dtype), `v_new *= 2^(g_last − g_t)`, `h *= 2^g_last`, `h += kᵀ v_new`; **stored** chunk states in k dtype, final state fp32 | `fla/ops/common/chunk_delta_h.py` |
| fla `chunk_o` | `o = scale·(q h · 2^g + tril(q kᵀ ⊙ 2^(g_i − g_j)) v_new)`; difference in the exponent, never a quotient of two exponentials; fp32 accumulators | `fla/ops/common/chunk_o.py` |
| **FlashQLA** (Qwen) | TileLang, fused kernel for GDN chunked prefill, warpgroup specialization, algebraic rewrite (less tensor/CUDA-core/SFU load), "gate-driven intra-card CP". **2–3× forward vs. fla-Triton** on Hopper/Blackwell (SM90/100/103/120/121). NVIDIA only. | <https://github.com/QwenLM/FlashQLA> |
| FlashKDA (Moonshot) | 1.72–2.22× over fla-Triton on H20 (secondary source) | web search 2026-10-24, not verified |
| **vLLM** | Qwen3-Next uses fla-Triton kernels for the GDN layers (vendored in `vllm/model_executor/layers/fla/ops`) | <https://vllm.ai/blog/2025-09-11-qwen3-next> |
| **SGLang** | On SM90 standard Triton for prefill+decode; FlashInfer GDN prefill selectable (only chunks up to 8192 by default), CuTe DSL kernel only for decode/MTP | <https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8>, <https://lmsys.org/blog/2026-08-12-qwen3-8-day0-support/> |
| **llama.cpp** | Op `GGML_OP_GATED_DELTA_NET`. Chunk form in the graph (`src/models/delta-net-base.cpp`, `build_delta_net_chunking`, C = 64, KDA C = 16) and a fused autoregressive kernel. PR #20340 (merged 2026-03-11) allows chunk backends; CUDA #20391 (column split over warps, HIP-compatible warp size), Metal #20361, Vulkan #20334, WebGPU #20687, Hexagon #22837. Measurement in #20340 (DGX Spark): Kimi-Linear pp2048 2.11×, Qwen3.5-27B 1.14×, Qwen3-Next-80B 1.07–1.10×. Normalization fix #28068. | <https://github.com/ggml-org/llama.cpp/pull/20340> |
| AMD | **Nothing specific found**: neither AITER nor a HIP chunk kernel for GDN, no RDNA measurement (search 2026-10-24). | – |

## 2. Algorithm for our dimensions

Dimensions per piece: `T = 2048`, chunk `C = 64` → `NC = 32` chunks. `dk = dv = 128`. 16 K heads, 48 V heads,
V head `j` uses K head `i = j mod 16` (GGUF) or `⌊j/3⌋` (HF). State `S ∈ ℝ^{128×128}` per V head,
layout here `[dk, dv]` (`Sᵀk` is a dv vector), fp32.

The recurrence from `06-qwen38-architecture.md` §4 (order: decay first, then read):

```
S ← α_t S ;  u_t = β_t (v_t − Sᵀ k_t) ;  S ← S + k_t u_tᵀ ;  o_t = Sᵀ (q_t / √128)
α_t = exp(g_t),  g_t = ssm_a · softplus(a_t + dt_bias) ≤ 0,  β_t = σ(b_t)
```

Inputs after convolution+SiLU and L2 norm: `q,k` per K head `[T,128]`, `v` per V head `[T,128]`, `g, β` per V head `[T]`.
Pre-scale `q` with `1/√128`.

### Step 0 – Preparation (fp32, elementwise, cheap)
- `G_t = Σ_{i≤t in chunk} g_i` (inclusive, **restarting at 0 for each chunk**), `G_last = G_{C−1}`.
- Compute in base 2: `g2 = g·log2(e)`, then `exp2f` (native `v_exp_f32` on AMD). fla does exactly that.
- Helper quantities per token: `e_t = exp(G_t)`, `r_t = exp(G_last − G_t)` (≤ 1), `γ = exp(G_last)`.
- All exponents are ≤ 0 as long as `g ≤ 0` (guaranteed here by `ssm_a < 0`, `softplus ≥ 0`). **Never**
  form `exp(G_i)/exp(G_j)`, always `exp(G_i − G_j)` with `i ≥ j`. Underflow to 0 is harmless.

### Step 1 – Gram matrices (parallel over all chunks and heads)
Raw Gram matrices depend only on the **K head**, so once per 16 heads instead of 48:
```
Gkk = K_c K_cᵀ      [64,64]   (WMMA f16 → f32; K_c: [64,128])
Gqk = Q_c K_cᵀ      [64,64]
```
Per V head `j` (β_j, G_j):
```
D[t,s]  = exp(G_t − G_s)                      for t ≥ s, else 0   (fp32)
A_s     = tril( diag(β) · (Gkk ⊙ D), −1 )     [64,64]  strictly lower triangular, row t scaled by β_t
Aqk     = tril( Gqk ⊙ D, 0 )                  [64,64]  including the diagonal
```
Apply masks **before** `exp` or do not compute the upper half (llama.cpp sets to 0 → 1 before `exp` and masks afterwards).

### Step 2 – Triangular inverse (fp32, CUDA cores, not WMMA)
```
T = (I + A_s)⁻¹         [64,64]
```
Forward substitution row by row: `T[t,:] = e_t − Σ_{s<t} A_s[t,s] · T[s,:]`. That is ~64³/3 ≈ 87 kFMA per
chunk and head (estimate), negligible at 1536 chunk-heads per piece and layer. The fla variant with
16×16 block merge by matmul saves latency but costs precision when the merge matmuls run in f16.
Recommendation: pure fp32 substitution first (one wave per chunk-head, 64 serial steps), block merge only if needed.

### Step 3 – WY quantities (WMMA, `T` as f16 or f32→f16 when loading)
```
W = T · ( diag(β) · (K_c ⊙ e) )     [64,128]    (e_t row-wise)
U = T · ( diag(β) · V_c )           [64,128]
```
Derivation/check (2 tokens): `u_2 + A_21 u_1 = β_2 v_2` with `A_21 = β_2 (k_1·k_2) e^{G_2−G_1}` and
`w_2 = β_2 e^{G_2} k_2 − A_21 w_1`, hence `(I + A_s) W = diag(β)(K⊙e)`. Matches fla `recompute_w_u_fwd`
and llama.cpp (`k_cumdecay`).

Further chunk quantities: `Q̃ = diag(e) · Q_c` and `K̂ = diag(r) · K_c`, both [64,128] (only scaled copies,
better computed while loading into LDS than stored).

### Step 4 – Recurrence over chunks c = 0 … 31 (serial, per V head; **the only sequential part**)
With `S_c` = state before chunk c (`S_0` = input state of the piece, fp32):
```
(a)  V_new = U_c − W_c · S_c                        [64,128] = [64,128] − [64,128]·[128,128]     WMMA, 2·64·128·128 = 2.1 MFLOP
(b)  O_c   = Q̃_c · S_c + Aqk_c · V_new              [64,128]                                      WMMA, 2.1 + 1.05 MFLOP
(c)  S_{c+1} = γ_c · S_c + K̂_cᵀ · V_new            [128,128] = ... + [128,64]·[64,128]           WMMA, 2.1 MFLOP
```
Decay of `S` (`γ·S`) elementwise in fp32. `S_c` is needed as an f16 operand for (a),(b); for rounding
see section 3. The output `O_c` then goes unchanged into step 7 of the block (RMSNorm_128 ⊙ w ⊙ σ(z)).

**Parallelism:** the `S` columns (dv) are independent. `V_new[:,cols]`, `O[:,cols]` and `S[:,cols]` need only the
same columns of `U` and `S`, but **all** of `W`, `Q̃`, `K̂`, `Aqk`. One workgroup per (V head, dv slice of 32 or 64)
→ 48 × 4 = 192 workgroups on 32 CUs, `S` slice fp32 = 128×32×4 B = 16 KB (or 32 KB for 64 columns).

### Step 5 – Piece boundary and snapshots
The final state after chunk 31 is the input state of the next piece (fp32). The states `S_c` at chunk boundaries
(multiples of 64 tokens) come for free; for rollback with MTP/prefix cache (`10-prefix-cache-hybrid.md`) they can be
written out (3 MiB per layer in fp32, `06` §9). A rollback to an arbitrary token position = snapshot + up to 63 single steps.

### Step 6 – Reference for testing (NumPy sketch, one head, one chunk)
```python
G = np.cumsum(g); D = np.tril(np.exp(G[:,None]-G[None,:]))
A = np.tril(beta[:,None]*(K@K.T)*D, -1)
T = np.linalg.inv(np.eye(C)+A)
W = T @ (beta[:,None]*K*np.exp(G)[:,None]);  U = T @ (beta[:,None]*V)
Vn = U - W@S
O  = (Q*np.exp(G)[:,None])@S + np.tril((Q@K.T)*D)@Vn
S  = np.exp(G[-1])*S + (K*np.exp(G[-1]-G)[:,None]).T@Vn
```
Check against the token-by-token recurrence in float64: maximum relative deviation of `O` and `S`.

## 3. Numerics

| Quantity | Precision | Reason / source |
|---|---|---|
| `g`, `G`, `exp` factors, `β` | fp32 | Cumulative log decays are precision-sensitive (GDN-2, 2605.22791; fla computes the cumsum before the kernel) |
| `Gkk`, `Gqk` accumulator | fp32 (WMMA f16→f32) | Inputs f16 with L2 norm → values ≤ 1, well representable |
| `A_s`, triangular inverse `T` | **fp32** | "most precision-critical step" (GDN-2); fla output fp32, option `ieee` |
| `T`, `W`, `U`, `Aqk`, `Q̃`, `K̂`, `V_new` as WMMA operands | f16 (not bf16: 3 more mantissa bits, values ≤ ~O(1)–O(100), no range problem) | fla casts these to the input dtype (bf16); f16 is strictly more accurate |
| State `S` | **fp32** | Model: `mamba_ssm_dtype = float32` (`06` §1); fla accumulator fp32, final state always fp32 |
| `S` as WMMA operand | f16, optional hi/lo split | fla casts `h` to bf16 for `w@h`. f16 rounding ~5·10⁻⁴ relative per chunk step |

Pitfalls:
- **Never** quotients `e^{G_i}/e^{G_j}` or `e^{−G}`: with strong decay (e.g. `g ≈ −5` per token → `G ≈ −320`)
  this overflows/underflows. Always the difference in the exponent; mask the upper triangle before `exp` (otherwise `inf·0 = NaN`).
- **Padding** at the end of a piece (length not a multiple of 64): `β = 0`, `g = 0`, `k = v = 0` → padding does not write
  into the state and does not change `G_last` (llama.cpp does the same).
- **Conditioning of the inverse:** the structure (non-expanding transitions, 2406.06484) keeps `W`, `U` bounded
  as long as `k` is L2-normalized (mind the epsilon!) and `β ∈ (0,1)`. A `β` above 1 (`allow_neg_eigval` in fla: `β = 2σ`) does not
  occur here.
- **Head mapping** (`mod 16` vs. `⌊j/3⌋`) must match `k_dn_recur`, otherwise nothing is right.
- **A real error measurement is missing**: I found no published bf16-vs-fp32 measurement for WY kernels
  (GDN-2 and fla state the principle, no numbers). We have to measure ourselves (section 7).
- Possible remedies if perplexity suffers: (1) `S` as `S_hi + S_lo` (two f16) only for `W·S` and `Q̃·S`
  → 2× matmul cost for these steps; (2) smaller chunk 32 (less error accumulation in `T`, twice as many
  serial steps); (3) keep `T` in fp32 and compute step 3 with an f16 split.

## 4. FLOP balance (estimate, own calculation)

Per chunk and V head (C = 64, d = 128):

| Part | FLOP |
|---|---:|
| `Gkk`, `Gqk` (per K head, ⅓ share per V head) | 2 × 1.05 M / 3 ≈ 0.7 M |
| Inverse (fp32 FMA) | ~0.17 M |
| `W = T·…`, `U = T·…` | 2 × 1.05 M |
| `V_new = U − W S` | 2.1 M |
| `O = Q̃ S + Aqk V_new` | 2.1 M + 1.05 M |
| `S` update | 2.1 M |
| **Sum** | **~10.3 M** |

Per piece (32 chunks) and layer, 48 heads: ≈ 15.8 GFLOP; × 36 layers ≈ 570 GFLOP per piece; × 5.86 pieces
(12,000 tokens) ≈ **3.3 TFLOP**, of which the lion's share as f16 WMMA. (With Gram sharing by K head; without ~3.9 TFLOP.)
Sequential: ≈ 130 kFLOP per token and head (`Sᵀk`, rank-1 update, `Sᵀq`, decay) → 2.7 TFLOP fp32, serial chain
of 432,000 steps (12,000 tokens × 36 layers) in ~1.1 s ≈ 2.5 µs per step (all 48 heads in parallel).

So the FLOP count rises ~1.2–1.4×, but on the 4× faster matrix unit (93 vs. 24 TFLOPS), and the serial
chain shrinks from 2048 to 32 steps per piece. With C = 32 the intra-chunk FLOP drop by about half of the
Gram/T parts, the state matmuls (7.35 M) remain, the chain becomes twice as long.

**Memory** (12,000 tokens, 36 layers, estimate):
- Mandatory traffic: read q,k,v (10,240 × 2 B ≈ 20 KB/token) + write o (6,144 × 2 B = 12 KB/token) → ~14 GB ≈ 45 ms at 315 GB/s.
- **Unfused, two kernels** (the front kernel writes `W`, `U`, `Aqk`, `T` in f16; the recurrence reads): `W`+`U`+`Aqk` ≈ 40 KB per chunk-head
  → ×(write+read) ≈ 20 GB ≈ 65 ms. Acceptable, and the front kernel is parallel over all 32 chunks × 48 heads (1536 workgroups),
  fills the GPU well.
- **Naive fla structure** stores the state `h` per chunk (32 × 64 KB fp32 per head, 100 MB per piece and layer,
  write+read ≈ 42 GB ≈ 130 ms): **avoid**, keep the state in registers/LDS.

## 5. Variants with less memory traffic

1. **State in registers/LDS, dv slices** (recommended, see step 4): no `h` writes. The whole state
   128×128 fp32 = 64 KB = the whole LDS of one CU, hence slices of 32/64 columns.
2. **Two-stage as first expansion step:** kernel A (`prep`, fully parallel): steps 0–3, writes `W`, `U`, `Aqk` (f16).
   Kernel B (`scan`): step 4 serial over chunks, `S` in registers. Easy to test (intermediate values comparable with fla/NumPy).
3. **Fully fused (FlashQLA style):** steps 1–4 in the same workgroup per (head, dv slice). Cost: `T`, `W`, Gram parts
   are computed twice/several times per slice (4 slices × ~3 MFLOP) – acceptable against ~7.4 MFLOP of state work; saves
   the 65 ms intermediate traffic. FlashQLA reports 2–3× over fla-Triton through exactly this fusion + overlap of data transport/matmul/elementwise.
4. **Embedding conv/L2 norm/gates** into `prep` (fla: `use_qk_l2norm_in_kernel`, `use_gate_in_kernel`), saves separate passes.
5. **GQA sharing:** 3 V heads share `q`,`k`: load `K_c`, `Q_c`, `Gkk`, `Gqk` once.

LDS/register planning per workgroup (estimate, wave32, 4 waves, dv slice 32): `S` slice 16 KB, `W` 16 KB, `Q̃` 16 KB,
`K̂` 16 KB, `Aqk` 8 KB, `U`/`V_new` 4 KB → ~76 KB, i.e. above 64 KB: stream `W`, `Q̃`, `K̂` through LDS in dk slices of 64
(double buffered) or keep `S` in registers (16 KB / 4 waves = 32 fp32 VGPRs per lane).

## 6. New / innovative for a small consumer GPU

- **Gate-driven context parallelism (FlashQLA):** Because of the exponential decay, `S_c` for strongly decaying heads depends
  hardly on the distant state, so sections of the piece can be started approximately independently.
  Hardly needed here (192 workgroups on 32 CUs suffice), but interesting when working with smaller pieces.
  Source: <https://github.com/QwenLM/FlashQLA>.
- **TFLA idea** (second level inside the chunk, larger chunks, fewer intermediate states), arXiv 2503.14376, unchecked for the delta
  rule.
- **Prefill pipeline overlap:** `prep` for piece n+1 does not depend on the state and can overlap with `scan` of piece n
  (own idea, unchecked). Likewise `prep` for all 6 pieces depends only on the inputs.
- **f16 instead of bf16 in WMMA**: mantissa advantage over all references (fla bf16), free on gfx1200 (93 vs. 97 TFLOPS).
- **fp8 WMMA** (186 TFLOPS) for `W·S` etc. is not recommended because of the state precision; not investigated.

## 7. Open points / recommended measurements

1. Perplexity and `max|ΔO|` against `k_dn_recur` for: (a) all-fp32 reference in chunk form, (b) f16 operands, (c) hi/lo state, (d) C = 32 / 64.
2. Kernel A alone: time per piece; kernel B alone: latency per chunk step (target << 2.5 µs × 64 tokens = 160 µs per chunk equivalent).
3. Check whether the WMMA operand layouts (A: rows, B: columns) work with `S` in registers without an LDS detour.
4. Follow up the llama.cpp hint: the crossover chunk vs. sequential is device-dependent (PR #20340); for short pieces (<256 tokens) the sequential kernel stays.
5. DeltaProduct (arXiv 2502.10297) and further variants were **not** checked in this research.

## 8. Sources

- arXiv 2406.06484 – Parallelizing Linear Transformers with the Delta Rule over Sequence Length (Jun 2024): <https://arxiv.org/html/2406.06484v6>
- arXiv 2412.06464 – Gated Delta Networks (Dec 2024): <https://arxiv.org/abs/2412.06464>
- arXiv 2510.26692 – Kimi Linear (Oct 2025): <https://arxiv.org/abs/2510.26692>
- arXiv 2605.22791 – Gated DeltaNet-2 (2026-05-21): <https://arxiv.org/html/2605.22791v1>
- arXiv 2503.14376 – Tiled Flash Linear Attention (Mar 2025, v3 Dec 2025): <https://arxiv.org/abs/2503.14376>
- arXiv 2610.26834 – Kronecker Delta Rule (2026-10-19): <https://arxiv.org/abs/2610.26834>
- fla-org/flash-linear-attention, `fla/ops/gated_delta_rule/{chunk,wy_fast}.py`, `fla/ops/common/{chunk_delta_h,chunk_o}.py`, `fla/ops/utils/solve_tril.py` (main, read 2026-10-24): <https://github.com/fla-org/flash-linear-attention>
- QwenLM/FlashQLA: <https://github.com/QwenLM/FlashQLA>
- llama.cpp `src/models/delta-net-base.cpp` (master, 2026-10-24) and PR #20340 (merged 2026-03-11), #20391, #20361, #20334, #20687, #22837, #28068: <https://github.com/ggml-org/llama.cpp/pull/20340>
- vLLM blog Qwen3-Next (2025-09-11): <https://vllm.ai/blog/2025-09-11-qwen3-next>
- SGLang Cookbook Qwen3.8: <https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8>; LMSYS blog 2026-08-12: <https://lmsys.org/blog/2026-08-12-qwen3-8-day0-support/>
