# Coder sparse prefill attention `k_attn_wmma` (2026-11-20, #175, analysis #78)

Kernel: `k_attn_wmma` (`engine/gpu/prefill_attn.hip`), the QSA prefill attention of the Coder (the reference model): per token and KV head one
block, at most 2,051 selected cells gathered from the int8 KV cache (in VRAM up to about 77K tokens, in pinned RAM above 1 GiB of cache).
Starting point: `research/34-prefill-attention-kernel.md` (shape, trace numbers, ranked ideas) and
`research/measurements/2026-10-30-prefill-attn.md` (idea 1a/1b, the pipelined gather, was already built in #78: it helps only with the cache in VRAM
(-7 % kernel time) and is 3-6 % slower in RAM, so the engine keeps the direct gather there; the kernel in RAM is limited by PCIe and by how much
of the gathered data the L2 holds). Goal here: faster in both placements, **byte-equal output** first.

## Test conditions

Output of `python3 tools/hw_report.py` before the first run (2026-11-19 23:38): RX 9060 XT gfx1200, 15.9 GiB VRAM, 1.83 GiB held by the desktop,
1 % busy; Ryzen 9 9950X, 53.3 GiB RAM available, load average 7.5 / 8.4 / 6.5 (a build of another worker ran; GPU jobs queue exclusively),
swap 1.87 GiB in use, no `TONA_*` set, commit `60d3fcd` clean. Quiet check: only the warning "1.83 GiB of 15.9 GiB VRAM are held by others" (the
desktop). Microbenchmarks are kernel timings in one process (median of launches after a warm-up), no server.

## Plan 1: baseline microbench (written before the run)

Scope: `test_gpu_attn_wmma --bench` (existing: the frozen kernel against production, 1,024 tokens per launch, selections drift / churn / random,
positions 4,000 / 60,000 / 128,000 / 250,000, KV in VRAM and in pinned RAM), one `exclusive` job, build `sa` at `60d3fcd`. Purpose: the starting
numbers for every later comparison, in the same units (us per token-layer).
Expectation (source: `2026-10-30-prefill-attn.md` table, old kernel at 128,000): VRAM 6.7 / 6.9 / 7.2 us (drift / churn / random), RAM 10.0 / 26.5 /
41.9 us, tolerance +-10 % (other clocks and desktop state). A value outside it is a finding, not noise.
Active tooling: none (no profiler, no `TONA_DIAG`).

## Result 1: baseline microbench (job `sa-base`, build `sa` at `60d3fcd`, 23:5x; quiet check before the first job: only the desktop's 1.83 GiB VRAM)

`test_gpu_attn_wmma --bench` (median of launches, us per token-layer; "ref" = frozen kernel of 75f85ac, "new" = production: pipelined gather with the
cache in VRAM, direct gather in RAM; "new" in RAM is the direct kernel plus the LDS-only barriers of #163):

| Selection | pos | VRAM ref | VRAM new | RAM ref | RAM new |
|---|---:|---:|---:|---:|---:|
| drift | 4,000 | 5.77 | 4.94 | 8.25 | 7.66 |
| drift | 60,000 | 6.68 | 5.88 | 10.11 | 9.35 |
| drift | 128,000 | 6.74 | 5.89 | 10.02 | 9.15 |
| drift | 250,000 | 6.73 | 5.87 | 10.06 | 9.27 |
| churn | 60,000 | 6.86 | 5.89 | 25.65 | 24.39 |
| churn | 128,000 | 6.93 | 5.92 | 26.51 | 25.18 |
| churn | 250,000 | 6.97 | 5.91 | 26.76 | 25.74 |
| random | 60,000 | 6.96 | 5.91 | 39.62 | 38.74 |
| random | 128,000 | 7.31 | 6.25 | 42.05 | 41.32 |
| random | 250,000 | 7.31 | 7.02 | 42.71 | 41.90 |

Inside the expectation (+-10 %) except where #163 helped: RAM new is 0.91-0.98 of ref (the barrier change), VRAM new 0.85-0.90 (pipelined gather
of #78 plus #163). The numbers to beat: VRAM about 5.9 us, RAM drift about 9.2 us, churn about 25 us, random about 41 us (this one is the PCIe limit
and cannot move bit-exact).

ISA of production (`scripts/isa-diff.py` tooling, `k_attn_wmma<false>`): 104 VGPRs, 30,144 B LDS (2 blocks per CU), the main loop is about
500 instructions per wave and 32-cell step: per K element `v_bfe_i32 + v_cvt_f32_i32 + v_fma_mix` (about 2.8), V elements the same plus one
`ds_store_b16` each (46 `ds_store_b16` and 16 `ds_store_b16_d16_hi` per iteration), 8 `ds_bpermute_b32` (the 16-wide `__shfl_xor` of the softmax:
an LDS round trip each, with `s_wait_dscnt`), 5 barriers. At about 500 instructions per wave, 8 waves per block, 130 block-iterations per
token-layer on 64 SIMDs the issue floor is about 3.4 us at 2.9 GHz against 5.9 measured in VRAM: the kernel is instruction-bound to roughly 60 %,
so fewer instructions per iteration are the lever in VRAM; in RAM the unique bytes (PCIe) bound it.

## Plan 2: bit-exact second kernel `k_attn_wmma2` (`engine/gpu/attn_wmma2.h`), correctness and microbench (written before the run)

Change (all bit-exact by construction, checked by bytes): dequantization by `fma(x + 128, s, -128 s)` with `v_cvt_f32_ubyteN` (no sign
extension, about 2 instead of 2.8 instructions per element; exact because the fma rounds the exact product once, to the same f32); a cell
past the last one dequantizes to +0 by a zero scale (no branches); V staged by pairs of cells (one 32-bit LDS store of two halves per
dimension instead of two 16-bit ones); S^T = K Q^T (the lane holds eight cells of one row: two 16-byte stores instead of eight 4-byte ones);
softmax maximum and sum exchanged by DPP (`row_xmask`) instead of `ds_bpermute`; running maximum and sum in registers (one barrier less per
step); 32-bit row offsets. Three gathers: V after S (as the first kernel), V with K (research/34 idea 1a), next rows ahead (the pipelined one).
Scope: (1) `test_gpu_attn_wmma` (frozen kernel against the gathers 0, 1, 2 of the new kernel, the first kernel direct and pipelined and the
production choice, 12 selections x VRAM / RAM, bytes) plus an exhaustive device check of the dequantization identity (256 values x 2 M scales);
one `gpu` job; (2) `test_gpu_attn_wmma --bench` with the three gathers next to the first kernel and the frozen one: selections drift / churn /
random, positions 4,000 / 60,000 / 128,000, VRAM and RAM, 1,024 tokens, median of 7 launches, two alternating rounds, one `exclusive` job.
Expectation: bytes equal everywhere; the instruction count per iteration falls from about 500 to about 400 (-20 %), so VRAM about -10 to -20 %
(5.9 us -> 4.7 to 5.3 us), RAM drift -3 to -10 % (9.2 -> 8.3 to 8.9, the unique bytes over PCIe bind from about 5.4 us), RAM churn and random
unchanged within 3 % (PCIe); the gathers: V with K a little better than V after S (one round trip less), ahead best in VRAM and not better in
RAM (as in #78). Tolerance of the microbench itself +-2 % between alternating rounds. Decision: any byte difference stops the work and the
cause is found; a variant slower than the first kernel in some cell of the table is not used there; the choice per placement is the best
column. Active tooling: none.

## Result 2: second kernel, bytes and microbench (jobs `sa-cmp-v4`, `sa-bench-v4`, build `sa` with `attn_wmma2.h`; quiet check before each job: only the desktop's 1.8 GiB VRAM)

**Bytes:** `test_gpu_attn_wmma`: 12 selections x VRAM / RAM x 7 kernels (frozen kernel against the second kernel with the gathers V late, V with
K, next rows ahead, K rows ahead; the first kernel direct and pipelined; the production choice): **all identical**, byte for byte, in every case
(25 lines identical). Dequantization identity `fma(x + 128, s, -128 s)` against `(float)x * s` to f16: 0 of 536,870,912 cases differ (every int8
value, 2 M scales spread over 2^-27 ... 2^2).

Microbench (median of 7 launches, best of two alternating rounds, us per token-layer; this session the clocks are higher than in Result 1: the
frozen kernel reads 4.8 instead of 5.8 us, so compare columns of one row):

| Selection, pos, KV | v1 direct | v1 pipelined | v2 V late | v2 V with K | v2 ahead | v2 K ahead |
|---|---:|---:|---:|---:|---:|---:|
| drift 128K VRAM | 4.47 | 3.76 | 3.17 | 2.95 | **2.70** | 2.85 |
| churn 128K VRAM | 4.77 | 3.95 | 3.87 | 3.71 | **3.66** | 3.74 |
| random 128K VRAM | 6.10 | 6.22 | **6.07** | 6.17 | 6.29 | 6.24 |
| drift 128K RAM | 7.91 | 7.73 | 7.56 | **7.30** | 8.19 | 7.66 |
| churn 128K RAM | 25.15 | 30.74 | **23.97** | 25.79 | 33.20 | 26.86 |
| churn 60K RAM | 24.62 | 29.99 | **22.97** | 25.02 | 32.54 | 26.25 |
| random 128K RAM | 40.48 | 47.48 | **40.67** | 43.04 | 48.48 | 44.94 |
| drift 4K RAM | 5.99 | 6.23 | 5.16 | 4.93 | 5.88 | **4.68** |

Against the first kernel as production uses it (pipelined in VRAM, direct in RAM): **VRAM -28 % (drift, 3.76 -> 2.70 us), -7 % (churn), 0 (random,
VRAM bandwidth bound); RAM -4.5 % (drift, 7.91 -> 7.56 with V late), -4.7 % (churn, 25.15 -> 23.97), +0.5 % (random, PCIe bound)**. In RAM every
variant that keeps more requests in flight is slower in the churn and random cases (ahead +33 % over late, K ahead +12 %): the unique bytes over
PCIe bind and more outstanding requests cost, as in #78 (the pipelined gather of the first kernel is +22 % there). The RAM choice is V late, the
VRAM choice is ahead. The expectation of plan 2 (VRAM -10 to -20 %, RAM drift -3 to -10 %) was met in RAM and exceeded in VRAM.
Registers: V late 99 VGPRs, V with K 99, K ahead 129, ahead 143 (the first kernel 104 and 119), LDS 31,104 B (30,144 B), no spills.

## Plan 3: server speed with `ab.py` (written before the runs; two `exclusive` jobs, build `sa` at `b6b807a` in both arms)

A = `TONA_ATTN_WMMA_OLD=1` (first kernel: pipelined gather in VRAM, direct in RAM), B = default (second kernel: ahead in VRAM, V late in RAM),
Coder, MTP head and expert profile as default, alternating AB/BA, one discarded warm-up start, `--drafter` default. Rounds: `bench/noise.toml` has
only `prefill@130000` (sd 0.018 %, df 9) and `decode@130000` for this model; for 16K and 64K no entry, so at least 2 rounds, stop when decided
(alpha 0.05, margin 0.5 %), at most 4.
Job 1 (KV in VRAM: ctx 70000 puts the cache under 1 GiB): targets `16000:100 64000:50`, `--ctx 70000`. Job 2 (KV in pinned RAM): target
`128000:50`, `--ctx 140000` (the cache is 1.94 GB, the RAM path). Why this size: the kernel's cost per token grows with the position, so 16K
(small share), 64K (VRAM path, largest share in VRAM) and 128K (RAM path, where it is 26 % of the prefill) are the three points; decode tokens
are few (50 to 100) because decode does not run this kernel: it is checked at the same lengths, not targeted. 250K is skipped (about 190 s per start;
the 128K point answers the RAM path and the curve beyond is the same kernel).
Expectation (B against A; source: Result 2 per token-layer applied to research/34 shares, kernel at 64K about 10-15 % of the VRAM prefill, at
128K 26 %): prefill@16000 +0.5 to +2 %, prefill@64000 +2 to +6 %, prefill@128000 +0.5 to +2 % (-4.5 to -5 % of 24 s of 92 s); decode at all three
lengths -0.5 to +0.5 %; output SHAs of A and B equal (byte-equal kernel). Decision: a slower prefill or decode beyond the margin, or a SHA
difference, takes the kernel out of the default (the old one stays selectable).
Active tooling: `TONA_ATTN_WMMA_OLD=1` in A; no `TONA_DIAG`; quiet check before and after each job.

## Result 3: Coder server, `ab.py`, A = `TONA_ATTN_WMMA_OLD=1`, B = default (jobs `sa-ab-vram`, `sa-ab-ram`; build `sa` at `b6b807a`; `work/ab/sa-vram.md`, `work/ab/sa-ram.md`)

Gates run before (build `sa`, all exit 0): `scripts/golden.sh compare` (decode), `GOLDEN_MODE=prefill GOLDEN_DRAFTER=dflash scripts/golden.sh compare`,
`scripts/golden-server.sh compare`, `test_gpu_fattn`, `test_gpu_mamba`, `test_gpu_moe_latent`, `test_gpu_bolt_quant`, `test_gpu_attn_wmma`: identical.
Quiet check before the jobs showed the load of a build that the queue then ran before the exclusive job (queue order), afterwards the warning
"1.83 GiB of 15.9 GiB VRAM held by others" (desktop) only; desktop VRAM 1.8 GiB.

| Rate | A median (tok/s) | B median (tok/s) | B vs A [95 % CI] | Rounds | Plan | Verdict |
|---|---:|---:|---|---|---|---|
| **prefill** @16000 (KV in VRAM, ctx 70000) | 1959.7 | 1994.3 | **+1.75 % [+1.06, +2.43]** (per round +1.67 ... +1.82) | 4 | +0.5 to +2 % | as planned |
| **prefill** @64000 (KV in VRAM) | 1928.9 | 1965.5 | **+1.89 % [+1.20, +2.57]** (+1.85 ... +1.93) | 4 | +2 to +6 % | slightly below: the kernel is a smaller share of the VRAM prefill than assumed from the older RAM traces |
| **prefill** @128000 (KV in pinned RAM, ctx 140000) | 1896.7 | 1931.2 | **+1.83 % [+0.99, +2.67]** (+1.81 ... +1.84) | 3 | +0.5 to +2 % | as planned (upper end) |
| **decode** after 16K / 64K / 128K | 67.4 / 67.7 / 67.4 | 67.3 / 67.6 / 67.5 | -0.09 / -0.12 / +0.47 % (the last: one round +1.3 %, the others +0.1 and +0.0) | 4 / 4 / 3 | -0.5 to +0.5 % | inside noise (CIs include 0) |

Output SHAs of A and B equal at 16000, 64000 and 128000 prompt tokens (f8ad4e04..., 038e48a1..., c34f9af1...), stable in every round; expert cache
hits and misses and tokens per pass identical in every round. So the kernel is byte-equal in the server too; decode does not run it and did not move.

## Summary and what stays open

- **Kernel** (`k_attn_wmma2`, byte-equal to the first kernel): VRAM -28 % at 128K drift (3.76 -> 2.70 us per token-layer), RAM -4.5 to -5 % with the
  V-late gather (the pinned-RAM path is bound by the unique bytes over PCIe, random selection unchanged). **Server prefill +1.8 to +1.9 %** at 16K,
  64K and 128K, decode unchanged.
- `TONA_ATTN_WMMA_OLD=1` restores the first kernel (pipelined in VRAM, direct in RAM).
- Open (bit-exact, not built): (1) LDS 31.1 KB keeps two blocks per CU; overlaying the partial scores on the K buffer would give 22 KB (five blocks
  per WGP) at the price of one more barrier per step; (2) in RAM more requests in flight cost time (ahead +33 %), so the gain there is capped by PCIe
  traffic: only fewer bytes help (the token union of research/34 idea 2, number-changing, off by default, needs the quality check of docs/backends/README.md#checks
  and the overlap statistic u(4) of real selections, which nobody has measured; not built, maintainer decides); (3) `IE_KV_HOST=0` policy
  (research/34 idea 4) is a separate ticket. A cell dump of real selections would replace the synthetic drift/churn/random brackets.
- Prior art re-checked in this work: aiter #6576 (union kernel for QSA) and the others of research/34 section 4 are numeric (union) or not for gfx12; nothing
  published gives a bit-exact route below the PCIe limit. Idea 1a/1b of research/34 were already built in #78 (the pipelined gather).
