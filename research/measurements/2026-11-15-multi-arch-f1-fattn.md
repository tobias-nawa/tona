# F1: full-context attention steps (decode and prefill) for Ornith and Bolt, 2026-11-15

## Plan (written before the runs)

- Scope: `test_gpu_fattn` once (decode and prefill against the oracle primitives, hd 256 and hd 128, up to 262,144
  positions, same bits across grid size and token batching), then `test_gpu_fattn --bench` three times (decode time
  against the 315 GB/s bound, against the QSA path, prefill per token). One queue slot, about 2 minutes.
- Expectation: all cases ok (tolerance printed per case); decode within 2x of the bandwidth time at long context;
  full attention within +-30 % of the QSA path at 2048 positions; prefill per token comparable to the QSA kernel.
- Active tooling: none beyond `TONA_GPU_LOCK_HELD`; no diagnostics, no profiler.

## Prior art

- llama.cpp #30803 (HIP RDNA4, quantized KV read once per KV head, closed), open PRs #31802 (fused q8/q4 dequant in
  MMA FA), #31761 and #31746 (D256 FA tiles on RDNA); FlashAttention-2 (arXiv 2307.08691), FlashInfer (arXiv
  2501.01005), flash-decoding split-K merge, vLLM #57286 (split-KV on RDNA4); `research/34-prefill-attention-kernel.md`.
- Not adopted: sparse attention for hybrids (arXiv 2611.06133), INT2/INT4 KV (arXiv 2611.09722): other trade-offs.
- Search: tools/papers/scan.py topics v1; topics: long-context attention, GitHub issues/PRs; window: arXiv
  2025-11-15..2026-10-30, GitHub 2026-08-01..2026-11-14 (shorter than 12 months for GitHub).

## Test conditions

### Test conditions

- CPU: AMD Ryzen 5 9600X 6-Core Processor, 6 cores / 12 threads
- GPU: AMD Ryzen 5 9600X 6-Core Processor, gfx1036, 2.0 GiB VRAM, 0.02 GiB held by others before the run, 0 % busy before the run, integrated
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 1.22 GiB held by others before the run, 1 % busy before the run
- RAM: 60.5 GiB, 54.6 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.2.15-200.fc44.x86_64 (x86_64)
- Driver and toolkit: hip 7.16.26385-0000000
- TONA_* variables: `TONA_GPU_LOCK_HELD=1`
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 1.23, 4.38, 4.74
- CPU use by other programs over 0.5 s: 0.7 % of all threads, swap in use 2.11 GiB
- sandbox: no
- Commit: `ba2c03e` (clean tree)

quiet_check before:
- Quiet check (2026-11-15 16:13:42): warning
  - warning: 1.22 GiB of 15.9 GiB VRAM are held by others

quiet_check after:
- Quiet check (2026-11-15 16:14:08): warning
  - warning: 1.22 GiB of 15.9 GiB VRAM are held by others

## Results

### Correctness (N2), `test_gpu_fattn`, rc 0, 20 s

```
decode  hd 256 nh 16 pos0      0 x 1 tokens: o 2.2e-04 q 1.9e-07 kv 0.50 steps q8 0.50 steps, same bits yes: ok
decode  hd 256 nh 16 pos0      1 x 2 tokens: o 3.6e-04 q 2.0e-07 kv 0.50 steps q8 0.50 steps, same bits yes: ok
decode  hd 256 nh 16 pos0     15 x 1 tokens: o 5.3e-04 q 1.5e-07 kv 0.50 steps q8 0.50 steps, same bits yes: ok
decode  hd 256 nh 16 pos0     16 x 3 tokens: o 6.1e-04 q 1.6e-07 kv 0.50 steps q8 0.50 steps, same bits yes: ok
decode  hd 256 nh 16 pos0    100 x 4 tokens: o 7.6e-04 q 1.7e-07 kv 0.50 steps q8 0.50 steps, same bits yes: ok
decode  hd 256 nh 16 pos0   4000 x 3 tokens: o 7.1e-04 q 1.5e-07 kv 0.50 steps q8 0.50 steps, same bits yes: ok
decode  hd 256 nh 16 pos0  32767 x 4 tokens: o 9.6e-04 q 1.6e-07 kv 0.50 steps q8 0.50 steps, same bits yes: ok
decode  hd 256 nh 16 pos0 262140 x 3 tokens: o 7.9e-04 q 1.2e-07 kv 0.50 steps q8 0.50 steps, same bits yes: ok
prefill hd 256 nh 16 pos0      0 x 40 tokens: o 9.1e-04 (5 sampled tokens): ok
prefill hd 256 nh 16 pos0    100 x 70 tokens: o 1.6e-03 (5 sampled tokens): ok
prefill hd 256 nh 16 pos0   5000 x 33 tokens: o 1.3e-03 (5 sampled tokens): ok
prefill hd 256 nh 16 pos0 262104 x 40 tokens: o 1.1e-03 (5 sampled tokens): ok
decode  hd 128 nh 32 pos0      0 x 1 tokens: o 2.2e-04 q 0.0e+00 kv 0.50 steps q8 0.07 steps, same bits yes: ok
decode  hd 128 nh 32 pos0     63 x 1 tokens: o 1.1e-03 q 0.0e+00 kv 0.50 steps q8 0.50 steps, same bits yes: ok
decode  hd 128 nh 32 pos0   4097 x 4 tokens: o 1.7e-03 q 0.0e+00 kv 0.50 steps q8 0.50 steps, same bits yes: ok
decode  hd 128 nh 32 pos0  65536 x 2 tokens: o 9.0e-04 q 0.0e+00 kv 0.50 steps q8 0.50 steps, same bits yes: ok
decode  hd 128 nh 32 pos0 262140 x 3 tokens: o 1.2e-03 q 0.0e+00 kv 0.50 steps q8 0.50 steps, same bits yes: ok
prefill hd 128 nh 32 pos0      0 x 37 tokens: o 1.2e-03 (5 sampled tokens): ok
prefill hd 128 nh 32 pos0   3000 x 64 tokens: o 2.5e-03 (5 sampled tokens): ok
prefill hd 128 nh 32 pos0 262104 x 40 tokens: o 1.5e-03 (5 sampled tokens): ok
decode  hd 128 nh 16 pos0    130 x 2 tokens: o 6.2e-04 q 1.3e-07 kv 0.50 steps q8 0.50 steps, same bits yes: ok
decode  hd 128 nh 16 pos0   9000 x 1 tokens: o 6.9e-04 q 1.2e-07 kv 0.50 steps q8 0.50 steps, same bits yes: ok
prefill hd 128 nh 16 pos0    500 x 20 tokens: o 1.2e-03 (5 sampled tokens): ok
all ok
rc=0
```

### Speed (N3), three runs, decode steps (split + merge, prep excluded) per layer and pass of 1 token

| case | run 1 | run 2 | run 3 | vs 315 GB/s time |
|---|---|---|---|---|
| decode hd 256 nh 16 nkv 2,   2048 pos | 120.9 us | 120.1 us | 119.9 us | 16.14x |
| decode hd 256 nh 16 nkv 2, 262144 pos | 2494.8 us | 2485.2 us | 2483.4 us | 2.60x |
| decode hd 128 nh 32 nkv 2,   2048 pos | 105.5 us | 106.3 us | 105.6 us | 28.17x |
| decode hd 128 nh 32 nkv 2, 262144 pos | 1934.0 us | 1939.9 us | 1938.4 us | 4.03x |

Prefill (rate: prefill, per token and layer): 1024 tokens: QSA kernel 1.79 / 1.77 / 1.82 us, full attention 1.85 /
1.89 / 1.85 us; 2048 tokens: 3.15 vs 3.35 us; 8192 tokens: 12.66 us (full attention only). Decode at 2048 positions
(dense QSA path vs full attention, same cache, Coder heads): 129.2 / 128.9 / 128.5 vs 122.1 / 121.8 / 121.7 us.

### Machine code
C1: all Coder kernels identical to `work/build/base` with the two R5 `--rename` rules (217 functions in A, 249 in B;
only added probe kernels and a second `op_qsa_prep` body from the probe unit, the base body is present). N1: every
`k_mega` 192 VGPRs, LDS 32024 B (fixed array); the fattn tile needs 17344 B.

## Reading against the plan

- N2: as expected.
- Full attention vs QSA at 2048 positions and in prefill: as expected (decode 6 % faster, prefill 3-6 % slower).
- **N3 outside the expectation:** decode at 262,144 positions is 2.59x (hd 256) and 4.04x (hd 128) the bandwidth
  time, above the 2x limit; at 32K 3.5x and 5.1x. The spread over runs is below 1 %. Cause: ungeprueft; the tile
  does four block barriers per 16 cells and each of the 64 chunks streams serially, so per-block latency, not the
  memory system, probably bounds it. Next step: a deeper cell step per barrier (32 or 64 cells) and double-buffered
  staging; not done in this ticket.
