## Plan (written before the run)

- Scope: workload server (2000:200 16000:100 64000:50 at --ctx 262144); A `scale1126` against B `scale1126`; rounds 2 to 2, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: the offer.json rows of the Coder (2K, 16K, 64K code summary, MTP); same targets as 2026-11-20-coder-decode.md; A/A gives the spread; 2 rounds is the minimum (decode tokens per prompt as in the targets, default 200)
- Duration: 5-5 min (2-2 rounds of 2.4 min, estimate before round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: A/A, same build: B against A is 0 within the known noise):
  - decode: -3 to +3 %
  - prefill: -3 to +3 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-1468-g0f0736635d; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, HSA_COREDUMP_PATTERN=<worktree>/work/diag/gpucore-%p.core, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/coder-iq1_m.bin
  - B: build commit v0.0.2-1468-g0f0736635d; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, HSA_COREDUMP_PATTERN=<worktree>/work/diag/gpucore-%p.core, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/coder-iq1_m.bin
