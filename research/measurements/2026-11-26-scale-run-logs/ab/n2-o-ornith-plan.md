## Plan (written before the run)

- Scope: workload server (2000:200 16000:200 64000:100 at --ctx 262144); A `scale1126` against B `scale1126`; rounds 2 to 3, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: the offer.json rows of Ornith (2K, 16K, 64K code summary, MTP head from the file); targets as 2026-11-20-ornith-bolt-clean-state.md; 2 to 3 rounds from noise.toml (decode tokens per prompt as in the targets, default 200)
- Duration: 5-7 min (2-3 rounds of 2.5 min, estimate before round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: A/A, same build: B against A is 0 within the known noise):
  - decode: -3 to +3 %
  - prefill: -3 to +3 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-1468-g0f0736635d; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, HSA_COREDUMP_PATTERN=<worktree>/work/diag/gpucore-%p.core, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/ornith-1.5-35b-a3b.bin
  - B: build commit v0.0.2-1468-g0f0736635d; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, HSA_COREDUMP_PATTERN=<worktree>/work/diag/gpucore-%p.core, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/ornith-1.5-35b-a3b.bin
