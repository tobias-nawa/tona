## Plan (written before the run)

- Scope: workload huge (512000:200 1000000:200 at --ctx 1048576, TONA_SEQS=1); A `n3-main` against B `n3-pr`; rounds 2 to 2, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); no warm-up start.
- Why this size: select_checks.py names R7 for attn_wmma2.h, mega_seq.cpp, prefill_plan.cpp, prompt_cache.cpp: 512K and 1M are past 262K where 32-bit index overflows and plan limits would show (decode tokens per prompt as in the targets, default 200)
- Duration: 85-85 min (2-2 rounds of 42.7 min, estimate before round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: 2026-11-26-scale-run.md c4 (prefill 1,536 / 1,164, decode 60.8 / 57.2 at 512K / 1M); two rounds (ab.py minimum), no warm-up start, no noise table for huge):
  - decode: -1 to +1 %
  - prefill: -1 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit d43a0b3fa7; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/coder-iq1_m.bin
  - B: build commit f50cd70c7a; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/coder-iq1_m.bin
