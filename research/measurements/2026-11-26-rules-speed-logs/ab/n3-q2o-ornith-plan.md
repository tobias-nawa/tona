## Plan (written before the run)

- Scope: workload server (30000:200 at --ctx 140000); A `n3-pr` against B `n3-hard`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: second model: decode after a 30K prompt, the decode rate is the one hardening could touch through the host loop (decode tokens per prompt as in the targets, default 200)
- Duration: 3-8 min (2-6 rounds of 1.3 min, estimate before round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: Coder expectation of q2; no noise table for Ornith (--no-noise), at least 2 rounds):
  - decode: -0.5 to +0.5 %
  - prefill: -0.5 to +0.5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit f50cd70c7a; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/ornith-1.5-35b-a3b.bin
  - B: build commit f50cd70c7a; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/ornith-1.5-35b-a3b.bin
