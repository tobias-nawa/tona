## Plan (written before the run)

- Scope: workload mid,long (30000:200 130000:200 at --ctx 140000); A `n3-main` against B `n3-main`; rounds 1 to 4, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: control: is a difference of 0.1 % between two starts of the same binary possible (start order, page cache); same two requests as q1b (decode tokens per prompt as in the targets, default 200)
- Duration: 4-16 min (1-4 rounds of 4.0 min, estimate before round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9), prefill@130000 sd 0.018 % (df 9), decode@130000 sd 0.42 % (df 9)
- Expectation (B against A, source: A/A of the same build: 0 within the noise of bench/noise.toml; control for the -0.12 % of prefill@130000 in q1):
  - decode: -0.5 to +0.5 %
  - prefill: -0.5 to +0.5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit d43a0b3fa7; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/coder-iq1_m.bin
  - B: build commit d43a0b3fa7; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/coder-iq1_m.bin
