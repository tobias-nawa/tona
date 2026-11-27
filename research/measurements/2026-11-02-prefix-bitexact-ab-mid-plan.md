## Plan (written before the run)

- Scope: workload mid (30000:200 at --ctx 140000); A `w81-base` against B `w81`; rounds 1 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: control for the decode@30000 finding of the first run: there the 30K request followed the 2K requests on the same server and wrote a fork/tail checkpoint (cold start of the server does not); alone it is a cold request without a checkpoint, so both arms must give the same SHA and equal rates (decode tokens per prompt as in the targets, default 200)
- Duration: 1-8 min (1-6 rounds of 1.3 min, estimate before round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9)
- Expectation (B against A, source: single cold request, no stash to fork from: the old code path; first run of this question (2026-11-02-prefix-bitexact-ab.md) was inside the expectation for every metric except decode@30000, which had another text):
  - prefill: -0.5 to +0.5 %
  - decode: -0.5 to +0.5 %
- Active tooling (IE_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-204-g98dc08946e; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit v0.0.2-208-ga810892b7a-dirty; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
