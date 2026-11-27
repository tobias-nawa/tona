## Plan (written before the run)

- Scope: workload server (2000:400 13000:200 30000:200 at --ctx 140000); A `merged` against B `w100`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: 13K is the size of the profile and of agent turns; 2K and 30K are the sizes with known noise (noise.toml); decode is checked at every size (decode tokens per prompt as in the targets, default 200)
- Duration: 4-12 min (2-6 rounds of 1.9 min, estimate before round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9)
- Expectation (B against A, source: run 3 plan in research/measurements/2026-11-02-prefill-int8.md (profile 7.41 s -> window -5 %)):
  - prefill: +2 to +8 %
  - decode: -1 to +1 %
- Active tooling (IE_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-234-gbcde1e5836; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_DIAG=1, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit v0.0.2-239-g83b28c9782; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_DIAG=1, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
