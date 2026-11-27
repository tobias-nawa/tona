## Plan (written before the run)

- Scope: workload mid,long (30000:200 130000:200 at --ctx 140000); A `w85-base` against B `w85`; rounds 1 to 2, alternating AB/BA, stop when the primary metrics (decode) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: R4: the 30K and 130K requests of bench/sha.toml (decode tokens per prompt as in the targets, default 200)
- Duration: 4-8 min (1-2 rounds of 4.0 min, estimate before round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9), prefill@130000 sd 0.018 % (df 9), decode@130000 sd 0.42 % (df 9)
- Expectation (B against A, source: bits must be equal (bench/sha.toml); rates of two rounds are not judged):
  - decode: -3 to +3 %
  - prefill: -3 to +3 %
- Active tooling (IE_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-204-g98dc08946e; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_DETERMINISTIC=1, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit v0.0.2-210-ge338d14fbd; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_DETERMINISTIC=1, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
