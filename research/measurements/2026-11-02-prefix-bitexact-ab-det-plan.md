## Plan (written before the run)

- Scope: workload mid,long (30000:200 130000:200 at --ctx 140000); A `w81-base` against B `w81`; rounds 1 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: R4: mid and long are the workloads whose SHA bench/sha.toml holds; a cold prompt of one message must reproduce them (decode tokens per prompt as in the targets, default 200)
- Duration: 4-24 min (1-6 rounds of 4.0 min, estimate before round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9), prefill@130000 sd 0.018 % (df 9), decode@130000 sd 0.42 % (df 9)
- Expectation (B against A, source: single-message cold prompts run the old path; SHAs of main in bench/sha.toml):
  - prefill: -0.5 to +0.5 %
  - decode: -0.5 to +0.5 %
- Active tooling (IE_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-204-g98dc08946e; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_DETERMINISTIC=1, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit v0.0.2-208-ga810892b7a-dirty; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_DETERMINISTIC=1, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
