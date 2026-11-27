## Plan (written before the run)

- Scope: workload mid,short (2000:400 2000:write:400 30000:200 at --ctx 140000); A `w83` against B `w83`; rounds 1 to 6, alternating AB/BA, stop when the primary metrics (decode) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: the loss showed at 2K and 30K in short and mid; the same workloads show whether it goes (decode tokens per prompt as in the targets, default 200)
- Duration: 2-11 min (1-6 rounds of 1.8 min, estimate before round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@2000:write sd 0.163 % (df 4), decode@2000:write sd 0.605 % (df 4), prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9)
- Expectation (B against A, source: 2026-11-02-86-ab-prefill.md: register route against main, decode -0.9 % and -1.2 %):
  - decode: -1.5 to -0.5 %
  - prefill: -1 to +1 %
- Active tooling (IE_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-149-gb18e69ad53; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, HSA_USERPTR_FOR_PAGED_MEM=1, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit v0.0.2-149-gb18e69ad53; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
