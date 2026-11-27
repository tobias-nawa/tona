## Plan (written before the run)

- Scope: workload mid,long (30000:200 130000:200 at --ctx 140000); A `w30dn` against B `w30dn`; rounds 1 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: mid (30K) and long (130K): prefill at the piece sizes of the server where the DN share was equal at 12K and 128K; decode is run by the workload and must not move (decode tokens per prompt as in the targets, default 200)
- Duration: 4-24 min (1-6 rounds of 4.0 min, estimate before round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9), prefill@130000 sd 0.018 % (df 9), decode@130000 sd 0.42 % (df 9)
- Expectation (B against A, source: 2026-11-03-dn-chunk.md Plan 2: k_dn_wave 10.3 % of prefill GPU time, kernel ratio 0.5-0.7; decode untouched):
  - prefill: +2.5 to +4.5 %
  - decode: -1 to +1 %
- Active tooling (IE_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-291-g18ad38f6e8-dirty; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_DIAG=1, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit v0.0.2-291-g18ad38f6e8-dirty; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_DIAG=1, IE_DN_CHUNK=1, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
