## Plan (written before the run)

- Scope: workload server (2000:400 13000:200 30000:200 at --ctx 140000); A `w100` against B `w100`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: same sizes as before; same build in both arms, only the switch differs; second attempt after fusing the weight conversion (dequant_fp8) and keeping the hc paths in f16 (decode tokens per prompt as in the targets, default 200)
- Duration: 4-12 min (2-6 rounds of 1.9 min, estimate before round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9)
- Expectation (B against A, source: profile with the switch: window 6.40 s against 6.88 s (-7 %), k_gemm2 + k_gemm8 1.85 -> 1.29 s, extra k_dequant_fp8 0.14 s and k_to_fp8):
  - prefill: +5 to +10 %
  - decode: -1 to +1 %
- Active tooling (IE_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-240-g04843157d6-dirty; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_DIAG=1, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit v0.0.2-240-g04843157d6-dirty; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_DIAG=1, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, IE_PREFILL_FP8=1, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
