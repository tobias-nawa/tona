## Plan (written before the run)

- Scope: workload mid,short (2000:400 2000:write:400 30000:200 at --ctx 140000); A `merged` against B `w83`; rounds 1 to 6, alternating AB/BA, stop when the primary metrics (prefill) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: the cap loss showed at a 30K prefill (-20 %) and at 130K (-27 %), so mid shows it; short adds decode and expert counters at 2K (decode tokens per prompt as in the targets, default 200)
- Duration: 2-11 min (1-6 rounds of 1.8 min, estimate before round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@2000:write sd 0.163 % (df 4), decode@2000:write sd 0.605 % (df 4), prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9)
- Expectation (B against A, source: s86u1-3, s86f1-2 against s86d1-2 in 2026-11-01-slow-decode.md: GTT with all experts pinned, prefill 16.8-16.9 s against 16.8 s):
  - prefill: -1 to +1 %
  - decode: -2 to +2 %
- Active tooling (IE_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-137-ge4c1010521; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit v0.0.2-147-gcce920d197; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
