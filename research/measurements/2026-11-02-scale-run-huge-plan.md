## Plan (written before the run)

- Scope: workload huge (512000:200 1000000:200 at --ctx 1048576, IE_SEQS=1); A `merged` against B `w83`; rounds 2 to 2, alternating AB/BA, stop when the primary metrics (decode,prefill) are decided (alpha 0.05, margin 0.5 %); no warm-up start.
- Why this size: past 262K the registered share of the pinned experts grows (1.2 GB at 140K, 4.7 GB at 262K); 512K and 1M are the sizes where earlier notes have numbers (decode tokens per prompt as in the targets, default 200)
- Duration: 85-85 min (2-2 rounds of 42.7 min, estimate before round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: 2026-10-31-long-context-topk.md (512K: prefill 1434, decode 60.5) and -yarn.md section 2 (1M); 2026-11-01-slow-decode.md (register route)):
  - decode: -2 to +0 %
  - prefill: -2 to +1 %
- Active tooling (IE_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-148-g8a08c6b73e; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, HSA_COREDUMP_PATTERN=./work/diag/gpucore-%p.core, IE_DIAG=1, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit v0.0.2-167-g2a12ac4a5e; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, HSA_COREDUMP_PATTERN=./work/diag/gpucore-%p.core, IE_DIAG=1, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
