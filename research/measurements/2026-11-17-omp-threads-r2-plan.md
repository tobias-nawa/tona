## Plan (written before the run)

- Scope: workload short,mid,set (2000:400 2000:write:400 30000:200 at --ctx 140000; bench-set at --ctx 32768); A `b5afe` against B `b5afe`; rounds 2 to 4, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: thread count of the CPU expert path: decode on short and set, prefill on mid; 2 to 4 rounds (docs/contribute/measurement.md, 2026-11-17-omp-threads.md) (decode tokens per prompt as in the targets, default 200)
- Duration: 18-37 min (2-4 rounds of 9.2 min, estimate before round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@2000:write sd 0.163 % (df 4), decode@2000:write sd 0.605 % (df 4), prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9)
- Expectation (B against A, source: 2026-11-17-omp-threads.md expectation table):
  - decode: -0.5 to +4 %
  - prefill: -0.5 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-1063-g96b4c2075c; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=16, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin
  - B: build commit v0.0.2-1063-g96b4c2075c; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=12, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin
