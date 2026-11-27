## Plan (written before the run)

- Scope: workload mid (30000:200 at --ctx 140000); A `w81-base` against B `w81`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: control for the decode@30000 finding of the first run: there the 30K request followed the 2K requests on the same server and wrote a fork/tail checkpoint (cold start of the server does not); alone it is a cold request without a checkpoint, so both arms must give the same SHA and equal rates (decode tokens per prompt as in the targets, default 200)
- Duration: 3-8 min (2-6 rounds of 1.3 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9)
- Expectation (B against A, source: single cold request, no stash to fork from: the old code path; first run of this question (2026-11-02-prefix-bitexact-ab.md) was inside the expectation for every metric except decode@30000, which had another text):
  - prefill: -0.5 to +0.5 %
  - decode: -0.5 to +0.5 %
- Active tooling (IE_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-204-g98dc08946e; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit v0.0.2-208-ga810892b7a-dirty; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0

## Result

A: `w81-base` env -; B: `w81` env -. Alpha 0.05, margin 0.5 %, rounds 3 of at most 6, stop: all metrics decided after round 3. All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1806.1 / 1805.8 / 1807.3 | 1806.3 / 1807.3 / 1806.9 | 1806.1 (1805.8-1807.3) | 1806.9 (1806.3-1807.3) | +0.03 % [-0.05, +0.11] | **equal** after round 1 |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 67.7 / 67.7 / 67.8 | 67.8 / 67.7 / 67.6 | 67.7 (67.7-67.8) | 67.7 (67.6-67.8) | -0.01 % [-0.51, +0.49] | **equal** after round 3 |

| Round | Order | prefill@30000 B vs A | decode@30000 B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | +0.01 % | +0.17 % | 79 |
| 2 | BA | +0.09 % | +0.06 % | 79 |
| 3 | AB | -0.02 % | -0.27 % | 79 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@30000 | +0.03 % | -0.5 to +0.5 % | within the expectation |
| decode@30000 | -0.01 % | -0.5 to +0.5 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 30000 | 67592, 9940 / 67592, 9940 / 67592, 9940 | 67592, 9940 / 67592, 9940 / 67592, 9940 | yes |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 30000 | 1fad1ead2f1ac34c | 1fad1ead2f1ac34c | yes | yes | yes | - |

Runtime: 4.6 min for 3 rounds (1.3 min per round); a fixed 3 rounds would take 4.0 min (+0.0 min saved).
