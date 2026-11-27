## Plan (written before the run)

- Scope: workload mid,long (30000:200 130000:200 at --ctx 140000); A `w81-base` against B `w81`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: R4: mid and long are the workloads whose SHA bench/sha.toml holds; a cold prompt of one message must reproduce them (decode tokens per prompt as in the targets, default 200)
- Duration: 8-24 min (2-6 rounds of 3.9 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9), prefill@130000 sd 0.018 % (df 9), decode@130000 sd 0.42 % (df 9)
- Expectation (B against A, source: single-message cold prompts run the old path; SHAs of main in bench/sha.toml):
  - prefill: -0.5 to +0.5 %
  - decode: -0.5 to +0.5 %
- Active tooling (IE_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-204-g98dc08946e; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_DETERMINISTIC=1, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit v0.0.2-208-ga810892b7a-dirty; HIP_PATH=./work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, IE_DETERMINISTIC=1, IE_EXPERT_PROFILE=./work/profile/experts-coder.bin, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0

## Result

A: `w81-base` env {'IE_DETERMINISTIC': '1'}; B: `w81` env {'IE_DETERMINISTIC': '1'}. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1806.5 / 1806.6 / 1806.6 / 1806.4 / 1806.5 / 1806.6 | 1807.0 / 1806.9 / 1805.5 / 1806.2 / 1806.7 / 1806.1 | 1806.6 (1806.4-1806.6) | 1806.4 (1805.5-1807.0) | -0.01 % [-0.06, +0.04] | **equal** after round 1 |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 71.3 / 71.5 / 71.5 / 71.3 / 71.5 / 71.2 | 71.4 / 71.4 / 71.3 / 71.4 / 71.6 / 71.4 | 71.4 (71.2-71.5) | 71.4 (71.3-71.6) | +0.01 % [-0.33, +0.36] | **equal** after round 4 |
| prefill@130000: Prefill tok/s after 130000 tokens of prompt | 1714.8 / 1714.8 / 1715.1 / 1715.1 / 1715.3 / 1714.5 | 1715.9 / 1717.0 / 1716.3 / 1716.2 / 1716.5 / 1716.0 | 1714.9 (1714.5-1715.3) | 1716.3 (1715.9-1717.0) | +0.08 % [+0.05, +0.11] | **B faster** after round 1 (real, but inside the margin) |
| decode@130000: Decode tok/s after 130000 tokens of prompt | 64.7 / 64.5 / 64.9 / 64.8 / 65.2 / 65.0 | 63.7 / 64.8 / 64.7 / 64.3 / 64.5 / 64.5 | 64.9 (64.5-65.2) | 64.5 (63.7-64.8) | -0.68 % [-1.38, +0.03] | open (undecided at the last round) |

| Round | Order | prefill@30000 B vs A | decode@30000 B vs A | prefill@130000 B vs A | decode@130000 B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +0.03 % | +0.08 % | +0.06 % | -1.57 % (off plan) | 237 |
| 2 | BA | +0.02 % | -0.19 % | +0.13 % | +0.40 % | 236 |
| 3 | AB | -0.06 % | -0.31 % | +0.07 % | -0.30 % | 236 |
| 4 | BA | -0.01 % | +0.12 % | +0.07 % | -0.75 % | 237 |
| 5 | AB | +0.01 % | +0.06 % | +0.07 % | -1.10 % | 236 |
| 6 | BA | -0.03 % | +0.30 % | +0.09 % | -0.75 % | 236 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@30000 | -0.01 % | -0.5 to +0.5 % | within the expectation |
| decode@30000 | +0.01 % | -0.5 to +0.5 % | within the expectation |
| prefill@130000 | +0.08 % | -0.5 to +0.5 % | within the expectation |
| decode@130000 | -0.68 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 30000 | 67129, 8953 / 67129, 8953 / 67129, 8953 / 67129, 8953 / 67129, 8953 / 67129, 8953 | 67129, 8953 / 67129, 8953 / 67129, 8953 / 67129, 8953 / 67129, 8953 / 67129, 8953 | yes |
| 130000 | 76065, 11270 / 76065, 11270 / 76065, 11270 / 76065, 11270 / 76065, 11270 / 76065, 11270 | 71526, 8498 / 71526, 8498 / 71526, 8498 / 71526, 8498 / 71526, 8498 / 71526, 8498 | NO |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 30000 | 08f0bd31e0c0277f | 08f0bd31e0c0277f | yes | yes | yes | 08f0bd31e0c0277f (B = ref) |
| 130000 | e9545d70f2400b87 | 536592a9aa29bd56 | yes | yes | NO | e9545d70f2400b87 (B DIFFERS) |

**WARNING: deterministic mode, but the texts differ between rounds, arms or from the reference of main. The change is not bit-exact (or the run is not deterministic).**

Runtime: 25.6 min for 6 rounds (3.9 min per round); a fixed 3 rounds would take 11.8 min (-11.8 min saved).
