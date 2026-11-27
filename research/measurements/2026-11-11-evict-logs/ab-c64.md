## Plan (written before the run)

- Scope: workload server (30000:200 at --ctx 262144); A `w103h` against B `w103h`; rounds 2 to 2, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: decode at 30K shows the loss (E10); one prompt size shortens the run (decode tokens per prompt as in the targets, default 200)
- Duration: 3-3 min (2-2 rounds of 1.3 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9)
- Expectation (B against A, source: E10 step 1: 4 MB pieces -27 %; 64 MB unknown, expecting between):
  - decode: -35 to +1 %
  - prefill: -1 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-370-gf977c4f594; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin, TONA_USERPTR_CHUNK_MB=0
  - B: build commit v0.0.2-370-gf977c4f594; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin, TONA_USERPTR_CHUNK_MB=64

## Result

A: `w103h` env {'TONA_USERPTR_CHUNK_MB': '0'}; B: `w103h` env {'TONA_USERPTR_CHUNK_MB': '64'}. Alpha 0.05, margin 0.5 %, rounds 2 of at most 2, stop: all metrics decided after round 2. All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1879.3 / 1878.9 | 1880.4 / 1879.3 | 1879.1 (1878.9-1879.3) | 1879.8 (1879.3-1880.4) | +0.04 % [-0.04, +0.12] | **equal** after round 1 |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 66.6 / 66.7 | 66.6 / 64.5 | 66.6 (66.6-66.7) | 65.6 (64.5-66.6) | -1.60 % [-3.13, -0.08] | **B slower** after round 2 |

| Round | Order | prefill@30000 B vs A | decode@30000 B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | +0.06 % | +0.02 % | 78 |
| 2 | BA | +0.02 % | -3.22 % | 78 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@30000 | +0.04 % | -1 to +1 % | within the expectation |
| decode@30000 | -1.60 % | -35 to +1 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 30000 | 71501, 6925 / 71501, 6925 | 71501, 6925 / 71501, 6925 | yes |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 30000 | 3a5ff5033f8f119b | 3a5ff5033f8f119b | yes | yes | yes | - |

Runtime: 3.3 min for 2 rounds (1.3 min per round); a fixed 3 rounds would take 3.9 min (+1.3 min saved).
