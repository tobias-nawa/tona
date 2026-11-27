## Plan (written before the run)

- Scope: workload set,mid (30000:200 at --ctx 140000; bench-set at --ctx 32768); A `p519c1ab` against B `w136b`; rounds 4 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: set is the production decode, mid the 30K case of the deterministic finding; 4-6 paired rounds resolve about 1 % (decode tokens per prompt as in the targets, default 200)
- Duration: 35-52 min (4-6 rounds of 8.7 min, estimate before round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9)
- Expectation (B against A, source: pass curve of this ticket: kernel without tile loops equals 519c1ab within 0.1 ms (23.29 against 23.24 ms at 1 token)):
  - decode-code: -1 to +1 %
  - decode-text: -1 to +1 %
  - decode@30000: -1 to +1 %
  - prefill@30000: -1 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-680-g519c1ab3a6; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin
  - B: build commit v0.0.2-738-g639e3290bd; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin

warm-up start of A (discarded)
round 1 (AB) done in 223 s
duration 15-22 min (4-6 rounds of 3.7 min, from round 1)
round 2 (BA) done in 224 s
round 3 (AB) done in 228 s
round 4 (BA) done in 225 s
round 5 (AB) done in 226 s
round 6 (BA) done in 225 s
raw data: work/ab/ab-prod136.json
noise table with this run added: work/ab/ab-prod136-noise.toml
not written; to take it: cp work/ab/ab-prod136-noise.toml bench/noise.toml
## Plan (written before the run)

- Scope: workload set,mid (30000:200 at --ctx 140000; bench-set at --ctx 32768); A `p519c1ab` against B `w136b`; rounds 4 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: set is the production decode, mid the 30K case of the deterministic finding; 4-6 paired rounds resolve about 1 % (decode tokens per prompt as in the targets, default 200)
- Duration: 15-23 min (4-6 rounds of 3.8 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9)
- Expectation (B against A, source: pass curve of this ticket: kernel without tile loops equals 519c1ab within 0.1 ms (23.29 against 23.24 ms at 1 token)):
  - decode-code: -1 to +1 %
  - decode-text: -1 to +1 %
  - decode@30000: -1 to +1 %
  - prefill@30000: -1 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-680-g519c1ab3a6; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin
  - B: build commit v0.0.2-738-g639e3290bd; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin

## Result

A: `p519c1ab` env {'TONA_EXPERT_PROFILE': '<repo>/work/profile/experts-coder.bin'}; B: `w136b` env {'TONA_EXPERT_PROFILE': '<repo>/work/profile/experts-coder.bin'}. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1886.8 / 1887.6 / 1886.5 / 1886.7 / 1879.6 / 1886.4 | 1887.1 / 1887.6 / 1885.1 / 1886.9 / 1877.4 / 1883.2 | 1886.6 (1879.6-1887.6) | 1886.0 (1877.4-1887.6) | -0.06 % [-0.13, +0.02] | **equal** after round 1 |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 69.7 / 72.4 / 68.9 / 71.6 / 72.7 / 72.3 | 73.5 / 73.1 / 72.2 / 72.8 / 72.4 / 70.9 | 71.9 (68.9-72.7) | 72.6 (70.9-73.5) | +1.75 % [-0.56, +4.06] | open (undecided at the last round) |
| decode-code: Decode tok/s (bench-set code) | 92.8 / 90.5 / 91.8 / 91.0 / 91.3 / 90.9 | 92.8 / 93.0 / 91.0 / 91.2 / 90.1 / 91.9 | 91.2 (90.5-92.8) | 91.6 (90.1-93.0) | +0.32 % [-1.37, +2.01] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 67.4 / 66.2 / 66.8 / 66.4 / 66.9 / 65.6 | 67.3 / 67.3 / 66.1 / 65.4 / 66.4 / 67.0 | 66.6 (65.6-67.4) | 66.7 (65.4-67.3) | +0.06 % [-1.67, +1.78] | open (undecided at the last round) |

| Round | Order | prefill@30000 B vs A | decode@30000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +0.02 % | +5.45 % (off plan) | +0.00 % | -0.15 % | 223 |
| 2 | BA | +0.00 % | +0.99 % | +2.76 % | +1.66 % | 224 |
| 3 | AB | -0.07 % | +4.73 % (outlier) | -0.87 % | -1.05 % | 228 |
| 4 | BA | +0.01 % | +1.65 % (outlier) | +0.22 % | -1.51 % | 225 |
| 5 | AB | -0.11 % | -0.40 % | -1.31 % | -0.75 % | 226 |
| 6 | BA | -0.17 % | -1.92 % (outlier) | +1.10 % | +2.13 % | 225 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@30000 | -0.06 % | -1 to +1 % | within the expectation |
| decode@30000 | +1.75 % | -1 to +1 % | **OUTSIDE the expectation: finding** |
| decode-code | +0.32 % | -1 to +1 % | within the expectation |
| decode-text | +0.06 % | -1 to +1 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 30000 | 64071, 10216 / 64071, 10216 / 64071, 10216 / 64071, 10216 / 64071, 10216 / 64071, 10216 | 64071, 10216 / 64071, 10216 / 64071, 10216 / 64071, 10216 / 64071, 10216 / 64071, 10216 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 87.7 | 29.6 | 12.9 | 0.0 | 0.0 | 0.6 | 29884.1 | 0.0 | 0.0 | 46 | 0.0 | 0.0 | - | 0.0 | - | - | 20.0 |
| 2 | A | 177.1 | 12.0 | 16.3 | 0.0 | 0.0 | 4.0 | 27031.5 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.0 |
| 3 | A | 88.1 | 33.0 | 20.4 | 0.0 | 0.0 | 0.6 | 27454.8 | 0.0 | 0.0 | 46 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.1 |
| 4 | A | 176.7 | 11.2 | 13.5 | 0.0 | 0.0 | 3.7 | 27112.7 | 0.0 | 0.0 | 51 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.2 |
| 5 | B | 89.2 | 30.9 | 17.3 | 0.0 | 0.0 | 0.7 | 28232.0 | 0.0 | 0.0 | 46 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.0 |
| 6 | B | 176.4 | 11.6 | 12.0 | 0.0 | 0.0 | 4.1 | 27398.9 | 0.0 | 0.0 | 51 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.4 |
| 7 | B | 88.4 | 20.7 | 13.9 | 0.0 | 0.0 | 0.8 | 27600.9 | 0.0 | 0.0 | 46 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.0 |
| 8 | B | 176.2 | 11.1 | 15.1 | 0.0 | 0.0 | 3.7 | 27135.0 | 0.0 | 0.0 | 51 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.3 |
| 9 | A | 88.5 | 20.5 | 10.5 | 0.0 | 0.0 | 0.6 | 27799.8 | 0.0 | 0.6 | 46 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.0 |
| 10 | A | 178.3 | 13.0 | 16.9 | 0.0 | 0.0 | 3.4 | 26995.1 | 0.0 | 0.9 | 50 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.1 |
| 11 | A | 85.9 | 48.6 | 73.2 | 0.0 | 0.0 | 0.8 | 29281.3 | 0.0 | 0.8 | 45 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 22.1 |
| 12 | A | 176.8 | 11.7 | 17.7 | 0.0 | 0.0 | 4.1 | 27305.7 | 0.0 | 0.7 | 51 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.7 |
| 13 | B | 84.6 | 19.2 | 14.8 | 0.0 | 0.0 | 0.5 | 27204.8 | 0.0 | 0.2 | 46 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.6 |
| 14 | B | 175.4 | 17.2 | 26.2 | 0.0 | 0.0 | 3.4 | 27362.3 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.4 |
| 15 | B | 84.5 | 19.3 | 14.6 | 0.0 | 0.0 | 0.6 | 27156.3 | 0.1 | 0.0 | 46 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.6 |
| 16 | B | 175.3 | 15.6 | 21.8 | 0.0 | 0.0 | 3.8 | 26963.6 | 0.0 | 0.0 | 51 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.4 |
| 17 | A | 87.8 | 19.3 | 14.9 | 0.0 | 0.0 | 0.6 | 27297.1 | 0.0 | 0.0 | 46 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.1 |
| 18 | A | 175.6 | 10.3 | 12.7 | 0.0 | 0.0 | 4.1 | 26909.7 | 0.0 | 0.0 | 51 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.3 |
| 19 | A | 84.3 | 19.3 | 13.1 | 0.0 | 0.0 | 0.5 | 27330.8 | 0.0 | 0.0 | 46 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.6 |
| 20 | A | 177.7 | 24.6 | 42.5 | 0.0 | 0.0 | 3.9 | 28415.3 | 0.0 | 0.0 | 51 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.0 |
| 21 | B | 84.2 | 20.8 | 11.6 | 0.0 | 0.0 | 0.7 | 27702.0 | 0.0 | 0.0 | 46 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.7 |
| 22 | B | 177.9 | 25.6 | 34.2 | 0.0 | 0.0 | 3.5 | 27395.4 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.1 |
| 23 | B | 84.7 | 22.0 | 16.0 | 0.0 | 0.0 | 0.7 | 27502.6 | 0.0 | 0.0 | 46 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.6 |
| 24 | B | 174.1 | 13.1 | 6.1 | 0.0 | 0.0 | 4.4 | 27024.0 | 0.0 | 0.0 | 51 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.2 |
| 25 | A | 84.4 | 19.2 | 2.6 | 0.0 | 0.0 | 0.6 | 27299.8 | 0.0 | 0.0 | 46 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.5 |
| 26 | A | 183.6 | 13.0 | 6.4 | 0.0 | 0.0 | 3.9 | 27750.6 | 1.9 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.1 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 30000 | ee58880d3f5b558f | ee58880d3f5b558f | yes | yes | yes | - |

Runtime: 24.4 min for 6 rounds (3.8 min per round); a fixed 3 rounds would take 11.3 min (-11.3 min saved).

rc=0
