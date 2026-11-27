## Plan (written before the run)

- Scope: workload short,set (2000:400 2000:write:400 at --ctx 140000; bench-set at --ctx 32768); A `b164t` against B `b164t`; rounds 6 to 8, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: decode cost of chunked slots: 6 to 8 rounds to narrow the CI of r1 (research/measurements/2026-11-18-vram-chunks-gtt.md addendum) (decode tokens per prompt as in the targets, default 200)
- Duration: 21-28 min (6-8 rounds of 3.5 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@2000:write sd 0.163 % (df 4), decode@2000:write sd 0.605 % (df 4)
- Expectation (B against A, source: 2026-11-18-vram-chunks-gtt.md addendum, r1 rounds 3 and 4):
  - decode: -0.5 to +0.5 %
  - prefill: -1 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-1106-gbc2589bb8b; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin
  - B: build commit v0.0.2-1106-gbc2589bb8b; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin, TONA_SLOTS_CHUNKED=1

## Result

A: `b164t` env -; B: `b164t` env {'TONA_SLOTS_CHUNKED': '1'}. Alpha 0.05, margin 0.5 %, rounds 8 of at most 8, stop: 8 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 1454.5 / 1451.0 / 1450.0 / 1454.5 / 1452.6 / 1452.6 / 1454.7 / 1449.9 | 1452.2 / 1453.0 / 1454.4 / 1454.5 / 1455.6 / 1454.1 / 1456.5 / 1455.0 | 1452.6 (1449.9-1454.7) | 1454.4 (1452.2-1456.5) | +0.13 % [-0.05, +0.31] | **equal** after round 1 |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 76.6 / 76.7 / 76.9 / 76.9 / 76.9 / 76.9 / 76.9 / 76.9 | 76.6 / 76.7 / 76.9 / 76.9 / 76.8 / 76.9 / 76.8 / 76.9 | 76.9 (76.6-76.9) | 76.9 (76.6-76.9) | -0.06 % [-0.25, +0.13] | **equal** after round 3 |
| prefill@2000:write: Prefill tok/s after 2000:write tokens of prompt | 1710.7 / 1725.1 / 1728.3 / 1723.3 / 1726.3 / 1726.0 / 1725.4 / 1725.6 | 1722.7 / 1711.6 / 1727.3 / 1727.0 / 1726.0 / 1727.3 / 1727.0 / 1727.1 | 1725.5 (1710.7-1728.3) | 1727.0 (1711.6-1727.3) | +0.04 % [-0.39, +0.47] | **equal** after round 7 |
| decode@2000:write: Decode tok/s after 2000:write tokens of prompt | 127.5 / 127.5 / 127.9 / 127.8 / 128.1 / 127.9 / 127.8 / 127.8 | 127.5 / 127.4 / 127.7 / 128.3 / 127.6 / 128.0 / 127.4 / 128.0 | 127.8 (127.5-128.1) | 127.7 (127.4-128.3) | -0.05 % [-0.58, +0.49] | open (undecided at the last round) |
| decode-code: Decode tok/s (bench-set code) | 94.6 / 95.5 / 95.5 / 95.6 / 95.5 / 95.6 / 95.5 / 95.5 | 94.7 / 94.9 / 95.5 / 95.6 / 95.5 / 95.6 / 95.6 / 95.5 | 95.5 (94.6-95.6) | 95.5 (94.7-95.6) | -0.05 % [-0.51, +0.40] | **equal** after round 6 |
| decode-text: Decode tok/s (bench-set text) | 68.0 / 68.7 / 68.7 / 68.7 / 68.7 / 68.7 / 68.7 / 68.6 | 68.0 / 68.1 / 68.7 / 68.7 / 68.7 / 68.7 / 68.7 / 68.6 | 68.7 (68.0-68.7) | 68.7 (68.0-68.7) | -0.11 % [-0.61, +0.39] | **equal** after round 6 |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@2000:write B vs A | decode@2000:write B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|---|---|
| 1 | AB | -0.16 % | -0.05 % | +0.70 % | -0.01 % | +0.11 % | +0.00 % | 210 |
| 2 | BA | +0.14 % | +0.03 % | -0.78 % | -0.12 % | -0.63 % | -0.87 % | 209 |
| 3 | AB | +0.30 % | -0.10 % | -0.06 % | -0.15 % | +0.00 % | +0.00 % | 209 |
| 4 | BA | +0.00 % | -0.05 % | +0.22 % | +0.37 % | +0.00 % | +0.00 % | 209 |
| 5 | AB | +0.20 % | -0.05 % | -0.02 % | -0.42 % | +0.00 % | +0.00 % | 209 |
| 6 | BA | +0.10 % | -0.03 % | +0.07 % | +0.14 % | +0.00 % | +0.00 % | 209 |
| 7 | AB | +0.12 % | -0.18 % | +0.09 % | -0.34 % | +0.10 % | +0.00 % | 209 |
| 8 | BA | +0.35 % | -0.05 % | +0.09 % | +0.17 % | +0.00 % | +0.00 % | 209 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | +0.13 % | -1 to +1 % | within the expectation |
| decode@2000 | -0.06 % | -0.5 to +0.5 % | within the expectation |
| prefill@2000:write | +0.04 % | -1 to +1 % | within the expectation |
| decode@2000:write | -0.05 % | -0.5 to +0.5 % | within the expectation |
| decode-code | -0.05 % | -0.5 to +0.5 % | within the expectation |
| decode-text | -0.11 % | -0.5 to +0.5 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 127532, 25317 / 127532, 25317 / 128026, 24823 / 128026, 24823 / 128026, 24823 / 128026, 24823 / 128026, 24823 / 128026, 24823 | 127532, 25317 / 127532, 25317 / 128026, 24823 / 128026, 24823 / 128026, 24823 / 128026, 24823 / 128026, 24823 / 128026, 24823 | NO |
| 2000:write | 80862, 29415 / 80862, 29415 / 81310, 28967 / 81310, 28967 / 81310, 28967 / 81310, 28967 / 81310, 28967 / 81310, 28967 | 80862, 29415 / 80862, 29415 / 81310, 28967 / 81310, 28967 / 81310, 28967 / 81310, 28967 / 81310, 28967 / 81310, 28967 | NO |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 145.5 | 28.3 | 8.3 | 0.0 | 0.0 | 0.8 | 23638.7 | 0.1 | 0.6 | 44 | 0.0 | 0.0 | - | 0.0 | - | - | 20.6 |
| 2 | A | 236.5 | 9.9 | 3.5 | 0.0 | 0.0 | 2.3 | 27530.8 | 0.0 | 0.0 | 49 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.0 |
| 3 | A | 139.2 | 22.1 | 6.2 | 0.0 | 0.0 | 0.9 | 27464.8 | 0.9 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.4 |
| 4 | A | 237.3 | 9.5 | 3.0 | 0.0 | 0.0 | 2.5 | 27487.1 | 0.3 | 0.0 | 49 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.2 |
| 5 | B | 138.6 | 22.3 | 3.9 | 0.0 | 0.0 | 0.9 | 27499.8 | 0.5 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.5 |
| 6 | B | 236.0 | 9.8 | 2.6 | 0.0 | 0.0 | 2.8 | 27477.8 | 0.0 | 0.0 | 49 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.2 |
| 7 | B | 138.6 | 22.1 | 3.7 | 0.0 | 0.0 | 1.1 | 27472.3 | 0.0 | 0.0 | 44 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.5 |
| 8 | B | 237.8 | 9.6 | 3.8 | 0.0 | 0.0 | 2.6 | 27432.7 | 0.0 | 0.0 | 49 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.1 |
| 9 | A | 142.4 | 22.3 | 4.2 | 0.0 | 0.0 | 1.0 | 27475.6 | 0.0 | 0.0 | 45 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.4 |
| 10 | A | 240.9 | 9.8 | 3.2 | 0.0 | 0.0 | 2.8 | 27480.6 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.2 |
| 11 | A | 136.5 | 22.1 | 2.3 | 0.0 | 0.0 | 0.9 | 27443.2 | 0.0 | 0.0 | 45 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.5 |
| 12 | A | 237.8 | 9.8 | 3.2 | 0.0 | 0.0 | 2.5 | 27480.3 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.2 |
| 13 | B | 138.7 | 22.1 | 2.0 | 0.0 | 0.0 | 1.1 | 27467.1 | 0.0 | 0.0 | 45 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.5 |
| 14 | B | 239.7 | 9.6 | 2.8 | 0.0 | 0.0 | 2.9 | 27467.9 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.2 |
| 15 | B | 139.3 | 22.1 | 2.3 | 0.0 | 0.0 | 0.9 | 27500.9 | 0.0 | 0.0 | 45 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.5 |
| 16 | B | 240.1 | 9.7 | 2.9 | 0.0 | 0.0 | 2.7 | 27475.4 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.1 |
| 17 | A | 138.0 | 22.3 | 3.3 | 0.0 | 0.0 | 0.9 | 27502.1 | 0.0 | 0.0 | 45 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.5 |
| 18 | A | 241.2 | 9.6 | 2.5 | 0.0 | 0.0 | 2.7 | 27450.5 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.2 |
| 19 | A | 139.5 | 22.1 | 2.5 | 0.0 | 0.0 | 1.1 | 27487.5 | 0.0 | 0.0 | 45 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.5 |
| 20 | A | 239.4 | 9.6 | 3.0 | 0.0 | 0.0 | 2.6 | 27451.2 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.2 |
| 21 | B | 136.5 | 22.5 | 2.6 | 0.0 | 0.0 | 1.3 | 27485.0 | 0.0 | 0.0 | 45 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.4 |
| 22 | B | 238.1 | 9.6 | 2.4 | 0.0 | 0.0 | 2.9 | 27457.8 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.1 |
| 23 | B | 138.5 | 22.1 | 2.5 | 0.0 | 0.0 | 1.2 | 27487.2 | 0.0 | 0.0 | 45 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.4 |
| 24 | B | 238.6 | 9.7 | 2.5 | 0.0 | 0.0 | 2.6 | 27435.1 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.2 |
| 25 | A | 138.3 | 22.1 | 2.5 | 0.0 | 0.0 | 1.2 | 27375.6 | 0.0 | 0.0 | 45 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.5 |
| 26 | A | 239.4 | 9.6 | 2.7 | 0.0 | 0.0 | 2.8 | 27435.3 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.1 |
| 27 | A | 138.1 | 22.0 | 2.8 | 0.0 | 0.0 | 1.1 | 27422.8 | 0.0 | 0.0 | 45 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.4 |
| 28 | A | 239.5 | 9.6 | 2.5 | 0.0 | 0.0 | 2.7 | 27470.8 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.1 |
| 29 | B | 136.8 | 22.5 | 2.5 | 0.0 | 0.0 | 1.2 | 27519.4 | 0.0 | 0.0 | 45 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.5 |
| 30 | B | 239.3 | 9.6 | 2.5 | 0.0 | 0.0 | 2.5 | 27434.1 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.2 |
| 31 | B | 137.1 | 22.3 | 2.5 | 0.0 | 0.0 | 1.1 | 27479.0 | 0.0 | 0.0 | 45 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.5 |
| 32 | B | 239.7 | 9.7 | 2.4 | 0.0 | 0.0 | 2.4 | 27386.5 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.2 |
| 33 | A | 137.5 | 22.2 | 2.4 | 0.0 | 0.0 | 1.2 | 27476.3 | 0.0 | 0.0 | 45 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.5 |
| 34 | A | 237.1 | 9.4 | 2.5 | 0.0 | 0.0 | 2.3 | 27437.7 | 0.0 | 0.0 | 50 | 0.0 | 0.0 | - | 0.0 | - | 1.1 | 20.1 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | 75eff9b8784c7d4c | 75eff9b8784c7d4c | yes | yes | yes | - |
| 2000:write | a6f07fd6fb86e448 | a6f07fd6fb86e448 | yes | yes | yes | - |

Runtime: 29.7 min for 8 rounds (3.5 min per round); a fixed 3 rounds would take 10.5 min (-17.4 min saved).

## Analysis against the plan (decision rule: decode CI inside -0.5 .. +0.5 %)

Test conditions (`hw_report.py`, `quiet_check.py` before and after; queue kind exclusive, desktop guard on, no requeue; ticket "166 r3 chunked longer", build `b164t`, 8 rounds, the maximum of the plan): see the end of this file. Quiet check before and after: warning only for about 2.1 GiB of VRAM held by the desktop.

| Metric (decode) | B vs A mean [95 % CI] | Inside -0.5 .. +0.5 %? |
|---|---|---|
| decode@2000 | -0.06 % [-0.25, +0.13] | yes |
| decode@2000:write | -0.05 % [-0.58, +0.49] | no, by 0.08 points (open at round 8) |
| decode-code (set) | -0.05 % [-0.51, +0.40] | no, by 0.01 points |
| decode-text (set) | -0.11 % [-0.61, +0.39] | no, by 0.11 points |

Prefill: equal or within +-1 % on all three (+0.13, +0.04 %; prefill@2000:write [-0.39, +0.47]). Tokens identical (SHA A = B, stable). Deterministic counters of the cache equal in every round per request.

Reading:
- The point estimates are -0.05 to -0.11 %, a tenth of the r1 values (-0.25 to -0.62 %). Medians per arm are equal (95.5 / 95.5 code, 68.7 / 68.7 text).
- The CI edges that miss the rule come from **one round**: round 2 (BA) with decode-code -0.63 % and decode-text -0.87 %; both arms show a low first measured set in their first appearance (A round 1: 94.6 / 68.0; B round 2: 94.9 / 68.1; all later rounds 95.5 / 68.7 in both arms). Rounds 3 to 8 are +0.00 on both set metrics except +0.10 once. The same pattern (a low first round, here -1.5 % in r1) was the r1 outlier. It is a start-up effect that hits the first set run of each arm, not a layout effect; this reading is from the per-round table, not tested separately. The plan says no round is dropped, so none was.
- Verdict by the **letter of the rule**: the CI of the two set metrics (and decode@2000:write) reaches the lower edge by 0.01 to 0.11 points, so chunked "stays a flag". Verdict by the **size of the effect**: decode loss is below 0.15 % on every metric, and the CI edge of -0.5 to -0.6 % is the width of the first-round start-up. Recommendation to the coordinator: the cost is at most a few tenths of a percent and probably zero; whether that clears "chunked becomes the default" (a decision about the stated rule, not about the data) is yours. I did not change any default.
- Not measured by this run: what chunked buys (that is level 0/1 of the compositor test; level 0 cannot show it, see `2026-11-18-vram-level0.md`).

## Test conditions

```
## hw_report 06:30:39
### Test conditions

- CPU: AMD Ryzen 9 9950X 16-Core Processor, 16 cores / 32 threads
- GPU: AMD 1002:13c0, gfx1036, 2.0 GiB VRAM, 0.02 GiB held by others before the run, 0 % busy before the run, integrated
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 2.1 GiB held by others before the run, 0 % busy before the run
- RAM: 60.5 GiB, 56.3 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.3.4-200.fc44.x86_64 (x86_64)
- Driver and toolkit: rocm 10.1.0
- TONA_* variables: `TONA_GPU_ARCH=gfx1200` `TONA_GPU_LOCK_HELD=1` `TONA_HIPCXX=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/lib/llvm/bin/amdclang++` `TONA_HIPFLAGS=-x hip --offload-arch=gfx1200 -isystem <repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/include` `TONA_HIPLDFLAGS=-L<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/lib -lamdhip64 -Wl,-rpath,<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/lib`
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 2.1, 5.32, 4.38
- CPU use by other programs over 0.5 s: 0.4 % of all threads, swap in use 1.16 GiB
- sandbox: no
- Commit: `56232f9` (clean tree)
## quiet before 06:30:40
- Quiet check (2026-11-18 06:30:41): warning
  - warning: 2.1 GiB of 15.9 GiB VRAM are held by others
## quiet after 07:00:20
- Quiet check (2026-11-18 07:00:21): warning
  - warning: 2.08 GiB of 15.9 GiB VRAM are held by others
```
