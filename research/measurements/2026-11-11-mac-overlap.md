# 2026-11-11: GPU bandwidth under SSD load, GPU kernel time after idle gaps, M1 Max (#107)

Step M0 of the Mac backend proposal ([../40-mac-backend-proposal.md](../40-mac-backend-proposal.md)). Two open
questions decide how much expert reading can hide behind GPU work:
- does SSD DMA at 13 GB/s slow the GPU's memory reads (Flash-MoE claims −73 % on an M3 Max, [../39-mac-engines-2026-11.md](../39-mac-engines-2026-11.md) §3)?
- does the GPU lose clock while it waits for the CPU?
Speed numbers on this machine are provisional (background load).

## Plan (written before the run)

**Scope.** Tool `bench/mac/overlap.swift`, test file of 16 GiB in `work/` written by `bench/mac/ssdbench.cpp`
(deleted afterwards).

| Block | What runs | Rounds | Duration |
|---|---|---|---|
| O1 | GPU reads a 1 GiB buffer in command buffers of 4 passes (~13 ms); phases: GPU alone 2 s → GPU while 16 threads `pread` random 2.046 MB blocks with `F_NOCACHE` 2 s → SSD alone 2 s → GPU alone 2 s. Median GB/s per phase | 3 runs | ~9 s each |
| O2 | busy phase (one 13 ms read command buffer), idle gap g = 0, 0.1, 0.3, 1, 3, 10, 30, 100 ms (`usleep`), then a probe: ALU-bound kernel of ~0.5 ms as its own command buffer; GPU time of the probe. g = 0 still contains one command buffer round trip (~170 µs, mac-dispatch) | 3 runs × 20 per gap | ~10 s each |

30 s pause between runs.

**Expectation.**

| Rate | Expected | Tolerance | Source |
|---|---|---|---|
| O1 GPU alone | 330 GB/s | 315 to 340 | 2026-11-11-mac-bandwidth (328 to 335) |
| O1 GPU during SSD reads | −3 % against alone | 0 to −10 % | 13 GB/s is ~3 % of the 409.6 GB/s bus (estimate); −73 % (Flash-MoE) would be a finding |
| O1 SSD during GPU load against alone | 13 GB/s both | 11.5 to 13.5, difference < 10 % | 2026-11-11-mac-ssd (12.9 to 13.2) |
| O2 probe after g ≤ 0.3 ms against g = 0 | same | ±10 % | short gaps below a DVFS step (knowledge, unchecked) |
| O2 probe after g ≥ 3 ms | 1.0 to 3× of g = 0 | - | a rise means the backend needs a keep-busy kernel or no idle gaps |

**Active tooling and conditions.** `swiftc -O`, no Metal validation, no capture, no profiler. AC power. Load before
the runs (12:51): load average 10.2 / 8.3 / 8.0 on 10 cores; memory free 75 % (`memory_pressure`); 102 GB free on the
SSD. The second project on this Mac keeps running; a file scanner may read the test file while it is written.

### Addendum to the plan (written after O1 and O2, before O2b)

O2 showed the probe at 3.3× its busy time after 30 to 100 ms of idle. O2b measures how long the ramp lasts.

| Block | What runs | Rounds | Duration |
|---|---|---|---|
| O2b | idle 100 ms or 1 s, then 40 probes (~0.27 ms each when fast) in their own command buffers, committed at once; GPU time and start of each | 3 runs × 3 repetitions per idle time | ~5 s each |

| Rate | Expected | Tolerance | Source |
|---|---|---|---|
| probes until the time is back at ~271 µs | after ~2 to 5 ms of GPU work | 1 to 20 ms | DVFS steps act in ms (knowledge, unchecked) |

## Results

Raw data: [2026-11-11-mac-overlap.jsonl](2026-11-11-mac-overlap.jsonl). Tool: `bench/mac/overlap.swift`.

**Conditions (12:51 to 12:57).** AC power. Load average 2.7 to 11.1 on 10 cores, falling during the runs (noted per
run in the shell log: O2 6.7 / 7.7 / 8.7, O1 11.1 / 7.1 / 4.4, O2b 3.8 / 2.8 / 2.7 (1-minute values)). Test file 16 GiB
written at 2.15 GB/s, deleted after O1.

**O1, GPU read bandwidth and SSD throughput** (median of the ~13 ms command buffers per phase):

| Run | GPU alone | GPU during SSD reads (min) | GPU after | SSD during GPU load | SSD alone |
|---|---|---|---|---|---|
| 1 | 335.7 | 299.1 (288.2) | 335.2 | 7.60 | 11.31 |
| 2 | 336.1 | 284.6 (276.4) | 335.6 | 12.71 | 13.21 |
| 3 | 336.5 | 284.7 (278.5) | 336.2 | 13.18 | 13.28 |
| Expected | 315 to 340 | 0 to −10 % | | 11.5 to 13.5 | 11.5 to 13.5 |

**O2, probe GPU time after an idle gap** (µs, median of 20; busy level 271):

| Gap | 0 | 0.1 ms | 0.3 ms | 1 ms | 3 ms | 10 ms | 30 ms | 100 ms |
|---|---|---|---|---|---|---|---|---|
| Run 1 | 271.2 | 271.1 | 271.2 | 271.4 | 224.1 | 256.9 | 832.3 | 904.5 |
| Run 2 | 270.8 | 271.0 | 271.0 | 271.4 | 224.7 | 273.9 | 724.9 | 902.8 |
| Run 3 | 271.1 | 271.0 | 271.2 | 271.4 | 223.9 | 273.3 | 902.1 | 903.4 |

**O2b, ramp after long idle** (40 probes back to back; 9 sequences per idle time over 3 runs): the first probe takes
902 to 923 µs after 100 ms and after 1 s alike; the time stays at ~900 µs, steps to ~451 µs and reaches the busy
level (< 300 µs) at probe 19 to 32, i.e. after **13.6 to 20.7 ms** of GPU work (median ~16 ms). Expected 1 to 20 ms.

- **SSD reads cost the GPU 11 to 15 % of its read bandwidth** (336 → 285 to 299 GB/s), more than the expected 3 to 10 %
  (finding), but far from Flash-MoE's −73 %. Reading experts during GPU work pays as long as the hidden read time is
  worth more than 11 to 15 % of the GPU's memory time in that window.
- **The SSD keeps its rate under GPU load** in runs 2 and 3 (12.7 to 13.2 against 13.2 to 13.3 GB/s). Run 1 was lower
  in both phases (7.6 / 11.3), right after the file was written (cause unchecked: write-back or a file scan).
- **Finding: the GPU drops to a third of its clock after 30 ms of idle and needs ~14 to 21 ms of work to come back.**
  Gaps up to 10 ms cost nothing, so per-layer event waits (55 µs to a few ms) inside a pass do not trigger it. It hits
  the first pass after a pause, e.g. the first token of a request or a pass after a long SSD stall: up to ~3× slower
  for its first ~15 ms (estimate from the probe). The Mac backend should start a short warm-up kernel when a request
  arrives, before tokenization and prefill planning finish, instead of a permanent spin kernel.
- Unexplained: after a 3 ms gap the probe is 17 % faster than right after the busy phase (224 against 271 µs), in all
  three runs. A guess (unchecked): after the memory-bound busy kernel the GPU runs at a state tuned for memory, and an
  ALU kernel gets a higher clock once that state is left.
