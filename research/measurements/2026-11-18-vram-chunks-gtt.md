# A/B of chunked slot buffers (C) and the GTT margin (D) for the desktop VRAM reserve (#166)

Plan, written before any run. Results and test conditions are added below the line `## Results

Run files: `2026-11-18-vram-chunks-gtt-d1/d2/r1/r2.md` (+ `.json`, `-logs/`, `-plan.md`), goldens in `/var/tmp/claude/166/run-g1-*.log` (identical lines quoted below).
Build `b164t`, commit stamp `v0.0.2-1106-gbc2589bb8b` (the `tona-server.commit` of #167 shows it in every arm's log; the code is `fce155f`, the plan commit adds files only).
All jobs ran exclusive through the queue, desktop guard on, no requeue was needed. Quiet check before and after every job: warning only for 2.06 GiB of VRAM held by the desktop;
hw_report: Ryzen 9 9950X (16 cores / 32 threads), load average 1.65 / 1.57 / 1.62 before r2, CPU use by other programs 0.1 %, swap in use 1.19 GiB.

### Tokens (Q1 and Q2): bit-exact

| Check | Result |
|---|---|
| g1-gpu: `TONA_SLOTS_CHUNKED=1`, `golden.sh compare` (csv, rust, explain) | all identical, rc 0 |
| g1-cpu: the same with `GOLDEN_PLAN=cpu` | all identical, rc 0 |
| d1: `TONA_DETERMINISTIC=1`, A default against B chunked, short and mid, 2 rounds | SHA equal in A and B and stable for 2000, 2000:write, 30000; the 30000 SHA equals the reference of main (ee58880d3f5b558f) |
| d2: the same with `TONA_GTT_MARGIN_MB=3072` | the same three SHAs, equal, stable, 30000 equals the reference |

Deterministic counters of the starts: A, chunked B and margin B all load with 70 expert slots per layer, VRAM 13.03 GB, 27.4 GB pinned (context 140000); the margin arm logs
"2.4 GB of the pinned experts do not fit the GTT and are registered as userptr" (2.4 of 2.4 GB on huge pages), as the arithmetic (3072 - 1024 MiB, rounded to pieces) predicted;
the A and chunked arms log no overflow. **Not confirmed from the logs:** that the chunked arm built its core from chunks. The chunk line needs `TONA_VRAM_LOG=1`, which the arms
did not set (it is a tooling variable the plan excluded); a short confirmation job (`TONA_SLOTS_CHUNKED=1 TONA_VRAM_LOG=1`, one start) ran after the chain, see the end of this section.

### Speed (r1, C chunked; r2, D margin 3072), 4 rounds each, alternating AB/BA

| Metric | r1 A | r1 B (chunked) | r1 B against A [95 % CI] | r2 A | r2 B (margin 3072) | r2 B against A [95 % CI] |
|---|---:|---:|---|---:|---:|---|
| prefill@2000 | 1,449.5 | 1,452.2 | +0.43 % [-0.13, +0.98] | 1,452.3 | 1,451.2 | +0.05 % [-0.30, +0.40] |
| decode@2000 | 76.9 | 76.4 | -0.42 % [-1.20, +0.35] | 76.8 | 76.9 | +0.21 % [-0.28, +0.70] |
| prefill@2000:write | 1,725.5 | 1,726.3 | +0.09 % [-0.15, +0.34] (equal) | 1,721.3 | 1,725.7 | +0.25 % [-0.16, +0.66] |
| decode@2000:write | 127.8 | 128.0 | +0.42 % [-0.55, +1.39] | 127.7 | 127.9 | +0.28 % [-0.77, +1.34] |
| prefill@30000 | 1,913.9 | 1,914.0 | +0.02 % (equal) | 1,916.8 | 1,917.8 | +0.04 % (equal) |
| decode@30000 | 70.1 | 70.0 | -0.25 % [-0.77, +0.28] | 70.1 | 70.2 | +0.12 % (equal) |
| decode-code (set) | 95.5 | 95.2 | -0.55 % [-1.56, +0.46] | 95.5 | 95.6 | +0.13 % [-0.59, +0.85] |
| decode-text (set) | 68.7 | 68.4 | -0.62 % [-1.72, +0.48] | 68.7 | 68.7 | +0.11 % [-0.67, +0.89] |

Rates are tok/s: prefill after the stated prompt, decode after it. Per round (B against A, decode-code / decode-text) r1: -1.47 / -1.61, -0.42 / -0.73, -0.10 / 0.00, -0.21 / -0.15 %; r2: +0.42 / +0.59, 0.00 / 0.00, +0.10 / 0.00, 0.00 / -0.15 %.

Against the expectation (decode -0.5 to +0.5 %, prefill -1 to +1 %):
- **D (margin 3072): inside on every metric**, no CI reaches -1 %; decode is +0.1 to +0.3 %, prefill equal. Moving 2.4 GB of pinned experts from GTT to registered userptr costs no measurable decode or prefill. (It does not test the eviction risk of userptr memory under pressure; that needs the balloon test, which was not run.)
- **C (chunked): prefill equal (+0.0 to +0.4 %); decode -0.25 to -0.62 % with every CI spanning zero**, and the two set metrics are flagged "outside" only because their CI reaches beyond -0.5 %. The first round is the outlier (-1.5 % on the set; its order AB also flags decode@30000 and decode@2000:write off plan); rounds 3 and 4 are within -0.2 %. Reading: no cost that these 4 rounds can separate from noise, upper bound of the loss about 1 to 1.5 % (the CI edge), and **not proven below 0.5 %**. The deterministic counters are equal (slots, VRAM, pinned), so the layout does not change what is resident.
- The known noise of `bench/noise.toml` is from the 9600X; baseline step 2 (the 9950X noise) is not done, so the "open" verdicts stay open.

### Decision (the rule of the plan)

- **D:** tokens identical, within 1 % on every metric: a larger GTT margin is affordable on the measured runs. What it buys (room for the compositor's fallback placement) is not measured here.
- **C:** tokens identical (goldens and SHA), prefill equal, decode within noise but with an upper bound near -1 %. By the plan's rule ("within 1 %, tokens identical: chunked becomes the default") it **qualifies on the point estimates and stays a flag until** (a) one more A/B with `--min-rounds 6` (or the 9950X noise of step 2) narrows the decode CI; the first round's -1.5 % should be explained or dropped by a repeat. This is a recommendation to the coordinator; no default was changed.

### Chunk activation check

Job `confirm.sh` (queue kind gpu, one `tona-gpu-gen` start with `TONA_SLOTS_CHUNKED=1 TONA_VRAM_LOG=1`, 2 tokens, same build): the log shows
`VRAM: the core (slots and prefill workspace, 8764 MB) is built from 262 chunks of 34 MB` (32 MiB rounded to the granule). **The flag is active and the chunked layout is what r1 and d1 measured.**
(Slot count in this start, 81, differs from the server's 70 only because this is the gpu-gen tool without the 140K-context KV and with another free-VRAM reading.)

## Addendum: run r3, a longer A/B of C (plan, written before the run, 2026-11-18)

Question: is the decode cost of chunked slot buffers (r1: -0.25 to -0.62 %, every CI spanning zero, round 1 an outlier at -1.5 %) below 0.5 %, or real?
Decision: CI inside -0.5..+0.5 % on decode-code and decode-text: chunked may become the default (follow-up with the coordinator); CI reaching below -0.5 %: stays a flag, the cost is quoted next to the benefit.

- Scope: the same binary `b164t`, A default, B `TONA_SLOTS_CHUNKED=1`, workloads `short,set` (decode only; prefill was equal in r1, so `mid` is dropped to save GPU time), `--min-rounds 6 --max-rounds 8`, alternating, stop rule of `ab.py`.
  Expected duration: about 2.5 min per round on these two workloads, 15 to 20 min in all. Queue kind exclusive; it waits for the desktop guard (monitors off) and requeues on exit 75.
- Expectation: decode within -0.5 .. +0.5 % (point estimate about -0.2 %, from rounds 3 and 4 of r1); source: r1 of this file.
- Active tooling: as r1 (`--load-log on`, no `TONA_DIAG`, no `TONA_VRAM_LOG`); test conditions `hw_report` and `quiet_check` before and after into `/var/tmp/claude/166/run-r3.tc`.
- A result outside the expectation is a finding; round 1 of r1 is not dropped when this is analysed, the first round of r3 is treated the same way as the others.
