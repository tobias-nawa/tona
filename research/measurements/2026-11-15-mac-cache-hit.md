# Mac: the 1.4 s of an agent's tool step after a cache hit

Issue #108. An agent's tool step reuses about 7900 cached tokens and computes about 36 new ones; that took 1.3 to
1.4 s on the M1 Max (DM-T + MTP-164, `--ctx 32768`), of which 35 tokens go through the prefill kernels.

## Plan (written before the keepalive runs)

**What is known from the probe runs** (log of each run below, scratch harness: one server, a request of about 7700
tokens, then tool steps of 36 new tokens with IDLE seconds between them; `TONA_METAL_CB_LOG=1` extended locally by
GPU time per kind of command buffer and the host times before the first one):

| Part of the 35-token prefill at 7900 tokens of context | ms | same chunk in `tona-bench` at context 35 |
|---|---|---|
| first command buffer: committed until the GPU starts it | 330 to 380 (IDLE 3 s), 0.5 (IDLE 0) | ~0 |
| experts | 420 to 440 | 300 |
| attention | 215 | 29 |
| layer starts (mixer, projections) | 250 | 240 |
| MTP entries | 27 | (26 with `--mtp`) |
| host before the first commit | 1 to 8 | |

The wait depends on the pause before the step: none at 0.5 s, once in three steps at 1 s (234 ms), always from 2 s
(307 to 407 ms). Not the cause: the residency sets (`TONA_METAL_PIN_OWN=1`, `TONA_METAL_PIN=rset` with 21.49 GB
pinned, both: 350 to 640 ms), the memory compressor (15 to 19 MiB decompressed per step), host work, a copy left in
the queue (every copy is waited on), the GPU's own idle (a 16-byte fill every 250 ms on a queue of its own: 374 to
442 ms). A keepalive that also reads 16 bytes of each weight buffer (`TONA_METAL_KEEPALIVE_WEIGHTS=1`) leaves 10 to
14 ms: the step takes 1.02 to 1.12 s instead of 1.35 to 1.45 s, and the host's encoding of the chunk drops from 720
to 840 ms back to about 400 ms.

**The decision these runs are for:** a keepalive inside a window after each request (default candidate), and its
interval. Possible results:
- (a) 500 ms keeps the wait under 20 ms and the idle power and wired memory with the keepalive stay within the spread
  of the arm without: the keepalive becomes a default candidate (A/B with ab.py next); the maintainer decides on the energy.
- (b) 500 ms does not hold it but 250 ms does: same as (a) at 250 ms, the power matters more.
- (c) the keepalive costs measurable power or wired memory: numbers to the maintainer, no default without approval.

**Runs.** (1) the agent scenario with `TONA_METAL_KEEPALIVE_MS=500 TONA_METAL_KEEPALIVE_WEIGHTS=1`, IDLE 3 s, 4 steps
(about 3 min). (2) Idle cost: per arm a server start, one short request, 5 s, then 60 s idle inside the window: the
system energy counter of the battery controller (`AccumulatedSystemEnergyConsumed` of `AppleSmartBattery`, readable
without root; unit not verified, compared between arms only) and vm_stat's wired pages; arms A (no keepalive), B
(250 ms with weights) in the order A B A B (about 8 min).
**Expectation.** (1) wait under 20 ms per step, steps 1.0 to 1.1 s. (2) Idle power B against A within ±0.5 W (the
GPU runs a 16-byte copy 4 times a second, microseconds each), source: none measured, a guess; wired memory B within
0.2 GB of A.
**Active tooling.** `TONA_METAL_CB_LOG=1` (run 1), the load log is not used (single starts, not ab.py).

## Result

Quiet check before: ok (10:24); after: one reading "GPU 99 % busy" right after the last server ended, ok 20 s later
(10:31). Conditions (`tools/hw_report.py`) in [2026-11-15-mac-cache-hit.log](2026-11-15-mac-cache-hit.log).

(1) Keepalive every 500 ms with the weight reads, IDLE 3 s:

| Step | committed until the GPU starts the first command buffer | host encoding | step |
|---|---|---|---|
| 1 | 9.4 ms | 416 ms | 1073 ms |
| 2 | 9.6 ms | 407 ms | 1039 ms |
| 3 | 9.2 ms | 411 ms | 1042 ms |

Without the keepalive (same harness, IDLE 3 s): 331 to 380 ms, 721 to 792 ms, 1351 to 1431 ms.

(2) Idle inside the window, 60 s per arm after one short request:

| Arm | energy counter, units/s | wired GB (at 5 s, at 35 s) |
|---|---|---|
| A, no keepalive | 1732 | 2.87, 2.87 |
| B, 250 ms with weight reads | 1752 | 2.87, 2.87 |
| A | 1698 | 2.86, 2.86 |
| B | 1688 | 2.86, 2.86 |

B against A: +0.3 % of the counter (spread between the two A arms 2 %), wired equal. The counter's unit is not
verified (an earlier 10 s sample gave 23288 units/s while a build ran); power in watts: unverified.

**Against the plan: result (a).** 500 ms keeps the wait under 10 ms (expected under 20), the step takes 1.04 to
1.07 s (expected 1.0 to 1.1); at 250 ms no measurable idle cost (within the spread of A, as expected). Next: the
switch as a default candidate (500 ms, window 180 s) through `scripts/ab.py`; the energy question goes to the maintainer.

**Not pursued: a warm-up command buffer when a request arrives** (before the tokenizer and the restore of the cached
state). Inside the keepalive's window the wait does not occur, so a warm-up helps only the first request after more
than 180 s without a pass, and there it could hide at most the host work before the first command buffer (about
140 ms of tokenizer and restore) of the 330 to 380 ms.

## Plan of the A/B (written before the run)

**Decision.** Whether `TONA_METAL_KEEPALIVE_MS=500` may become the default (with the maintainer's approval on the energy): only if
it loses nothing on the standard workload. Possible results: no metric decided slower → default candidate; a metric
decided slower → no default, the loss is a finding (the keepalive's command buffers compete with the passes).
**Run.** `scripts/ab.py --workload set,server --targets 8000:100 --ctx 32768`, DM-T + MTP-164, 6 fixed rounds (about
80 min, as the earlier A/Bs: effects of 0.2 to 2 % against a spread of about 1 %), `--load-log on`, same build
(4ad914d), B `TONA_METAL_KEEPALIVE_MS=500`.
**Expectation.** prefill@8000 0 to +1.5 % (the first chunk's wait of up to 0.6 s of 48 s, if the request follows a
pause over 1 s); decode@8000, decode-code, decode-text -0.5 to +0.5 % (the decode rate does not include the first
command buffer of a request; two keepalive buffers per second of microseconds each); answers identical.
**Active tooling.** The load log; defaults otherwise.

## Result of the A/B

[2026-11-15-mac-keepalive-server-ab.md](2026-11-15-mac-keepalive-server-ab.md) (6 rounds, 77.8 min; quiet before;
after: GPU busy at 60 s from the last server ending, then other programs 8 % of the CPU threads).

| Metric | B vs A, mean [CI] | Expected | Verdict |
|---|---|---|---|
| prefill@8000 | +2.05 % [+0.50, +3.60] | 0 to +1.5 % | **B faster**, decided after round 2; above the expectation |
| decode@8000 | -0.57 % [-1.47, +0.33] | -0.5 to +0.5 % | open |
| decode-code | -0.60 % [-1.96, +0.76] | -0.5 to +0.5 % | open |
| decode-text | -0.05 % [-2.12, +2.02] | -0.5 to +0.5 % | open |

Answers identical (SHA 727ff658ae742fa7). No metric decided slower: the decision's condition holds, and the maintainer
agreed to the default (500 ms, window 180 s).

**Finding.** Prefill gains more than the first chunk's wait explains (0.6 s of 48 s would be +1.3 %): the wait also
comes inside the prefill, after the pauses between ab.py's requests. **Defect of the A/B's build:** every pass woke
the keepalive's thread, which then sent at once, so during decode it sent once per pass instead of every 500 ms; the
open decode deltas of -0.6 % may be that. The default sends only after a pause of an interval without a pass (and is
not woken by passes); verified on the agent steps below, not by a second A/B.

The default (sends only after 500 ms without a pass), agent steps at IDLE 3 s: wait 13.1 / 19.0 / 11.7 ms, steps
1113 / 1055 / 1024 ms. Tokens identical to the references with it (`tona-check` 64 steps, `--widths 7`,
`--prefill 5`).

**Decode with the fixed keepalive: open.** No paired A/B of the fixed default against keepalive off. Across runs the
M1 A/B ([2026-11-15-mac-m1-registry.md](2026-11-15-mac-m1-registry.md), both arms with the default) gave decode@8000
25.2, decode-code 39.0 to 39.2, decode-text 27.0 to 27.4 tok/s, against 25.1, 38.7 and 27.1 for the arm without
keepalive above: no sign of a loss, not decided. The next Mac scale run covers it.
