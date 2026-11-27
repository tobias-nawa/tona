# Mac: the backend's own buffers in a residency set (`TONA_METAL_PIN_OWN`)

Issue #108. Follows [2026-11-15-mac-cold-submit.log](2026-11-15-mac-cold-submit.log) and
[2026-11-15-mac-early-commit.md](2026-11-15-mac-early-commit.md#where-the-stall-sits-the-first-decode-pass-after-a-long-prefill):
after a long prefill the first decode pass waits 0.1 to 0.25 s for the GPU to start, after an agent's tool step 17 to
40 ms; with every buffer of `Device::floats` in a residency set on the queue no stall over 3 ms is left.

## Plan (written before the run)

**Prior art.** Apple's residency sets (macOS 15) make allocations resident for the queues they are added to, instead
of per command buffer; llama.cpp's Metal backend keeps its buffers in a residency set and holds it for 3 min after
the last use (`ggml-metal-device.m`, quoted in [2026-11-14-mac-pin.md](2026-11-14-mac-pin.md)); the reference PC has no
such paging of device memory (VRAM), so nothing transfers.
**Run.** `scripts/ab.py --workload set,server --targets 8000:100 --ctx 32768`, DM-T + MTP-164, 6 fixed rounds (about
80 min), `--load-log on`, both arms `TONA_GPU_GAPS=500`, same build (132dfa4), B `TONA_METAL_PIN_OWN=1`.
**Expectation.** decode@8000 +2 to +6 % if the server counts the first decode pass in the decode rate (0.2 s of a
4.0 s decode), else within ±1 % and prefill@8000 +0.3 to +0.6 % (0.2 s of 47 s); decode code and text 0 to +0.5 %
(17 ms per request of about 10 s); answers identical; the probe's longest split under 1 ms in B, 150 to 250 ms in A.
Memory: the load log's compressed and wired sums within their spread.
**Active tooling.** `TONA_GPU_GAPS=500` both arms, the load log, defaults otherwise (early commit k = 2, GPU pick,
read-ahead level 2).

## Result

[2026-11-15-mac-pin-own-server-ab.md](2026-11-15-mac-pin-own-server-ab.md) (6 rounds, 78.5 min; conditions and load
per start there; test conditions of the probe runs in `2026-11-15-mac-pin-own.log` (this log was not kept)).

| Metric | B vs A, mean [CI] | Expected | Verdict |
|---|---|---|---|
| prefill@8000 | -4.09 % [-6.78, -1.41] | -1 to +1 % | **B slower**, decided after round 4; outside the expectation |
| decode@8000 | +1.09 % [-0.38, +2.55] | +2 to +6 % | open; below the expectation |
| decode-code | +0.26 % [-1.00, +1.52] | 0 to +0.5 % | open |
| decode-text | +0.37 % [-0.96, +1.70] | 0 to +0.5 % | open |

Answers identical (SHA 727ff658ae742fa7 in both arms). The stall is gone as planned: longest split 183 to 317 ms in
every A server, 0.1 ms in every B server; longest submit of a first command buffer after a pause 40 to 59 ms in A,
1.2 to 1.4 ms in B. Starts 8 to 14 (rounds 2 to 4) ran under heavy system load (memory level 4 to 5, a backup); the
prefill loss holds in the quiet rounds 5 and 6 too (-5.7 %, -6.4 %). The load log's compressed sum per start is lower
with B (4.2 to 7.5 GB in the quiet starts against 12 to 21 GB in A).

**Finding.** The stall of 0.2 s per long prompt goes, but the 8000-token prefill (about 48 s) gets about 2 s slower,
and the agent's short prefill after a cache hit would lose more than the 17 to 40 ms it gains. Not a default; the
switch stays a probe. Not explained yet: all buffers of `Device::floats` are created at start (no residency commit
during a run), and the wired sum at rest grows only 0.17 GB. Candidates for a later probe: the residency set changes
how the GPU maps the prefill scratch (per-token buffers of `Pass`), or the memory it pins makes the system evict
mapped weight pages that prefill reads; a split by buffer group (KV cache and state only, scratch not) would tell.
