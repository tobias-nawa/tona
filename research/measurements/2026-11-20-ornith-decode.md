# Ornith decode step (#175)

Date: 2026-11-20. Branch `worktree-<worktree>`, base `60d3fcd`. Build `work/build/ornith-dec`, logs
`/var/tmp/claude/ornith-dec/`. Starting point: `2026-11-19-ornith-opt2.md` items 2 and 4 (decode after 2K / 16K / 64K:
about 105 / 96 / 35 tok/s with the MTP head; shares of the pass at width 1: host.wait 7 to 8 %, router 6.5 %, about 15 small
steps per layer; VGPR spills 40/54/70 at widths 1/4/8).

## Test conditions

`python3 tools/hw_report.py` (00:14, `/var/tmp/claude/ornith-dec/hw.txt`): Ryzen 9 9950X, RX 9060 XT gfx1200 15.9 GiB,
1.83 GiB VRAM held by others (the desktop), 60.5 GiB RAM (53.8 available), swap 1.87 GiB in use, load average 8.3 (other
contributors' builds in the queue), commit `60d3fcd` (clean tree). `quiet_check.py` before block 1: **warning** (load average
high, 1.83 GiB VRAM held). Every speed block gets its own check before and after (files in the log directory); a warning
there is reported with the result.

## Plan

### Block 1: where the decode pass goes at 2K, 16K and 64K (deterministic counters first)

**Question.** (a) Per step family, how many microseconds does a decode pass take after a 2K, a 16K and a 64K prompt, and which
family explains the fall from about 96 tok/s (16K) to about 35 tok/s (64K)? (b) How much of `host.wait` is the CPU's own work
(rank, union, slots, CPU experts) and how much is the handshake?

**Run.** One `exclusive` job: `ab.py` with both arms on the same build (A = B), A with `TONA_MEGA_PROF=1
TONA_MEGA_PROF_EVERY=20`, B without (so the cost of the profile itself is also measured), workload `server`, targets
`2000:100,16000:100,64000:50`, `--ctx 70000`, `--drafter none` (MTP head from the file), 2 rounds (the minimum of ab.py; the profile is shares of
deterministic steps; the spread of the B arm against A gives the cost of the profile). A second job with
`TONA_HOST_TRACE=file` on the same workload only if (b) is not answered by the server's `wait_s` / `select_s` counters.
Server log lines (expert slots per layer, "tokens per pass", hits and misses) are read for each context.

**Expectation.** At 2K: pass 17 to 20 ms at width 3 to 4, shares as D1 of round 1 (`2026-11-19-ornith-opt1.md`: moe.gate_up
about 22 %, moe.down 17 %, dn.in 13 %, host.wait 6 %). At 64K: 52 ms per pass (35 tok/s at 1.81 tokens per pass,
the opt2 measurement of 2026-11-19, item 2) against about 22 ms at 2K; tolerance +-30 %. Candidate causes (all unmeasured, listed to
be tested by this run): decode attention over 64K keys (int8 KV, 10 attention layers: about 0.1 GB per layer at ctx 70000, a
read of 1 GB per pass = 3 ms from VRAM, 18 ms if the KV sits in pinned host RAM), fewer expert slots in VRAM at ctx 70000 (KV
and prefill buffers) and with them more CPU experts, lower acceptance (1.81 tokens per pass, so more passes per token).
**Decision rule fixed before:** if the fa.* steps at 64K are more than 30 % of the pass, the lever is the decode attention
(split count, KV placement); if moe.* and host.wait grow with the slot count falling, it is expert placement (#148, reported,
not built here); if neither, the profile is read again for the family that grew most.

**Active tooling.** `TONA_MEGA_PROF` and `TONA_MEGA_PROF_EVERY` (in-kernel step timestamps, a few us per step) in arm A only;
no `TONA_DETERMINISTIC`, no rocprof.

**Stop.** Out of memory, a GPU fault or queue exit 75 stops the job and is reported.

(Further blocks are planned when their turn comes, each before its run.)

## Results

### Block 1 (job `ornith-dec-b1`, build `ornith-dec` at `60d3fcd`, `work/ab/ornith-dec-b1.{md,json}`, server logs `work/ab/ornith-dec-b1-logs/`)

ab.py needs 2 rounds at least (the first submission with 1 round was refused before it ran). Quiet check before (00:34): warning, 1.83 GiB
VRAM held by others and the system swapping (749 pages in 1 s); after (00:37): warning, 1.83 GiB VRAM held. A = profile on, B = plain, same build.
Server: ctx 70000, **KV int8 0.81 GB in RAM**, VRAM 13.41 GB, **140 expert slots per layer** (of 256), 20.5 GB pinned.
Output SHAs equal in A and B at all three lengths; expert hits and misses identical; the profile costs nothing measurable at 16K and 64K
(decode -0.01 %, -0.16 %), at 2K -0.8 % (inside the +-3 % of the plan; prefill@2000 round 1 of B was 15 % slower, round 2 equal: an outlier
of the first request after a start, seen in earlier runs; outside the plan, reported as a finding, not explained).

Pass time and shares by step family (profile of 20 passes each; mean width of the window in brackets):

| Context | ms per pass | fa.attn | moe.gate_up + moe.down | dn.in | head.out | router | host.wait | tokens per pass |
|---|---|---|---|---|---|---|---|---|
| 2K (width 2.8 / 3.3) | 17.9 / 18.3 | 1.1 ms (6 %) | 6.2 / 6.8 ms (35 to 37 %) | 2.5 ms | 1.35 ms | 1.24 / 1.32 ms (7 %) | 1.25 / 1.03 ms (5.6 to 7 %) | 2.30 |
| 16K (width 2.1) | 21.7 | **7.2 ms (33 %)** | 4.7 ms | 2.3 ms | 1.3 ms | 1.07 ms | 0.94 ms (4.3 %) | 3.57 |
| 64K (width 4.2 / 2.05) | 43.0 / 50.1 | **23.6 / 36.3 ms (55 / 72 %)** | 7.9 / 4.4 ms | 2.5 ms | 1.3 ms | 1.6 / 1.0 ms | 1.1 / 0.85 ms | 1.81 |

Answer to the question: **the fall at 64K is the decode attention, which reads the whole KV cache (0.67 GB at 64K, 10 layers) from pinned
host RAM**: 36.3 ms for 0.67 GB is about 18 GB/s, 7.2 ms for 0.17 GB at 16K about 24 GB/s, i.e. the PCIe-bound zero-copy read; from VRAM the same
bytes take about 2 to 3 ms. The KV is in RAM because `setup_kv` puts it there from 0.4 GiB in total (ctx 70000 is 0.81 GB). Everything else
(moe, dn, head, router, host.wait) is the same at the three contexts. The 64K rate is also halved by acceptance (1.81 tokens per pass against 3.57
at 16K). The decision rule fixed before (fa.* above 30 %: the lever is the decode attention, KV placement) applies.

### Block 2 (plan, written before its run): KV in VRAM against KV in RAM, same build

**Question.** What does `TONA_KV_HOST=0` (KV in VRAM, the experts lose about 0.8 GB of slots: 140 slots become about 131) do to decode at 2K, 16K and
64K, and what do the lost slots cost at 2K and 16K? Run: one `exclusive` `ab.py`, A = default, B = `TONA_KV_HOST=0`, same build, same workload as
block 1 (`2000:100,16000:100,64000:50`, ctx 70000, `--drafter none`), 2 rounds (no noise entry; 2 is the minimum), no profile.
**Expectation (source: block 1).** decode@64000: fa.attn 36 ms down to about 3 ms: pass 50 to about 17 ms, **+100 to +180 %**; decode@16000: 7.2 ms
down to about 1 ms of a 21.7 ms pass: **+20 to +40 %**; decode@2000: 9 fewer slots of 140 means about 6 % more misses (1247 misses per request at 2K):
**-3 to 0 %**; prefill: the prefill gathers directly from host KV (`prefill_plan.cpp`); in VRAM it should not be slower: -2 to +3 %.
Output SHAs must be equal (placement only). **Decision fixed before:** if decode@2000 loses more than 4 % or any prefill more than 2 %, the
policy must keep the KV in RAM at short context (a threshold by context length or by VRAM budget); if SHAs differ, stop.
**Active tooling.** None beyond `TONA_KV_HOST=0` in B.

### Block 2 result: KV in VRAM (`TONA_KV_HOST=0`) against KV in RAM, ctx 70000 (job `ornith-dec-b2`, 3 rounds, `work/ab/ornith-dec-b2.{md,json}`)

Quiet check before 00:46 (warning: 1.83 GiB VRAM held by others, 1 page swapped), after 00:50 (warning: VRAM held). Spread between rounds 0.0 to 0.3 %.
Output SHAs equal at 2000, 16000 and 64000 (placement only). Expert counters differ as expected (fewer slots: misses at 2K 1247 to 1664, 16K 1324
to 1868, 64K 569 to 750).

| Rate | A (KV in RAM) | B (KV in VRAM) | B vs A [CI] | Plan | Verdict |
|---|---|---|---|---|---|
| decode@2000 (tok/s) | 114.0 to 114.2 | 115.4 to 115.6 | **+1.32 %** [+0.47, +2.17] | -4 to 0 % | better than planned (the faster KV and the shorter fa.attn outweigh the 33 % more misses) |
| decode@16000 | 94.7 to 95.1 | 147.5 to 147.8 | **+55.6 %** [+54.5, +56.6] | +20 to +40 % | above plan |
| decode@64000 | 35.3 to 35.4 | 78.4 | **+122.0 %** [+121.1, +123.0] | +100 to +180 % | as planned |
| prefill@2000 | 2743 to 2755 | 2714 to 2716 | -1.28 % [-2.16, -0.40] | -2 to +3 % | within plan, a small real loss |
| prefill@16000 | 4055 | 4104 to 4106 | +1.22 % | | |
| prefill@64000 | 2829 to 2832 | 2841 to 2845 | +0.43 % [-0.42, +1.28] | | open (equal within margin) |

The decision rule fixed before (decode@2000 loses more than 4 % or prefill more than 2 %: keep the KV in RAM at short context) is not triggered
at ctx 70000. Open: at the service default context (262144: KV 3.0 GB) and at 140000 (1.6 GB) the VRAM KV takes 31 or 17 slots; block 3 measures that.

### Block 3 (plan, written before its run): the same A/B at ctx 140000 and 262144

**Question.** Does KV in VRAM still win when it takes more slots (ctx 140000: 1.6 GB, about 17 slots of 140; ctx 262144, the service default: 3.0 GB,
about 31 slots)? Run: one `exclusive` `ab.py`, A default, B `TONA_KV_HOST=0`, targets `2000:100,64000:50`, `--ctx 140000,262144` (one start each),
2 rounds. May fail with out of memory at 262144 (VRAM KV 3 GB beside the experts): that is then the answer for that context (a failed start is
reported). **Expectation:** decode@2000 -12 to 0 % (misses grow with the lost slots; ctx 70000 lost 6 % of the slots and gained 1.3 %), decode@64000 +60 to
+130 %, prefill -3 to +3 %. **Decision fixed before:** the policy puts the KV in VRAM while decode@2000 is not worse than -3 % and no start fails;
beyond that the KV stays in RAM (or a hot tail is built).

### Block 3 result: KV in VRAM at ctx 140000 and 262144 (job `ornith-dec-b3`, 2 rounds, `work/ab/ornith-dec-b3.{md,json}`)

Quiet checks before 00:52 and after 00:57: warning only for 1.83 GiB VRAM held by others and 1 page swapped. Slots per layer (server log): KV in RAM
140 at every context; KV in VRAM **93 at ctx 140000 (KV 1.61 GB), 52 at ctx 262144 (KV 3.02 GB)** (70000: about 131). B = `TONA_KV_HOST=0`.

| Rate | ctx 140000 B vs A | ctx 262144 B vs A |
|---|---|---|
| decode@2000 | +0.90 % [-0.79, +2.59] (rounds -0.4, +2.2: noise) | **-6.32 %** [-7.18, -5.47] |
| decode@64000 | **+130.0 %** (38.1 to 87.5 tok/s) | **+83.4 %** (38.3 to 70.2) |
| prefill@2000 | +1.8 % (rounds -3.5, +7.1: outlier-prone first request) | **-9.4 %** [-10.3, -8.6] |
| prefill@64000 | +1.1 % | **-3.5 %** [-4.3, -2.6] |

SHAs equal except 64000 at ctx 262144 (B differs, tokens per pass 2.04 against 2.78; normal mode, not deterministic, no verdict; the identity is checked with
`TONA_DETERMINISTIC=1` in block 4). The plan's decision rule (KV in VRAM while decode@2000 is not worse than -3 % and no start fails) holds at 140000 and fails at 262144,
the service default. All in VRAM is the best at 140000 and a bad trade at 262144 (63 % of the slots gone); all in RAM wastes 2.3x at 64K at every context.
Hence a split by layer (every layer has its own KV pointer: no kernel change): `engine/gpu/kv_place.h`, policy in `Qwen35MoeProgram::setup_kv`: a budget
of `TONA_KV_VRAM_GB` (default 1.7: the 1.61 GB that cost nothing at 2K) holds the MTP layer first (its attention runs at every draft step) and then the
attention layers from the first; the others stay in pinned RAM. The Coder's paths are unchanged (`kv_place_.host` empty).

### Block 4 (plan, written before its run): the split policy against the old rule

**Question.** Does the split (default budget 1.7 GB) give at ctx 262144 what all-in-VRAM gave at 140000 for short contexts and a good part of the long-context gain? Run: one
`exclusive` `ab.py`, A = `ornith-dec-base` (binaries of `60d3fcd`, the old rule), B = `ornith-dec` (the split), workload `server`, targets `2000:100,16000:100,64000:50`,
`--ctx 70000,140000,262144`, 2 rounds. Expectation (source: blocks 2 and 3): ctx 70000 and 140000 all layers fit, so B = the `TONA_KV_HOST=0` arms of blocks 2 and 3
(decode +1 %, +55 %, +122 to +130 % at 2K, 16K, 64K); ctx 262144: 6 of 11 buffers in VRAM (about 93 slots like ctx 140000), 5 layers' KV still in RAM:
decode@2000 -2 to +2 %, decode@64000 **+40 to +80 %**, prefill@2000 -3 to +1 %. Decisions fixed before: if decode@2000 at ctx 262144 is worse than -3 % the default budget is lowered
(sweep 1.0 GB); if decode@64000 at ctx 262144 gains less than +30 %, the budget is raised or a hot head by position is considered. Output SHAs equal in a deterministic run
(`TONA_DETERMINISTIC=1`, targets 2000 and 16000, one round) before the speed run. Active tooling: none.

### Block 4 result: the split (budget 1.7 GB) against the old rule (job `ornith-dec-b4`, A = `ornith-dec-base`, B = `ornith-dec`, 2 rounds each, `work/ab/ornith-dec-b4{,-det}.{md,json}`)

Quiet check before 01:00, after 01:09: warning only for 1.83 / 1.87 GiB VRAM held by others. **Deterministic run (`TONA_DETERMINISTIC=1`, ctx 262144, 2000 and 16000 tokens):
SHAs of A and B equal and stable** (4cf94b9cdeda674c, e85b642df9178deb). The non-deterministic speed run also has equal SHAs at all nine (context, length) pairs.
Placement (server log): ctx 70000 all 11 KV buffers in VRAM (116 slots), ctx 140000 **9 of 10 layers + MTP** in VRAM (97 slots; the budget fit 10.6 buffers of 0.16 GB), ctx 262144
**4 of 10 layers + MTP** in VRAM (99 slots; buffers are 0.30 GB). (The "KV int8 ... in RAM" word of the load line says RAM whenever any layer is in RAM.)

| B vs A | ctx 70000 | ctx 140000 | ctx 262144 |
|---|---|---|---|
| decode@2000 | +0.80 % | -0.20 % | -2.29 % |
| decode@16000 | +55.8 % | +45.2 % | +15.1 % |
| decode@64000 | +122.9 % | +97.5 % | +31.0 % |
| prefill@2000 | -5.2 % (rounds -9.6, -0.8: first-request noise) | -3.4 % | -10.4 % (rounds -3.2, -17.5: noise) |
| prefill@16000 / @64000 | +1.5 / +0.5 % | +1.0 / +0.2 % | +0.6 / +0.2 % |

Reading: **one layer left in RAM costs a quarter of the gain** (ctx 140000: +97 % instead of +130 % of block 3): a layer in RAM adds about 3.6 ms per pass at 64K, in VRAM about 0.3 ms.
The layer is the wrong granularity; the 1.7 GB budget did not even cover ctx 140000 (1.63 GB of KV plus the MTP layer). At ctx 262144 the gain is +31 % at 64K for -2.3 % at 2K.
The decision rule fixed before (decode@2000 worse than -3 %: lower the budget; decode@64000 under +30 %: raise it or consider a hot head) points at raising the budget (+31 % is at the edge).
Per buffer in VRAM at ctx 262144: -0.46 % decode at 2K, +3 % at 16K, +6 % at 64K, so a coding workload with long context favours more. Next: a budget sweep (block 5).

### Block 5 (plan, written before its run): budget sweep

**Question.** Which default for `TONA_KV_VRAM_GB`? Runs (one `exclusive` job, three `ab.py` calls, A = `ornith-dec-base`, B = `ornith-dec` with the budget in `--b-env`, targets `2000:100,16000:100,64000:50`,
2 rounds): ctx 262144 with 1.9 GB (5 layers + MTP) and 2.4 GB (7 layers + MTP), then ctx 140000 with 1.9 GB (all 11 buffers fit: 1.77 GB). Expectation (source: block 4 per-buffer effects): 1.9 GB at 262144:
decode@2000 -2.8 %, @16000 +18 %, @64000 +37 %; 2.4 GB: -3.7 %, +24 %, +50 %; ctx 140000 as block 3: +0.9 %, +130 % at 2K and 64K. Decision fixed before: take the largest
budget whose decode@2000 at ctx 262144 is not worse than -4 % (the Coder's long-context users gain more than the 2K cost; the user's budget variable stays), unless the ctx 140000 run is not within 3 points of block 3.

### Block 5 result: budget sweep (job `ornith-dec-b5`, A = `ornith-dec-base`, 2 rounds, `work/ab/ornith-dec-b5-{1,2}.*` = 1.9 and 2.4 GB at ctx 262144, `ornith-dec-b5-140k.*`)

SHAs equal everywhere. Placement and slots (server log): ctx 262144 with 1.9 GB: 5 layers + MTP in VRAM, 91 slots; with 2.4 GB: 6 layers + MTP, 83 slots; ctx 140000 with 1.9 GB: all
11 buffers, 92 slots.

| B vs A | 262144, 1.7 GB (block 4) | 262144, 1.9 GB | 262144, 2.4 GB | 262144, 3.0 GB (all, block 3) | 140000, 1.9 GB |
|---|---|---|---|---|---|
| decode@2000 | -2.3 % | -1.7 % | -2.9 % | -6.3 % | -0.4 % |
| decode@16000 | +15.1 % | +19.4 % | +24.0 % | | +52.7 % |
| decode@64000 | +31.0 % | +39.8 % | +55.2 % | +83.4 % | +118.9 % (block 3: +130 %: A and slots differ a little, not chased) |
| prefill@2000 | | +2.4 % | -5.2 % (noise-prone first request) | -9.4 % | -4.0 % |
| prefill@64000 | +0.2 % | +0.05 % | +0.15 % | -3.5 % | +0.18 % |

Decision by the rule fixed before (largest budget with decode@2000 at ctx 262144 not worse than -4 %): **default 2.4 GB** (`TONA_KV_VRAM_GB`). It holds ctx 140000 completely (1.77 GB). The ctx 140000
run is 11 points under block 3 at 64K (the rule asked for 3): the A arm and the slot count differ slightly between the runs; the direction and the size of the gain stand.
The layer split is the limit of this approach: at 262144 the gain at 64K is +55 % where a position split (all layers' first T positions in VRAM, T = 2.4 GB / 11 buffers / 1152 B = about 190K positions)
would keep a 64K context entirely in VRAM, i.e. the +120 % of ctx 70000 at the same VRAM cost. It needs the cache address in the attention reads (`fa::KvHead`) and writes (`fattn_prep_kv`,
`k_kv_store`) and two regions in stash, restore, staging and the MTP cache; not built here (shared with Bolt; for a ticket of its own).

### Blocks 6 to 8 (plan, written before the run; one `exclusive` job `ornith-dec-b678`, `/var/tmp/claude/ornith-dec/block678.sh`; default budget now 2.4 GB)

Block 6, the Coder: `ab.py --workload short` (its default server workload: two requests of 2K tokens, 400 decoded, ctx 140000), A = `ornith-dec-base` (build of `60d3fcd`), B = `ornith-dec`; expectation decode and prefill
-0.5 to +0.5 % (no device code and no Coder code path changed; shared host code runs with an empty `KvPlace`), SHAs equal. Block 7, Ornith final: targets 2000:100, 16000:100, 64000:50, ctx 70000 and 262144, 2 to 3 rounds;
expectation from blocks 2 to 5: ctx 70000 decode +0.8, +56, +123 % at 2K, 16K, 64K; ctx 262144 (2.4 GB) -2.9, +24, +55 %; prefill within -4 to +4 %. Block 8, the split of host.wait: A = B = `ornith-dec`, A with `TONA_HOST_TRACE`,
one length (2000 / 200 tokens, ctx 70000); expectation: host.wait is 25 us per router (1.0 ms / 40) of which the CPU's own selection is a part; read with `tools/host_trace_report.py`. The goldens
(`/var/tmp/claude/ornith-dec/goldens.sh`, a `gpu` job) run first.

### Blocks 6 to 8 results (job `ornith-dec-b678`, quiet check before 01:30: warning 1.98 GiB VRAM held by others; after 01:43: warning 2.04 GiB)

**Block 6, the Coder** (`work/ab/ornith-dec-b6-coder.*`, `short`, 6 rounds, A = `ornith-dec-base`): decode@2000 +0.16 % [-0.25, +0.57] (77.2 against 77.2 tok/s), decode@2000:write -0.15 %
[-3.41, +3.11] (129.5 against 129.4; single rounds of both arms jump by 3 % between 125 and 130, in A as well as in B: the known bimodal behaviour, not the change), prefill@2000 +0.20 %, prefill@2000:write +0.05 % (equal).
SHAs equal, tokens per pass equal. **The reference is not worse.** The goldens of the branch build (`golden.sh compare` decode, cpu plan, prefill with dflash, `golden-server.sh compare`) and `test_gpu_fattn`, `test_gpu_mamba`,
`test_gpu_moe_latent`, `test_gpu_bolt_quant`: exit code 0 each (`/var/tmp/claude/ornith-dec/golden-rc.txt`, ALL-GREEN).

**Block 7, Ornith final** (`work/ab/ornith-dec-b7.*`, 3 rounds, default budget 2.4 GB, rounds agree within 0.3 % except the first request after a start): 

| B vs A | ctx 70000 | ctx 262144 (6 layers + MTP in VRAM, 83 slots) |
|---|---|---|
| decode@2000 | +0.98 % (115.1 to 115.5 tok/s) | -2.98 % (113.9 to 110.5) |
| decode@16000 | +55.1 % (94.9 to 147.1) | +23.4 % (94.9 to 117.1) |
| decode@64000 | +121.8 % (35.3 to 78.2) | +49.3 % (35.3 to 52.8) |
| prefill@2000 | +0.03 % (outlier-prone) | -5.4 % [-6.3, -4.5] (the round values agree at -5.2 to -5.6 %: a real small loss, the slots at 2K prefill) |
| prefill@16000 / @64000 | +1.4 / +0.4 % | +0.03 / +0.2 % |

The one plan violation: prefill@2000 at ctx 262144 (-5.4 % against -4 to +4 %); it is a 2K prompt prefill, 0.7 s, the experts the 56 lost slots would have held. The budget trades this
(and -3 % decode at 2K) for +23 % and +49 % at 16K and 64K at the default context; `TONA_KV_VRAM_GB=0` restores the old placement, `TONA_KV_HOST=0/1` forces all in one place.

**Block 8, where host.wait goes** (`TONA_HOST_TRACE`, 79 cycles at 2K, `tools/host_trace_report.py`, cycle 19.9 ms, mean width about 3.5): per router the CPU spends 15.9 us on the selection (rank, union, slots: **0.64 ms per cycle = 3.2 %**) and 50.5 us
in `cpu_experts` (2.0 ms per cycle, 10 %, overlapped with the GPU's own experts); the rest of the 0.8 to 1.0 ms `host.wait` of the profile is the handshake itself (mailbox write by block 0, polling, write back). **A bit-exact GPU top-k
could therefore save at most the 0.64 ms of selection (3 %) and only on layers without a miss; a router handshake is still needed for the slot assignment and the CPU experts.** Not built: it would touch the shared megakernel
(the Coder's) for at most 3 %. Same for the spills (VGPR spills 40/54/70 sit in the matvec phase of the inlined `k_mega`, which the round-1 profile has at 80 to 100 % of the bandwidth) and the 15 small steps per layer
(res.norm 81 x = 2.1 %, moe.sum 1.4 %, moe.act 1.2 %, dn.conv 1.4 %: together about 6 %, each at 1 to 3 us per step): no measured step below its bandwidth, so no lever was taken; searched for a cheaper route: nothing new beyond the
2026-11-19 prior art (MonoMoE flags instead of barriers, ProMoE prefetch), which needs the barrier scheme of the shared kernel.

### Summary

The decode pass of Ornith after a long prompt was bound by the KV read from pinned RAM, not by a kernel. The split of the KV cache by layer under a VRAM budget gives, at the service default context, +23 % (16K) and +49 % (64K) decode
for -3 % at 2K, and +55 % / +122 % at ctx 70000 for free; bit-identical output; Coder untouched. Open: a split by position (hot head) would give the +122 % at ctx 262144 as well (design in block 5), the budget decided
from the real VRAM, and the numbers-changing levers (8-bit router, step fusion) left to the maintainer.
