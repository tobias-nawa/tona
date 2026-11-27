# KV cache: the hot head by position instead of the layer split (#175)

Date: 2026-11-20. Branch `worktree-<worktree>`, base `a69c6b0` (the per-layer split of
`2026-11-20-ornith-decode.md`). Builds: `work/build/kvh-base` (base sources, copy in `/var/tmp/claude/kvh-base-src`), `work/build/kvh`
(this branch). Logs in `/var/tmp/claude/kvh/`, ab.py output in `work/ab/kvh-*`.

## Prior art

- **vLLM tiered KV offloading** (blog 2026-09-10, docs `features/kv_offloading_usage`): the CPU tier is the only one with direct GPU
  access; blocks move whole, the unit is a block of a request, not a position range inside the attention kernel. Aimed at prefix reuse
  across requests (Red Hat H100 benchmark 2026-11-02: CPU offload 3.3x, with NVMe 5.8x when the working set exceeds RAM), not at the
  decode attention reading a live context. Status: shipped. Not a template: our kernel reads the cold part in place over PCIe.
- **llama.cpp** `-ngl` / `--no-kv-offload`: KV is placed by layer (the layers whose weights are offloaded have their KV on the GPU), or
  all of it on the host. That is the per-layer split this note replaces. No position split found in the issue tracker; not searched
  beyond the first page of results (**unverified** for the tracker).
- **FlexGen** (Sheng et al. 2023) and **InfiniGen** (Lee et al. 2024): FlexGen splits the KV cache between GPU and CPU by a percentage of
  the tokens/layers (an LP-chosen policy), InfiniGen keeps the whole cache on the CPU and prefetches only the speculated important
  entries. Both are known from the papers, **not re-read for this note** (the search tool returned nothing on them): the idea
  "first part resident, rest on the host" is FlexGen's; the reading of the cold part in place by the kernel is ours.
- "Where should the KV cache live?" (arXiv 2609.16215): hot blocks in the fastest tier, predictable accesses prefetched. Our access is
  fully predictable (every cell, every pass) and the hot part is chosen by position, the cheapest rule: no per-cell bookkeeping.

What we do that these do not: the cold part is read in place by the decode and prefill kernels (zero copy, PCIe), the split is a
single cell index per run, the write path and both kernels stay bit-identical to the flat cache.

## Design (what the code does)

`KvSplit` (`engine/gpu/kv_place.h`): the first `hot` cells of every attention layer are a VRAM buffer, the cells from `hot` on a pinned
buffer; the MTP layer's cache stays whole and comes first in the budget. `fa::KvHead` carries `cold` and `hot`; `kv_row_at` picks
the row (decode and prefill kernels). The decode write (`fattn_prep_kv`) takes the pinned part from `p[13]` / `i[7]`, the decode
read from `p[3]` / `i[5]`. The prefill writes (Ornith `k_qsa_prep`, Bolt `k_kv_store`) are two launches of the unchanged kernels
(`prefill_kv_split.h`): the tokens before the split with the VRAM base, the others with the pinned base minus `hot * row`; the Coder's
kernel is not touched. Prefill staging copies the two parts into one flat staging area when a group reaches past the split. Stash
and restore copy the VRAM part and the pinned part. Budget: `TONA_KV_VRAM_GB`, else 0.18 of (free VRAM per sysfs - `TONA_VRAM_RESERVE_MB`).

## Test conditions

`python3 tools/hw_report.py` (02:05, `/var/tmp/claude/kvh/hw.txt`): Ryzen 9 9950X, RX 9060 XT gfx1200 15.9 GiB, **2.15 GiB VRAM held
by the desktop**, 60.5 GiB RAM (51.8 available), swap 3.84 GiB in use, load average 1.2 (5 min: 6.5; other contributors' builds
in the queue). `quiet_check.py` before: warning, only for the VRAM held by others. Every block gets a check before and after
(`/var/tmp/claude/kvh/quiet-*.txt`).

## Results so far (no speed claim)

`test_gpu_fattn` (build `kvh`, 2 runs, exit 0, `/var/tmp/claude/kvh-fattn-1.log`): all earlier checks as before, plus the new
`split_all` (`engine/tests/fattn_split_check.h`): decode steps with the split at cell 37, 1000 (not a tile multiple) and 4093, passes
that cross it, hd 256 and hd 128: outputs and the written K/V rows **bit-equal** to the flat cache; prefill kernel on a split cache
**bit-equal**. `kv_place` unit test passes.

## Plan

### Block 1: identity of the output (deterministic), small budget so that every prompt crosses the split

**Question.** Does a cache split by position give the same tokens as the flat one when prompts and decode cross the split (prefill
writes, staging, decode writes, decode reads)? **Run.** Two `exclusive` `ab.py` jobs, `TONA_DETERMINISTIC=1` in both arms, A = `kvh-base`
with `TONA_KV_VRAM_GB=0` (old all-RAM placement), B = `kvh` with a budget that puts the split at about 17K cells (Ornith 0.5 GB) or
about 21K cells (Bolt 0.1 GB), ctx 262144, targets `2000:100,16000:100,64000:50` (2K inside the hot head, 16K around it, 64K far
beyond it), 2 rounds. **Expectation.** SHAs equal at all lengths; speed is not claimed here (decode at 64K between the all-RAM and
the all-VRAM value). **Decision fixed before:** one SHA differs = stop and find it; the placement must not change a bit.
**Active tooling:** `TONA_DETERMINISTIC=1`, log line "KV cache:" read for the split point.

### Block 2: Ornith speed at ctx 262144 and 70000

**Question.** Does the hot head give at ctx 262144 the decode of the all-in-VRAM placement at 16K and 64K, and are the 2K losses no
worse than the per-layer split's (decode -3 %, prefill -5.4 %)? **Run.** (a) A = `kvh-base` with `TONA_KV_VRAM_GB=0` (all in RAM),
B = `kvh` default budget; (b) A = `kvh-base` default (per-layer 2.4 GB), B = `kvh`; both `server`, targets `2000:100,16000:100,64000:50`,
ctx 262144, then (a) again at ctx 70000 as a control (everything fits: the new code must equal all-in-VRAM), 3 rounds each.
**Expectation (source: `2026-11-20-ornith-decode.md` blocks 2, 5, 7).** (a) decode +55 % (16K, +-10), +122 % (64K, +-20), -3 % (2K, +-2);
prefill@2K -5 % (+-3), @16K/@64K 0 (+-2). (b) vs the per-layer split: decode@16K +25 % (+-10) (117 -> about 147 tok/s), @64K +50 % (+-20)
(52.8 -> about 78), @2K -1..+2 %, prefill@2K -3..+3 %. Control ctx 70000 (a): as block 7 there (+1, +55, +122 %).
**Decisions fixed before:** decode@16K or @64K below half of the expected gain = look for the cause (cold reads in decode, staging
cost) before going on; decode@2K or prefill@2K worse than the per-layer split by more than 2 points = lower the share (0.18) or
the cap.

### Block 3: Bolt speed (ctx 262144), Coder guard

Planned after block 2 has results (same style: A = `kvh-base`, B = `kvh`, Bolt `server` 2K/16K/64K and `longdoc`; Coder `short`
default workload and one long point 2000:100,64000:50 at ctx 262144; goldens).

## Results

### Block 1 result (job `kvh-b1`, `work/ab/kvh-b1-{ornith,bolt}.*`, quiet check before and after: warning for the VRAM held by others only)

**Ornith**: split at 17152 cells (budget 0.5 GB, MTP layer in VRAM), A = all in RAM: SHAs **equal at 2000, 16000 and 64000** in the deterministic
run (4cf94b9cdeda674c, e85b642df9178deb, 4936cc76a20b07ef; the same hashes as block 4 of `2026-11-20-ornith-decode.md`), stable in both
arms, tokens per pass equal. The 16K and 64K prompts cross the split in prefill and in decode.
**Bolt**: split at 21504 cells (budget 0.1 GB): SHA equal at 2000, **different at 16000 and 64000** (stable per arm). Note that the 16K
prompt does not even reach the split. What differs between the arms besides the placement: the prefill plan. `pf_stage_on(end)` now
says "no staging" for a prompt that stays inside the hot head, the base stages every prompt (KV in RAM); the plan (chunk, group) changes, and
Bolt's chunked Mamba-2 prefill depends on the chunk (KL 0.0013 against sequential, `2026-11-19-bolt-opt2.md`); the expert hits and misses of
the two arms differ a lot at 16K (802 against 13765 misses), which fits a different plan. **Unverified until block 1b** (below).

### Block 1b (plan, written before its run): is the Bolt difference the plan or the placement?

**Question.** With the same prefill plan in both arms, are Bolt's SHAs equal? **Run.** One `exclusive` job, Bolt, deterministic, ctx 262144,
`ab.py` 2 rounds each: (i) target 16000:100 with A = `kvh-base` `TONA_KV_STAGE=0` (no staging, as B's plan at 16K), B = `kvh` budget 0.1 GB;
(ii) target 64000:50, A = `kvh-base`, B = `kvh` budget 0.1 GB with `TONA_PF_DEBUG=1` in both arms to read the plans. **Expectation.** (i) SHA
equal. (ii) the prompt reaches past the split, B stages like A; SHA equal if the plan is the same, read from the debug lines (group sizes,
layers borrowed). If (i) is equal and (ii) differs the cause is the plan of the loan (B's `pf_layered_need` counts the same stage bytes), read next.
**Decision fixed before:** (i) differs = a real placement bug in decode or prefill-without-staging, stop and bisect with `test_gpu_fattn` extended;
(i) equal = the difference is the plan; then the plan must be chosen the same way for equal contexts or the Bolt SHA comparison is made per plan.

### Block 1b result (job `kvh-b1b`, `work/ab/kvh-b1b-{16k,64k}.*`)

(ii) 64000 tokens, both arms stage (the plan is the same): Bolt SHA **equal** (e7ff5b566ec88464, stable), the prompt and the decode cross the split at 21504
cells. (i) 16000 with A without staging: **A is not stable between its own rounds** (5c55e0ca5fda4040 in one round, another in the other; expert hits 34027 / 22733
between rounds) and B's value (fb5a7dd79cb5dc3c) is not the one of block 1 (356900539592d273) either. So the 16K difference of block 1 is not caused by the
placement: Bolt's tokens at 16K depend on the plan and on the state of the expert cache (its decode with GPU or CPU experts and the chunked Mamba-2 prefill are not
bit-stable across plans even with `TONA_DETERMINISTIC=1`), the arms of block 1 had different plans. At 2K (one chunk, equal plans) and 64K (equal plans, split crossed) equal.
Known limit, not chased here: Bolt's determinism across prefill plans (a ticket of its own). The decision fixed before ((i) differs = bisect) is answered by the
instability of A itself: a placement bug would not make A unstable.

### Block 2 result: Ornith (job `kvh-b2`, `work/ab/kvh-b2{a,b,c}.*`, 3 rounds a and b, 2 rounds c, quiet check before and after: warning for the VRAM held by others only; desktop 2.15 GiB)

Default budget at ctx 262144: free VRAM 14.5 GB, budget 0.18 x (14.5 - 1.07) = 2.4 GB, MTP layer whole (0.30 GB), split at 2.1 GB / 10 layers / 1152 B = **182,000 cells**; slots per layer see the logs of `work/ab/kvh-b2a-logs`. **SHAs equal in all three runs at 2K, 16K and 64K** (4cf94b9cdeda674c, e85b642df9178deb, 4936cc76a20b07ef: the same as the deterministic hashes of block 1), tokens per pass equal.

| B vs A (tok/s, A to B) | (a) ctx 262144, A = all in RAM | (b) ctx 262144, A = per-layer split of main | (c) ctx 70000 control, A = all in RAM |
|---|---|---|---|
| decode@2000 | -2.85 % [-3.50, -2.21] (113.4 to 110.1) | +0.27 % [-0.34, +0.87] (109.9 to 110.2) | +0.80 % (114.2 to 115.1) |
| decode@16000 | **+47.0 %** [+46.4, +47.6] (94.7 to 139.3) | **+19.6 %** (116.5 to 139.2) | +54.9 % (94.7 to 146.7) |
| decode@64000 | **+118.8 %** [+111.8, +125.8] (34.9 to 75.1) | **+42.6 %** [+42.1, +43.2] (52.6 to 75.0) | +119.6 % (35.2 to 77.4) |
| prefill@2000 | -6.45 % [-7.14, -5.76] | -0.60 % [-1.20, -0.00] | -3.41 % [-5.6, -1.2] (outlier-prone first request) |
| prefill@16000 | -0.18 % | +0.14 % | +0.86 % |
| prefill@64000 | +0.37 % | -0.03 % (equal) | +0.33 % |

All results lie inside the plan except prefill@2000 of the control (c) (-3.4 % against -3..+3 %: two rounds of 2860 and 2755 tok/s in A, the known first-request noise;
B's two rounds agree at 2712 and 2710). Reading: against the all-in-RAM placement the hot head gives at ctx 262144 +47 % (16K) and +119 % (64K), the values of ctx 70000
within 4 and 3 percentage points (139.3 against 146.7 and 75.1 against 77.4 tok/s: the difference is the expert slots the 2.4 GB take). Against the per-layer
split of main (same VRAM budget, same cost at 2K) +19.6 % and +42.6 %, at an equal 2K cost (decode +0.27 %, prefill -0.6 %). The 2K losses against all-in-RAM (-2.9 % decode,
-6.5 % prefill) are those of the per-layer split (-3.0 % / -5.4 % in block 7 of the earlier note): not worse.

### Block 3 (plan, written before its run): Bolt speed and the Coder's upper bound

**Question.** (1) Bolt (full-attention KV 0.15 GB per layer at ctx 262144, 1.2 GB for all layers, with its KV in RAM from 0.4 GiB on): what does the VRAM KV
(the default budget holds all of it) and a half-size hot head (0.5 GB: about 108K cells; a 16K and a 64K prompt both inside it) do to decode and prefill? (2) Could the Coder's KV
(in RAM from 0.4 GiB; its attention reads 2048 selected cells plus the dense head) gain anything from VRAM at a long context? The Coder's
code path stays untouched; the upper bound is measured with `TONA_KV_HOST=0` (all in VRAM, flat) on the same build. **Runs** (one `exclusive` job, `ab.py`,
`server`, targets 2000:100,16000:100,64000:50 for Bolt; 2000:100,64000:50 for the Coder; ctx 262144, 3 rounds Bolt, 2 rounds Coder; Coder also `short` 3 rounds on the default workload):
Bolt A = `kvh-base`, B = `kvh` default; Bolt A = `kvh-base`, B = `kvh` with `TONA_KV_VRAM_GB=0.5`; Coder `short` A = `kvh-base`, B = `kvh`; Coder long A = `kvh`, B = `kvh` with `TONA_KV_HOST=0`.
**Expectation (source: the Bolt arms of block 1b: decode 41.3, 40.9, 32.0 tok/s at 2K, 16K, 64K with the KV in RAM, so the whole RAM penalty at 64K is at most 22 %, i.e. a gain of at most +28 %).**
Bolt B default: decode@2000 -3..+1 % (slots lost to 1.2 GB), @16000 +2..+10 %, @64000 +10..+30 %; prefill -3..+3 %; 0.5 GB arm: between. Coder `short`: decode and prefill -0.5..+0.5 %, SHAs equal.
Coder upper bound: decode@64000 -2..+4 % (selected cells only: 2048 x 576 B per layer per token; the dense part is the first 2048 cells). **Decisions fixed before:** a Bolt arm with decode@2000 worse than -3 %
and a 64K gain below 10 % = lower the Bolt share; the Coder upper bound above +5 % at 64K = propose a hot head behind its own flag, otherwise the Coder keeps its placement.
Output SHAs of Bolt are not compared here (plan-dependent, see block 1b). **Active tooling:** none beyond the environment variables named.

### Block 3 result (job `kvh-b3`, `work/ab/kvh-b3{a,b,c}.*`; quiet check files `quiet-*-b3.txt`, warning for the VRAM held by others only)

**Bolt**, ctx 262144, A = KV in RAM (`kvh-base`), 3 rounds, spread of the rounds 0.0 to 0.2 % except prefill@16000 in A (1574 to 1584):

| B vs A | (a) default budget: all 1.2 GB of KV in VRAM | (b) `TONA_KV_VRAM_GB=0.5`: split at about 108K cells |
|---|---|---|
| decode@2000 | -2.68 % [-3.26, -2.10] (41.3 to 40.2) | -0.75 % [-1.34, -0.16] (41.3 to 40.9) |
| decode@16000 | +2.24 % (37.6 to 38.4) | +4.47 % [+3.87, +5.08] (37.6 to 39.3) |
| decode@64000 | **+20.1 %** [+19.5, +20.6] (30.3 to 36.4) | **+22.1 %** [+21.5, +22.7] (30.3 to 37.0) |
| prefill@2000 | -1.88 % | -0.74 % |
| prefill@16000 | +0.16 % | +1.38 % |
| prefill@64000 | +7.79 % | +9.00 % |

Bolt SHAs equal in both runs at 2K, 16K and 64K (464308741868de30, 356900539592d273, e7ff5b566ec88464; non-deterministic mode, plans equal). Decision fixed before: not triggered
(decode@2000 -2.7 % in the worst arm, 64K gain 20 %). Reading: the gain at 64K (+20 to +22 %) is below the +28 % ceiling of the plan and the same for 1.2 GB and 0.5 GB; the
**smaller budget is better everywhere** (more slots: -0.75 % instead of -2.7 % at 2K). The 0.18 share that suits Ornith over-spends for Bolt, whose KV is 2.3 times smaller per cell
(576 B x 8 layers against 1152 B x 10) and whose pass is longer. Next: block 4 (budget per model).

**Coder** (`short`, 3 rounds, SHAs equal, `work/ab/kvh-b3c.*`): decode@2000 +0.59 % [-0.28, +1.45], prefill@2000 +1.38 % [-0.43, +3.19] (A's round 2 was 1415.7: outlier), :write decode -0.08 %,
prefill +0.53 %; expert counters identical; SHA 75eff9b8784c7d4c both. **The reference is not slower.** The Coder upper-bound run at ctx 262144 (`TONA_KV_HOST=0`) did not start:
"VRAM only fits 0 expert slots (in use 14.42 GB)": all KV (3 GB) in VRAM does not exist as an option at the Coder's default context. Re-planned at ctx 70000 in block 4.

### Block 4 (plan, written before its run): budget per model, the Coder's bound at a context where it starts

**Question.** (1) Ornith: is a smaller budget than 2.4 GB better (hot head 1.2 GB = about 78K cells, 1.8 GB = about 130K cells; both cover the 64K point)? (2) Bolt: 0.3 GB (65K cells,
just above the 64K point) against the default (all in VRAM)? (3) Coder: what does all KV in VRAM bring at ctx 70000 (KV 0.8 GB), the upper bound of a hot head? **Runs** (one `exclusive` job):
(1) A = `kvh` default (2.4 GB), B = `kvh` with 1.2 GB, then B with 1.8 GB; Ornith `server` 2000:100,16000:100,64000:50, ctx 262144, 3 rounds;
(2) Bolt A = `kvh` default (all), B = `kvh` with 0.3 GB, same targets, 3 rounds; (3) Coder A = `kvh`, B = `kvh` with `TONA_KV_HOST=0`, `server` 2000:100,64000:50, ctx 70000, 2 rounds.
**Expectation (source: blocks 2 and 3).** (1) 1.2 GB / 1.8 GB: decode@2000 +1..+2.5 % (about 20 and 10 fewer slots lost: the earlier sweep gave 0.5 point per 0.1 GB-step of 1.7 to 2.4 GB), decode@16000 and @64000 equal (-2..+2 %),
prefill@2000 0..+3 %. (2) Bolt 0.3 GB: decode@2000 +2..+3.5 % (0.5 GB: -0.75 %, all: -2.7 %), 16K +0..+3 % against all-in-VRAM... in numbers against A (all): decode@16000 +0..+3 %, @64000 -2..+1 %, prefill@64000 -3..+1 %.
(3) Coder bound: decode@64000 -2..+4 %, decode@2000 -3..+1 %, prefill -3..+2 %. **Decisions fixed before:** (1) a budget with decode@64000 within 2 % of 2.4 GB and a better 2K is taken as the share for Ornith
(the context beyond that point loses; the user variable stays); (2) the same for Bolt; (3) above +5 % at 64K: propose a Coder hot head behind its own flag, otherwise the Coder keeps its placement.
**Active tooling:** none beyond the variables named.

### Block 4 result (job `kvh-b4`, `work/ab/kvh-b4-{ornith-1.2,ornith-1.8,bolt,coder}.*`; the first three 3 rounds, the Coder 2; quiet check before and after, warning for the VRAM held by others only)

Ornith ctx 262144, A = default 2.4 GB (182K cells), SHAs equal everywhere (same three hashes as before):

| B vs A | 1.2 GB (78K cells) | 1.8 GB (130K cells) |
|---|---|---|
| decode@2000 | **+3.67 %** [+3.08, +4.26] (110.4 to 114.5) | **+2.15 %** [+1.53, +2.78] |
| decode@16000 | **+4.28 %** (139.8 to 145.9) | +3.16 % |
| decode@64000 | **+2.72 %** (74.9 to 77.0) | +1.76 % |
| prefill@2000 | **+4.35 %** [+3.73, +4.97] | +2.32 % |
| prefill@16000 / @64000 | +0.66 / +0.14 % | +1.10 / +0.45 % |

Bolt ctx 262144, A = default (all 1.2 GB in VRAM), B = 0.3 GB (65K cells): decode +2.64 / +2.92 / +2.26 % at 2K / 16K / 64K, prefill +5.2 (outlier-prone first request) / +2.2 / +1.8 %; SHAs equal.
Everything is better with the smaller hot head: the expert slots it frees (Ornith 2.4 to 1.2 GB: expert misses at 2K 3013 to 1927) pay more than the cells beyond the 64K point cost at 2K, 16K and 64K.
The decision fixed before is met by both Ornith budgets (decode@64000 within 2 % of 2.4 GB or better, 2K better). The 128K point is not measured by the 64K targets: 1.2 GB covers only 78K cells.
Taken: **cover 128,000 cells** of every layer (Ornith 1.78 GB: the 1.8 GB arm, +2.15 / +3.16 / +1.76 % over 2.4 GB; Bolt about 0.6 GB), capped at 13.5 % of the free VRAM after the reserve (`KvSplit::default_budget`).

**Coder upper bound** (ctx 70000, A = default placement, B = `TONA_KV_HOST=0`, 2 rounds, SHAs equal): decode@2000 -2.20 % [-3.12, -1.27] (64.6 to 63.2), **decode@64000 +7.09 %** [+6.28, +7.90] (67.7 to 72.5),
prefill@2000 -1.51 %, **prefill@64000 -6.33 %** [-7.14, -5.52] (1967 to 1843; the slots the KV takes). The plan's rule (above +5 % at 64K: propose a hot head behind its own flag) is formally met on decode, but it is
a trade, not a gain: +7 % decode at 64K for -6 % prefill at 64K and -2 % decode at 2K, with all of the KV in VRAM (0.8 GB at ctx 70000; at the default context 262144 the KV does not fit at all, "VRAM only fits 0 expert slots").
**Proposal, not built**: a hot head for the Coder would need the QSA read sites (`mega_op_attn.h` x3, `attn_wmma1/2.h`, `mega_op_qsa.h`, `prefill_qsa.hip`) and the Coder's goldens byte-identical; a hot head of a size
below the all-in-VRAM bound would capture part of the +7 % at less slot cost, and the Coder's attention reads only the selected cells, so most of its KV is read rarely. Left for a ticket with its own measurement; the Coder's code
and placement are unchanged here.

### Block 5 (plan, written before its run): the shipped default

**Question.** With the default of "128,000 cells, at most 13.5 % of the free VRAM" (Ornith 1.78 GB, Bolt 0.6 GB): (a) Ornith against the all-in-RAM placement incl. a 128K point; (b) Ornith against the per-layer split of main at 2K and 64K;
(c) Bolt against its old placement; (d) the Coder's default workload untouched. **Runs** (one `exclusive` job, `ab.py`, A = `kvh-base`, B = `kvh`, ctx 262144 except (d)): (a) A with `TONA_KV_VRAM_GB=0`, `server`, targets
2000:100,16000:100,64000:50,128000:50, 2 rounds; (b) default A, targets 2000:100,64000:50, 2 rounds; (c) Bolt `server` 2000:100,16000:100,64000:50,128000:50, 3 rounds; (d) Coder `short`, 3 rounds.
**Expectation (sources: blocks 2 to 4).** (a) decode@2000 -1.5..-3.5 % (1.8 GB is +2.15 % over 2.4 GB, which was -2.85 %: about -0.8 %; wide), @16000 +50..+60 %, @64000 +115..+135 %, @128000 +100..+220 % (RAM read bound,
the +119 % of 64K is the same bytes per cell), prefill@2000 -4.5..-1 %, @16000 and @64000 -1.5..+1.5 %, @128000 -2..+2 %. (b) decode@2000 +1..+4 %, @64000 +42..+55 %, prefill@2000 0..+3 %. (c) decode@2000 -1..+2 %, @16000 +3..+8 %,
@64000 +20..+26 %, @128000 +20..+50 % (RAM bytes per cell are 2.3 times fewer than Ornith's), prefill -2..+8 %. (d) decode and prefill -0.5..+0.5 %, SHA equal. SHAs of the Ornith arms equal; Bolt SHAs equal at equal plans.
**Decisions fixed before:** a 128K point of Ornith where B gains less than +60 % = the cover is too small or the 128K read has another cause: read the log before accepting; any Ornith SHA different = stop.
**Active tooling:** none.

### Block 5 result: the shipped default (job `kvh-b5`, `work/ab/kvh-b5{a,b,c,d}.*`, 2 rounds a and b, 3 rounds c and d; quiet check before and after in `/var/tmp/claude/kvh/quiet-*-b5.txt`)

Default budget at ctx 262144 (server log): Ornith 1.78 GB, 127,744 cells of every layer; Bolt about 0.6 GB. **Goldens first (build `kvh`, exit 0 each):** `test_gpu_fattn` (with the new split cases), `test_gpu_mamba`, `test_gpu_moe_latent`,
`test_gpu_bolt_quant`, `golden.sh compare` decode, cpu plan, prefill with dflash, `golden-server.sh compare`: ALL-GREEN (`/var/tmp/claude/kvh/golden-rc.txt`); `kv_place` unit test passes.

**(a) Ornith against all in RAM** (ctx 262144, 2 rounds, rounds agree within 0.4 %; SHAs equal at all four lengths, including the 128K point: 0d18e8629f9f0513):

| | A (RAM) to B (hot head) | B vs A | plan |
|---|---|---|---|
| decode@2000 | 113.4 to 112.5 | -0.75 % [-1.58, +0.07] | -3.5..-1.5 %: better than planned |
| decode@16000 | 94.4 to 143.7 | **+52.3 %** | +50..+60 % |
| decode@64000 | 35.1 to 76.1 | **+116.5 %** | +115..+135 % |
| decode@128000 | 14.6 to 41.8 | **+186.8 %** | +100..+220 % |
| prefill@2000 | 2728 to 2622 | -3.88 % [-4.81, -2.95] | -4.5..-1 % |
| prefill@16000 / @64000 / @128000 | | +0.12 / +0.48 / +2.02 % | within |

All inside the plan. The 64K decode is 76.1 against 77.4 tok/s with everything in VRAM at ctx 70000 (block 2c) and 16K 143.7 against 146.7: the hot head gives at ctx 262144 within 2 % the decode of the all-in-VRAM placement.
**(b) against the per-layer split of main** (2 rounds): decode@2000 **+2.66 %** (109.6 to 112.5), decode@64000 **+49.9 %** (57.7 to 86.5; the A arm of this run is faster than in block 2, 52.6, the cause was not chased:
A and B ran in alternating order in the same job), prefill@2000 +1.34 %, prefill@64000 +1.09 %; SHAs equal. The 2K losses are not worse than the per-layer split's, they are smaller.
**(c) Bolt** (3 rounds, spread 0.1 %; SHAs equal at 2K, 16K, 64K, 128K): decode@2000 -1.19 % [-1.77, -0.61], @16000 **+4.19 %**, @64000 **+21.9 %**, @128000 **+36.5 %** (23.4 to 31.9 tok/s); prefill@2000 -0.83 %, @16000 +0.33 %, @64000 **+8.0 %**, @128000 **+14.9 %**.
All inside the plan. **(d) The Coder** (`short`, 3 rounds, final build): decode@2000 -0.10 % [-0.72, +0.52] (76.0 to 75.8 tok/s), :write decode +0.03 %, prefill@2000 -0.96 % [-2.4, +0.5] (A rounds 1446 to 1472, B's round 1 was 1406: first-request outlier,
rounds 2 and 3 equal at 1469 and 1467), prefill:write -1.16 % (B round 1 1686, rounds 2 and 3 1750.3 / 1750.9 = A); SHA 75eff9b8784c7d4c both, expert counters identical. The reference is neither slower nor different; its goldens are byte-identical.

### Block 6 (plan, written before its run): stash and restore across the split

**Question.** Does a conversation stashed with positions on both sides of the split come back bit-exact (stash and restore copy the VRAM part and the pinned part)? Block 1 restored only a prefix inside the hot head. **Run.** One `exclusive` job, Ornith, deterministic,
ctx 262144, `TONA_KV_VRAM_GB=0.5` in B (split at 17152 cells), A = `kvh-base` with 0 (all RAM), targets 20000:50,64000:50 (the 64K request reuses the 20K conversation from RAM: the restore crosses the split), 2 rounds.
**Expectation.** The log of B says "20000 reused ... from RAM" for the second request, SHAs equal at both lengths. **Decision fixed before:** a different SHA = the stash or restore copies the wrong range, stop.

### Block 6 result and block 6b (plan, written before its run)

Block 6 as planned (`work/ab/kvh-b6.*`): SHAs equal at 20000 and 64000 (3372503555279df7, b554c21d7cba2dfc) but the log says **0 tokens reused**: the server workload's prompts of different targets
share no prefix unless one is a prefix of the other, so no restore happened. The question is open. **Block 6b:** the targets of block 1 (2000:100,16000:100,64000:50), where the 64K request reuses 15547 tokens of the 16K conversation
(seen in block 1's log: "15547 reused, conversation from RAM"), with `TONA_KV_VRAM_GB=0.45` in B (MTP layer 0.30 GB, the split at about 12.8K cells, so the 15547 reused tokens are on both sides of it), deterministic, 2 rounds.
**Expectation.** B's log shows the reuse; SHAs equal at all three lengths (the hashes of block 1). **Decision fixed before:** as block 6.

### Block 6b result (job `kvh-b6b`, `work/ab/kvh-b6b.*`)

B log: "budget 0.45 GB ... 12800 of 262144 cells of every attention layer in VRAM"; the 64K request: "15547 reused ... conversation from RAM in 30 ms" (restore of positions on both sides of the split). SHAs **equal at 2000, 16000 and 64000**
(4cf94b9cdeda674c, e85b642df9178deb, 4936cc76a20b07ef), stable. Stash and restore across the split are bit-exact.

## Summary

- Placement by position replaces the layer split (Ornith and Bolt; the Coder is untouched, `KvSplit` unplanned). Output identical wherever the plan is the same (Ornith 2K to 128K, Bolt 2K, 64K, 128K); Bolt's tokens depend on the prefill plan and the expert cache state even with
  `TONA_DETERMINISTIC=1` (block 1b), which is unrelated to the placement.
- **Ornith ctx 262144, decode** against all in RAM: -0.75 % (2K), +52 % (16K), +117 % (64K), +187 % (128K); against the per-layer split: +2.7 % (2K), +49.9 % (64K); prefill@2K -3.9 % against all in RAM (the per-layer split cost -5.4 %), against the per-layer split +1.3 %.
- **Bolt**: decode -1.2 % (2K), +4.2 % (16K), +21.9 % (64K), +36.5 % (128K); prefill +8.0 % (64K), +14.9 % (128K).
- **Coder**: neutral (decode -0.10 %, prefill -0.96 % with a first-request outlier, SHAs and goldens identical). A hot head would give at most +7 % decode at 64K at ctx 70000 for -6 % prefill at 64K: not built (block 4).
- **Budget rule** (`KvSplit::default_budget`): cover 128,000 cells of every layer plus the whole MTP layer, at most 13.5 % of the VRAM free after the `TONA_VRAM_RESERVE_MB` reserve (1 GiB; the desktop holds 1.4 to 3 GB, measured 1.83 to 2.15 GiB on this machine today;
  the reserve is taken from the sysfs reading at load); `TONA_KV_VRAM_GB` and `TONA_KV_HOST` override. Ornith 1.78 GB, Bolt 0.6 GB here. The 13.5 % is the 1.8 GB of Ornith out of the 13.4 GB free; a smaller card gets a smaller share.
- Open: the 13.5 % rule has one data point (this card); the elastic governor (`TONA_VRAM_GOVERNOR`) does not know the KV and cannot shrink it when the desktop grows later; the Coder's hot head; Bolt's plan-dependent determinism.
