# Long context: stash budget at 1M, a needle test that decides, larger indexer top-k (#77, phase 2)

Date: 2026-10-31. Build `work/build/w77b` (branch `77-long-ctx2`, main f72c8de), reference `work/build/p1` (main f72c8de).
Model Coder IQ1_M, MTP head, desktop on the GPU, `systemd-run --user --scope -p MemoryMax=54G -p MemorySwapMax=0`.
Raw data (per-needle JSONL, server logs, scripts) is in `/var/tmp/claude/w77b/` (`out/nd/`, `out/srv-*.log`). Rates are
named: **prefill** (prompt tokens per s) or **decode** (generated tokens per s, MTP).

## 1. Stash budget (1M OOM)

**Correction to phase 1:** a 1M conversation is **16.8 GB** as a stash (1,002,016 tokens; 0.5 GB state plus
KV and pool), not the 14.5 GB of the KV alone. At 524K it is 8.9 GB.

Cause: `stash_current` copied the conversation first and `trim_stash` dropped it afterwards, so the copy was made on top
of 45 GB of pinned memory whatever `IE_PROMPT_RAM_GB` said. Fix: `Backend::stash_bytes()` gives the size before the copy;
`backend::stash_fits` stores only if it is at most `IE_PROMPT_RAM_GB` and leaves 8 GB of RAM free (MemAvailable and the
headroom of every cgroup up to the root, with anonymous, shared and kernel memory counted as used, page cache not).
Older stashes are dropped first, the one being fetched stays. A conversation that does not fit is not stored (also not
on the SSD, the checkpoint is written from the RAM copy), the log says why once. **With the default budget (12 GB) a 1M
conversation is never stored.**

| Run | Result |
|---|---|
| `--ctx 1048576`, `IE_SEQS=1`, `MemoryMax=54G`, 1,001,235-token conversation (prefill 927 s = 1,080 tok/s), then two independent requests | log: `Conversation of 1002016 tokens not stashed: needs 16.8 GB, budget 12.0 GB (IE_PROMPT_RAM_GB), free RAM 14.4 GB`; no OOM; both requests answered (93.2 / 98.5 **decode** tok/s) |
| 100K, `IE_PROMPT_RAM_GB=1`, `IE_SEQS=1`, two texts | `not stashed: needs 1.8 GB, budget 1.0 GB`, both texts answered |
| unit test `test_env_budget` (both modes) | pass, including the numbers of the hang below |

**Hang found on the way.** `--ctx 540672`, two sequence slots (default), `MemoryMax=54G`, third independent 524K text:
both GPU slots full, `stash_current` of 8.9 GB (the first version of the rule left 0.5 GB). The engine thread stood
**48 min in `hipMemcpy`** (`mega_state.cpp:101`, device to host copy of the indexer pool; gdb: `hipMemcpy` ->
`hsa_signal_wait_scacquire`, GPU idle at 2 %, thread at 100 % CPU, `/status` answered with `state=prefill`,
`prefill_done=0`). At that moment: memory.current 52.9 of 58.0 GB (anon 9.5 + shmem 43.2), host MemAvailable 5 GB, free
0.5 GB, swap 5.3 GB used, VRAM 16.5 of 17.1 GB; no kernel message (`journalctl -k`), scope peak 49.2 GB. Before the copy the
rule saw about 12.6 GB of headroom and passed it. Cause not established (not tested on p1). Fix: 8 GB must remain free
after the copy (twice the 4 GB first guess, which this case would still have passed). Counter test, same sequence
(2 slots, three 524K texts, watchdog 900 s on a prefill): log `not stashed: needs 8.9 GB, budget 12.0 GB, free RAM
14.0 GB`, no hang, all three texts answered. Blocking `hipMemcpy` without a time limit remains in the stash path
(`engine/gpu/mega_state.cpp`: `Mega::stash`, `Mega::unstash`/`restore_qsa`; `Mega::load_state` uses `hipMemcpyAsync` plus
`hipStreamSynchronize`); not changed.

Small context unchanged: `scripts/golden.sh compare` (decode csv, rust, explain), prefill dflash (long) and
golden-server identical to the stored output (section 3).

## 2. The needle test (`bench/needle/`)

Greedy, chat format, one conversation per text (document and first question, then every question as a further turn),
40 needles per text at five depths, 20 asked singly and 20 in one answer. Paired comparison by McNemar. Details,
sample size and limits: `bench/needle/README.md`. 120 pairs find 15 points at a discordance of 0.35 (80 % power).

**Plain test (sanity), `off`:** hits of 160 (4 texts) at 64K 159, 128K 160, 250K 160; 512K 40/40 and 1M 39/40 (1 text).
It is saturated, so it cannot show a gain. Prefix reuse holds at every length: follow-up turns compute 26 to 28 tokens
(1,000,000 reused at 1M); a turn takes 0.4 s there.

**Hard test (`--hard`), `off`:** each needle is revision 3, revisions 1 and 2 and a lookalike vault `<name>-b` stand
elsewhere (120 decoy lines per text).

| length | texts | hits / needles | all [95 % Wilson] | single | multi |
|---|---|---|---|---|---|
| 64K | 4 | 82/160 | 51.2 % [43.6, 58.9] | 66.2 % | 36.2 % |
| 128K | 4 | 80/160 | 50.0 % [42.3, 57.7] | 60.0 % | 40.0 % |
| 250K | 4 | 69/160 | 43.1 % [35.7, 50.9] | 47.5 % | 38.8 % |
| 512K | 3 | 47/120 | 39.2 % [30.9, 48.1] | 38.3 % | 40.0 % |
| 1M | 3 | 47/120 | 39.2 % [30.9, 48.1] | 40.0 % | 38.3 % |

By depth (`off`): the first 10 % of the document 84 / 81 / 91 / 50 / 71 %, the last 10 % 12 / 6 / 3 / 4 / 21 % (64K to 1M).
**What the misses are** (single questions, `off`; 27, 32, 42, 37, 36 misses at 64K to 1M): 89 to 100 % are the code of a
decoy line of the asked vault (stale revision 21, 30, 31, 29, 27; lookalike 6, 1, 9, 6, 5), 0 to 11 % a code that is in no
decoy line (0, 1, 2, 2, 4), none without a code. The vault is found; the model takes the wrong revision.

## 3. Larger top-k: `IE_IDX_TOPK` (C3)

Switch: `IE_IDX_TOPK=<tokens>` (default: the model's 2048 = 512 blocks of 4), applied in the hparams, so decode,
prefill, MTP layer and CPU reference see the same value. **Limits:** the kernels hold `MEGA_MAX_CELLS = 4096` cells per
token (LDS `sc`, merge buffers, `cells` rows); top-k plus a tail of up to 3 tokens must fit, so the largest value is
**4092 tokens (1023 blocks), 2.0x**; the start-up check in `gpu/mega_init.cpp` refuses more. The radix `k_topk` (#71) has
no limit of its own (`n_top` is a parameter, histogram 2048 bins x 3 passes). A factor of 4 (8192 tokens) needs
`MEGA_MAX_CELLS` 8192: LDS per block and the merge area double for every kernel including the reference
decode, so it was not built (rule 1). A stored conversation made with another value does not match (fingerprint).

**Retrieval, hard test, `off` (2048) against 4092** (paired over the same needles; single / multi / all):

| length | texts | 2048 all | 4092 all | diff pp [95 %] (all) | McNemar p | only 2048 / only 4092 |
|---|---|---|---|---|---|---|
| 250K | 4 | 43.1 % | 47.5 % | +4.4 [-0.3, +9.1] | 0.118 | 4 / 11 |
| 512K | 3 | 39.2 % | 39.2 % | +0.0 [-7.3, +7.3] | 1.000 | 10 / 10 |
| 1M | 3 | 39.2 % | 34.2 % | -5.0 [-12.6, +2.6] | 0.286 | 14 / 8 |

Single / multi: 250K +6.2 [-1.0, +13.5] / +2.5; 512K +6.7 / -6.7; 1M -1.7 / -8.3 (all p >= 0.18). The 4092 run misses
are decoy codes as well (stale 28, 26, 26; lookalike 6, 4, 8; other 3, 3, 3). Intervals are naive (needles of a text are
not independent); with 3 to 4 texts only a difference of 10 points or more would show.

**Cost** (server, `IE_SEQS=1`, `IE_PROMPT_RAM_GB=0`, ABBA order, desktop on the GPU; "Summarize" prompt of source code, greedy,
200 tokens; pass time = tokens per pass / **decode** rate):

| length | variant | **prefill** tok/s (runs) | **decode** tok/s | tokens per pass | decode pass ms |
|---|---|---|---|---|---|
| 128K | 2048 | 1,726-1,727 (6) | 66.9 | 2.98 | 44.6 (44.1-45.0) |
| 128K | 4092 | 1,606-1,610 (6) | 63.4 | 2.95 | 46.5 (45.7-47.1) |
| 512K | 2048 | 1,434 (4) | 60.5 | 2.83 | 46.5 (44.5-48.6) |
| 512K | 4092 | 1,336-1,348 (4) | 55.9 | 2.75 | 49.1 (47.0-51.1) |

4092 costs 6.8 % of the **prefill** rate at 128K and 6.1 % at 512K, and 4.3 % / 5.6 % of the decode pass time. In the
needle runs the first turn gives **prefill** 250K 1,659 -> 1,511, 1M 1,066 -> 1,029 tok/s (-8.9 % / -3.5 %).

**Quality at 128K** (4,096 tokens through the megakernel, reference = default top-k, method of #73):

```
run                      n   NLL ref NLL here   dNLL   95% low  95% up     KL    top1=
cpp-131072-4096-topk4092 4096  0.2513  0.2476  -0.0037  -0.0085  +0.0007  0.01195  98.85
sh-131072-4096-topk4092  4096  0.7769  0.7819  +0.0050  -0.0021  +0.0122  0.01923  96.53
```

C++ passes the 0.01 rule (upper end 0.0007), shell is undecided (mean 0.0050, upper end 0.0122).

**Default is untouched:** `IE_DETERMINISTIC=1`, greedy, 200 tokens, 30K prompt then 130K prompt, server, 2 pairs
alternating p1 / w77b: SHA `065922c2ff602991` (30K) and `439aad472c59ae6f` (130K) in all four runs. (One w77b run
decoded the 130K request at 40.2 tok/s instead of 64-65; the others and the output hash are normal, cause not
examined.) Golden on w77b: decode (csv, rust, explain), prefill dflash (long) identical; golden-server all 20 cases identical, exit 0.

**Barrier abort (#75):** the first teacher-forced reference run of the shell text at 128K (default top-k) ended with
`Error: gpu::Mega: barrier aborted (time limit) at position 131841 (step 15: matvec moe.shared, longest step 4 dn.in
0.1 ms, pass up to there 0.3 ms, blocks at step 15, all equal)`. The rerun of the same command passed (NLL 0.7769); no
other abort in about 60 server and tool runs.

## 4. Findings

1. **The selection does not limit at 250K to 1M.** Top-k doubled from 512 to 1023 blocks: +4.4 / 0.0 / -5.0 points
   (250K / 512K / 1M), none significant; the upper ends of the intervals are +9.1, +7.3 and +2.6, so a gain of 10 points is excluded at
   all three lengths (3 to 4 texts). A limited selection would have shown as a gain; it does not.
2. **The hard test misses are mostly not lost needles:** the model answers with a code of the asked vault in 89 to 100 %
   of the misses at every length, also at 64K (51 %), where the length is not the issue. Reading the revisions limits;
   distance acts little (51 / 50 / 43 / 39 / 39 % from 64K to 1M: 12 points over a factor 16 in length, while the depth within a
   document moves the rate by 60 to 90 points).
3. **Plain retrieval is at 99-100 % up to 512K and 97.5 % at 1M** (1 text); the state (DeltaNet) and RoPE distance do
   not show a loss in this test.
4. **Cost of 4092** is 6-7 % **prefill** and 4-6 % decode pass for no measured retrieval gain.

## 5. Recommendation

- `IE_IDX_TOPK` stays at the model's value for every length (2048). No gain to pay 6 % **prefill** and 5 % decode for.
- Jet-Long (C2): not justified by these data; the failing part is not selection and not distance. A test that loses
  needles at long range (not decoy revisions) would be needed before it, and none of ours does at 1M (plain 97.5 %).
- Open: indexer recall (C1 experiment 2) is still not measured; with the selection not limiting, it has low priority.
  The multi-turn needle test at 1M (3 texts, 17 min each) takes 51 min per variant.
