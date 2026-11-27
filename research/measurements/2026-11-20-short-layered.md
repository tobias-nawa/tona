# 2026-11-20: layered prefill for short Ornith prompts, on top of the layered-prefill fix (#175)

Branch `worktree-<worktree>`, base `62b6f85`. Builds `work/build/sl-a` (main, A) and `work/build/sl-b` (branch, B). Logs and scripts: `/var/tmp/claude/sl/`
(`job.sh`). Every run is a queued job; quiet checks before and after are in `/var/tmp/claude/sl/quiet-{before,after}-<job>.txt`.
Starting point: `2026-11-20-ornith-prefill.md` result 3 (levers measured before the fix, so multi-chunk numbers were not valid) and
`2026-11-20-bolt-determinism.md` (layered prefill now bit-equal to chunked).

## Test conditions (before the first run, 10:32)

`python3 tools/hw_report.py`: Ryzen 9 9950X (16C/32T), RX 9060 XT gfx1200 15.9 GiB (3.92 GiB held by others), RAM 60.5 GiB (48.2 GiB available), kernel 7.3.4,
hip 7.16, no `TONA_*` set, commit `62b6f85` plus the hook patch. `tools/quiet_check.py`: do not measure (CPU 82 % used, load 11): the build of arm B ran at that moment;
no number comes from that state. Each job below has its own quiet check before and after.

## Change

A default-0 hook `PrefillSequence::short_layered_min()` (shortest prompt that goes layer by layer with one chunk of its own size); Ornith first 1024, final 640 (Result 3). A second hook
`layered_chunk()` (Ornith 8192) was built first and dropped (Result 2). `TONA_PF_LAYERED_MIN` and `TONA_PF_LCHUNK` still override. The Coder and Bolt keep 0.

## Plan 1: identity (job `id`, `gpu` kind)

Question: is the layered plan bit-equal to chunked at the sizes the hook now changes? Run: `ab.py`, both arms `sl-b`, A with `TONA_PF_LAYERED=0` (chunked reference), B default,
both `TONA_DETERMINISTIC=1` and `TONA_PF_DEBUG=1`; prompts 500, 1000, 1500, 2000, 6000, 16000 (code), 100 decode tokens, 2 rounds (no early stop). About 8 min (5 starts).
Expectation (source: g8 of the determinism file, layered = chunked bit for bit): SHA A = SHA B at every size, and the plan lines show chunk by chunk at 500 and 1000 in B
only if the threshold excludes them (the threshold is `n > 1023`, so 1000 is chunked, 1500 and above layered). Tolerance none. Active tooling: the two variables above, shipped profile.
Decision: any SHA difference at a layered size = the hook is not shipped, find the cause.

## Plan 2: speed (job `speed`, `exclusive` kind)

Question: prefill (and decode) rate of Ornith at 500, 1000, 2000, 16000, 64000 tokens, B (branch) against A (main), `--ctx 262144`. Rounds: `ab.py` decides, 2 to 4
(no known spread for most of these targets in `bench/noise.toml`; `prefill@2000` 0.09 %, `prefill@2000:write` 0.16 %). About 12 min.
Expectation (B against A, source `2026-11-20-ornith-prefill.md` result 3, one-chunk numbers; larger chunk +1.2 % / +1.3 % measured on the faulty path, so wide):
prefill@500 -1..+1 % (below the threshold, same plan), prefill@1000 -1..+5 % (+3 % measured), prefill@2000 +8..+18 %, prefill@16000 and @64000 -1..+2.5 %, decode -1..+1 %
(SHAs equal, same tokens). Active tooling: `TONA_PF_DEBUG=1` (log only), shipped expert profile. Decision: the threshold is the smallest size with a gain beyond the margin, 500 and
1000 must not lose; a loss at any size moves the threshold above it.

## Plan 3: Bolt check (job `bolt`, `exclusive` kind)

Question: does forcing layered prefill from one token (`TONA_PF_LAYERED_MIN=1`) change Bolt's plan or rate at 1000 and 2000 tokens? Both arms `sl-b`. Expectation: both arms
chunk by chunk (the layered plan needs more than the loan limit at the default slot count), prefill equal within 1 %; if the plan line says layered, the rate is the finding.
Active tooling: `TONA_PF_DEBUG=1`. Duration 6 min.

## Result 1 (job `id`, `work/ab/sl-id.md`, quiet checks: "do not measure" for the 3.9 to 4.1 GiB of VRAM held by the desktop only)

Server prompts in this table are named by the ab.py target; the real token counts (plan lines) are 300 (500), 959 (1000), 1269 (1500), 1907 (2000), 3872 (6000) and 9769 (16000).
A = chunk by chunk (`TONA_PF_LAYERED=0`), B = default of the branch, `TONA_DETERMINISTIC=1`: **SHA A = SHA B at all six sizes**, stable over both rounds. The plan lines of B: chunk by chunk at 300 and 959 tokens
(below the threshold), layer by layer, one chunk, at 1269 (chunk 1280), 1907 (1920), 3872 (3968) and 9769 (three chunks, 4992). The prefill rates of this job (layered against chunked, not against main): 1269 tokens +17.5 %,
1907 +11.7 %, 3872 +33.5 %, 9769 +36.6 %; not a claim against main, which already takes layered above 2048 tokens.

## Result 2 (job `speed`, `work/ab/sl-speed.md`, A = main, B = branch, 4 rounds, quiet check "do not measure" for 4.0 GiB of VRAM held by the desktop only)

Real prompt tokens: 300, 959, 1933, 15547, 48335. Prefill (rates per request): 300 tokens +0.17 % [-2.9, +3.2], 959 +0.07 % [-0.8, +0.9] (both below the threshold: same plan),
**1933 tokens +15.31 % [+14.6, +16.0]** (3,165 to 3,651 tok/s; layered one chunk 1,933 against chunked 2,048), 15547 +0.06 %, 48335 -0.03 % (equal). Decode: all within -0.2 to +0.2 %.
SHAs equal in all five. Both plan lines at 15547 and 48335 read `layer by layer, chunk 6144`: **the larger layered chunk (8192) does not fit in the borrowed slots** (the search starts at 8192
and halves down), so the hook `layered_chunk()` changed nothing. The +1.2 % at 16K and 64K measured earlier (before the fix, TONA_PF_LCHUNK set explicitly) is not reproduced by the default. The hook is removed.

First start of `thr` (target 1200) ended in an ab.py crash (division by zero after round 1: the 1200 request produced no prefill rate, the server log has no plan line for it; not analysed further, no report written,
logs in `work/ab/sl-thr-logs`). Re-planned with target 1500 (1269 real tokens, known to work in job `id`) in place of 1200: targets 500, 700, 1000, 1500; expectation for prefill@1500 +8..+22 % (Result 1: +17.5 % against chunked), the others as in Plan 4.

## Plan 4: crossover (job `thr`, `exclusive`)

Question: from which prompt length does the layered one-chunk plan beat chunk by chunk? Both arms `sl-b` (shipped hook, threshold 1024 tokens), B with `TONA_PF_LAYERED_MIN=1` (layered from one token),
A default (chunked below 1024). Targets 500, 700, 1000, 1200 (about 300, 560, 959 and 1100 real tokens), 3 to 4 rounds, about 15 min. Expectation (source: result 3 of the Ornith prefill file: -7 % at 500, +3 % at 1K, +10 % at 1.5K,
and Result 1 above: +17.5 % at 1269 tokens): prefill@500 -12..0 %, @700 -8..+4 %, @1000 -2..+8 %, @1200 +2..+14 %, decode -1..+1 %. Active tooling: `TONA_PF_DEBUG=1`. Decision: the threshold is the smallest length where B beats A beyond
the margin (0.5 %) with the layered plan taken; a loss at a length keeps it chunked.

## Result 3 (job `thr`, second start, `work/ab/sl-thr.md`, B = layered from one token, A = chunked below 1024; 4 rounds; quiet check "do not measure" for 4.1 GiB of VRAM held by the desktop)

Real tokens and **prefill** B against A: 300 tokens **-12.78 % [-15.3, -10.3]** (outside the planned -12..0 by a hair, B slower), 617 tokens -0.22 % [-1.0, +0.5] (equal), 959 tokens **+12.92 % [+12.2, +13.7]**
(2,152 to 2,430 tok/s; plan line: layer by layer, chunk 1024), 1269 tokens +0.13 % (both arms layered: the control). Decode equal within noise (-0.6 to +0.04 %).
Interpolating the three points the crossover is near 620 tokens (the layered copy moves every missing expert before the routing is known: more bytes below that size, as in the Ornith prefill file).
**Decision: the threshold is 640 tokens** (a multiple of the chunk grain 128, just above the crossover where the gain is still near zero); `layered_chunk()` is dropped (Result 2: the chunk of 8192 never fits).
Final code: `short_layered_min()` of the Ornith sequence returns 640, the plan search is unchanged for prompts above the layered chunk.

## Plan 5: the final build against main (job `final`, `exclusive`)

Change after the first start of this job (crash, division by zero as in `thr`): target 800 gives no prefill of its own (the corpus has no prompt between 617 and 959 tokens; the same reason as 1200 in `thr`), so the
targets are 500, 700, 1000, 6000, 16000 (300, 617, 959, 3.9K to 5.8K, 9.8K real tokens) and the 800 expectation is dropped; the 617 to 959 range stays unmeasured beyond the two points.

A = `sl-a` (main), B = `sl-b` (rebuilt with threshold 640, no chunk hook), `--ctx 262144`, targets 500, 700, 800, 1000, 6000, 16000 (about 300, 617, 770, 959, 3.9K, 9.8K real tokens), 200 decode tokens, 2 to 4 rounds, about 16 min.
Expectation (B against A, prefill unless named): 500 and 700 -1..+1 % (below the threshold, same plan), 800 -1..+10 % (just above, gain growing from 0), 1000 +8..+18 %, 6000 -1..+3 % (main layered with chunk
6144 for the whole prompt, the branch with a chunk of the prompt's size: smaller workspace), 16000 -1..+1 % (plan unchanged, Result 2); decode -1..+1 %. Active tooling: `TONA_PF_DEBUG=1`. Decision: any loss beyond the margin at a size moves
the threshold or the chunk clamp back.

## Result 4 (job `final`, second start, `work/ab/sl-final.md`, A = main, B = branch with threshold 640, 4 rounds; quiet check "do not measure" for 4.2 GiB of VRAM held by the desktop)

**Prefill** B against A: 300 tokens -0.22 % [-1.1, +0.7], 617 tokens +0.07 % (both below the threshold: same plan), **959 tokens +11.60 % [+10.9, +12.3]**, 3.9K to 5.8K tokens (target 6000) **-0.44 % [-1.1, +0.3]**
(4,523 against 4,505 tok/s, B lower in all four rounds), 9.8K (target 16000) -0.13 %. Decode: equal at all sizes (the 6000 row +1.35 % is one A round at 99.1 tok/s, medians 104.4 and 104.5 tok/s). SHAs equal at all five.
Finding: the small loss at 3.9K to 5.8K is the clamp of the layered chunk to the prompt's size (chunk 5888 against 6144 in main); it was meant for prompts below one chunk. Fix: the clamp applies only to prompts of up to
one chunk (2048 tokens); longer prompts take main's plan exactly.

## Plan 6: confirmation (job `final2`, `exclusive`)

After the fix (clamp only up to 2048 tokens), rebuilt `sl-b`. A = `sl-a`, B = `sl-b`, targets 1000, 2000, 6000 (959, 1933, 3.9K to 5.8K real tokens), 200 decode tokens, 3 rounds, about 8 min.
Expectation: prefill@1000 +8..+16 %, prefill@2000 +10..+18 % (Result 2: +15.3 %), prefill@6000 -0.3..+0.3 % (the plan is main's again); decode -1..+1 %. Active tooling: `TONA_PF_DEBUG=1`.
Decision: prefill@6000 within the margin = the code ships.

## Result 5 (job `final2`, `work/ab/sl-final2.md`, A = main, B = final branch, 3 rounds; quiet check "do not measure" for 4.2 GiB of VRAM held by the desktop)

**Prefill** B against A: 959 tokens **+17.67 % [+15.5, +19.9]** (A 1,745 / 1,751 / 1,812, B 2,106 / 2,030 / 2,110 tok/s; A is 20 % slower than in Result 4, 2,150 tok/s, the machine was busier in the A rounds: the
range of the gain is +11.6 to +17.7 %), 1933 tokens **+16.40 %** (expected +10..+18), 3.9K to 5.8K tokens **-0.00 % [-0.75, +0.75]** (plan is main's again, the -0.4 % of Result 4 is gone). Decode -0.6 to -0.3 % at
1000, 2000, 6000 (CIs span 0, SHAs equal in all three; decode runs the same code in both arms). The expectation of +8..+16 % at 959 tokens was exceeded by the noisy A arm, noted.

## Result 6 (job `bolt`, `work/ab/sl-bolt.md`, both arms the branch build, B with `TONA_PF_LAYERED_MIN=1`, 3 rounds)

Bolt (24 GB file, `--ctx 262144`): both arms log `chunk by chunk` at 944 and 1901 tokens (loan limit 1383 MB): **forcing layered from one token does not make Bolt layered here**, the plan needs more borrowed room than
the slots give at this free VRAM (as in `2026-11-20-bolt-determinism.md` g1). Prefill -1.10 % [-2.0, -0.2] at 944 tokens and -0.29 % at 1901, with the same plan and the same SHAs: not a plan effect, noise of
this workload (no known spread in `bench/noise.toml` for it); decode -0.24 / -0.09 %. **Bolt is not changed**: no hook, no measurable gain to take (its layered plan is unreachable at the default slot count).

## Summary and open points

Shipped: `PrefillSequence::short_layered_min()` (default 0), Ornith 640 tokens; prompts of 640 to 2048 tokens go layer by layer with one chunk of their size. Prefill gain +11.6 to +17.7 % at 959 tokens,
+15.3 / +16.4 % at 1933 tokens, nothing elsewhere; threshold from the crossover near 620 tokens (300 tokens -12.8 %, 617 -0.2 %, 959 +12.9 %). Identity: SHAs equal to the chunked plan at 300, 959, 1269, 1907, 3872 and 9769 real
tokens (`TONA_DETERMINISTIC=1`). The larger layered chunk (8192) does not fit and is not built. Open: the 617 to 959 range has no corpus prompt (two points only); the layered copy moves every expert of the next layer, so a
router-aware copy (the experts a short prompt uses, known only after the routing) remains the way to win at 300 tokens.

## Gates (final build `sl-b` after the refactor into `layered_from` and `first_layered_chunk`, which is pure: same arithmetic as the measured build; quiet check "do not measure" for 4.3 GiB of VRAM held by the desktop only)

`just check` (hooks, openmp, clang-tidy of the changed files): green. `ctest -L unit` (71 tests): green. `just test-py`: 681 passed, 1 skipped (torch). Coder goldens: decode gpu plan and cpu plan (csv, rust, explain), prefill (long), prefill
dflash (long), server (all 44 cases): identical. `test_gpu_fattn`, `test_gpu_attn_wmma`, `test_gpu_mamba`, `test_gpu_moe_latent`, `test_gpu_bolt_quant`, `test_prefill_order`: exit 0. The speed and identity runs above used the build before the refactor; they were not repeated
on the refactored build (the refactor moves two expressions into functions).
