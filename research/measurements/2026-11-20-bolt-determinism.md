# 2026-11-20: Bolt determinism under `TONA_DETERMINISTIC=1` (#175)

Branch `worktree-<worktree>`, base `0abbbed`. Build directory `work/build/det`. Logs and scripts: `/var/tmp/claude/bolt-det/`.
Every build and GPU program goes through `scripts/gpu_queue.py`; correctness runs are `gpu` jobs (no speed number is taken from them).
Each run below is planned before it starts; results follow its plan.

Problem (from `2026-11-20-kv-hot-head.md`, blocks 1 and 1b): Bolt (Nemotron 3.5 Bolt 74B-A9B) gives different tokens in rounds of the same
build and arm at 16,000 prompt tokens (expert hits 34027 against 22733), and the tokens depend on the prefill plan.

## Test conditions

`python3 tools/hw_report.py` (2026-11-20 04:12, before the first run, `/var/tmp/claude/bolt-det/hw.txt`):

```
- CPU: AMD Ryzen 9 9950X 16-Core Processor, 16 cores / 32 threads
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 2.57 GiB held by others before the run, 0 % busy before the run
- RAM: 60.5 GiB, 53.1 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.3.4-200.fc44.x86_64 (x86_64)
- Driver and toolkit: hip 7.16.26385-0000000
- TONA_* variables: none set
- Load average (1, 5, 15 min): 0.37, 2.75, 3.43; swap in use 4.56 GiB
- Commit: `0abbbed` (dirty tree: `tona-bolt-bench --repeat`)
```

`python3 tools/quiet_check.py` 04:12: warning (2.57 GiB VRAM held by the desktop, the system swaps a little). Each job repeats the check before
and after (`/var/tmp/claude/bolt-det/quiet-*`). No speed is measured in the determinism runs.

## Tool: `tona-bolt-bench --repeat R`

R times (reset, prefill, greedy decode of `--gen` tokens), one `generated ids hash` line per repeat with the expert hits, misses and CPU expert
count of that repeat. The expert cache state carries over from one repeat to the next. Prompt ids: `/var/tmp/claude/bolt-det/ids-prose.txt`
(the first 69,386 tokens of the prose corpus, the ids of `2026-11-19-bolt-opt2.md`).

## Plan of job `e1`/`e2`/`e3` (one `gpu` queue job each, `/var/tmp/claude/bolt-det/run.sh`)

All runs: `tona-bolt-bench` (Bolt 24 GB file), `TONA_DETERMINISTIC=1`, `TONA_PF_DEBUG=1` (plan lines), `TONA_MEGA_INFO=1` (slot counts),
`--ctx 262144` (the service default; the KV split and the slot count depend on it), 100 greedy tokens, prompt from the prose ids. Active tooling
beyond that: only the variables named in the config list below. No profiler.

### E1: is the output stable from repeat to repeat and from process to process?

Question: at fixed build and flags, do the 3 repeats of one process and the 2 to 3 processes agree? Configs: prompt 2000 (2 processes) and
16000 (3 processes), 3 repeats each; the cache state carries over inside a process, a new process starts cold. Expectation: 2000: one hash
everywhere (one chunk; blocks 1/1b of `2026-11-20-kv-hot-head.md`: equal at 2K); 16000: at least two different hashes among the 9 (blocks 1b:
two rounds of one arm gave different texts), with different expert hit counts. Tolerance: none (a hash is equal or not). Duration about 8 min.
Decision: 16000 unstable inside a process = the state (cache) matters; only across processes = startup state (free VRAM, slots, plan) matters.
2000 unstable = a numeric race in decode or prefill, bisect with E2.

### E2: are CPU experts bit-equal to GPU experts (decode path)?

Prompt 2000 (one chunk, plan identical), 2 repeats: `TONA_PLAN=gpu` (all pinned misses on the GPU), `TONA_PLAN=cpu`, `TONA_CPU_EXPERTS=0`.
Expectation: all hashes equal (the CPU experts use integer dot products and the same fixed-point sum, `gpu/arch/nemotron_h/moe.cpp`; the Coder's
`GOLDEN_PLAN=cpu` golden equals the GPU one). Decision: equal = the expert path is not a source; different = in deterministic mode force one path.
Duration 3 min.

### E3: does the prefill plan change the tokens (decode forced onto the GPU with `TONA_PLAN=gpu`)?

Prompt 16000, 2 repeats: default twice (two processes), `TONA_PF_LAYERED=0` (chunk by chunk, 2048), `TONA_PF_LCHUNK=2048`, `TONA_KV_STAGE=0`,
`TONA_PF_GROUP=4096`. Expectation: the two default runs equal each other (if not, the prefill itself is unstable at a fixed plan, bisect); the plan
variants differ from the default where chunk size or group changes (chunked Mamba-2 KL 0.0013 against sequential, `2026-11-19-bolt-opt2.md`), equal
where only the staging changes. Decision: variants differing = the chunk size defines the output; then fix the plan in deterministic mode.
Duration 10 min.

### Result E1 (job `e1`, quiet checks `/var/tmp/claude/bolt-det/quiet-{before,after}-e1.txt`)

Hashes of 100 decoded tokens, `tona-bolt-bench`, prompt 2000 and 16000, 3 repeats in each of 2 (2000) and 3 (16000) processes:
**all equal**: 2000: `1888fd36065a62f8` in all 6 repeats; 16000: `40bf709329746081` in all 9 repeats, although the expert state differs from repeat
to repeat (repeat 0 against 1 and 2: 26889 / 30413 / 30410 hits, 3022 / 2191 / 2194 experts computed by the CPU). Expectation for 16000 (a difference)
not met: this plan is not the one of the server runs. The log shows `Prefill 15999 tokens: chunk by chunk, chunk 2048` (loan limit 2313 MB):
without a profile the bench pins experts by number and the server runs of the kv-hot-head file used the shipped profile
(`toolchain/profiles/nemotron-bolt-74b-a9b-24gb.bin`, 126 slots per layer, 18.5 GB pinned), so their plan was layered. E3 was started without the
profile and stopped after 10 s (the plan variants are meaningless in a chunk-by-chunk plan); it is replaced by job `f1` below.
First finding: **with equal prefill plan the output does not depend on the expert cache state (GPU or CPU experts, hits, misses)**, 9 repeats over
3 processes with different CPU expert counts per repeat.

## Plan of job `f1` (`gpu` queue job, `/var/tmp/claude/bolt-det/run2.sh f1`: as run.sh plus `TONA_EXPERT_PROFILE` = the shipped Bolt profile, as the server runs)

Same tool, flags and `--ctx 262144` as E1. Prompt 16000, 100 tokens. Configs (processes): `16k-a`, `16k-b` (default, 3 repeats each: does the instability of the
kv-hot-head file show up in the bench when the plan is the server's?); then with `TONA_PLAN=gpu` (decode experts on the GPU only, so a difference is
the prefill) 2 repeats each: default, `TONA_PF_LAYERED=0` (chunk by chunk 2048), `TONA_PF_LCHUNK=2048`, `TONA_KV_STAGE=0`, `TONA_PF_GROUP=4096`;
and `TONA_PLAN=cpu` (default plan). Expectation: default plan layered with chunk 4096 to 6144 (the log line tells); `16k-a` = `16k-b` (E1 saw no
instability with equal plans); the variants with another chunk or group differ from the default hash or not, no expectation from a source (the
kv-hot-head file only guesses a Mamba-2 chunk effect, while `prefill_mamba_ssd.hip` documents bit-equality of cuts at multiples of 64, which a chunk of 128
multiples keeps). Decision: any variant different from `gpu-def` = that input defines the output, find the kernel by tapping (`tools/tap_compare.py`)
between two plans; all equal and `cpu-def` different = CPU experts differ from GPU experts at this plan. Duration about 15 min.

### Result f1 (`/var/tmp/claude/bolt-det/f1.out`)

All 8 processes (16 repeats, `16k-a/b`, `TONA_PLAN=gpu` with five prefill variants, `TONA_PLAN=cpu`): **one hash, `40bf709329746081`** (the same as E1 without the profile).
Experts computed by the CPU 3860 / 0 / 13479 (default, `PLAN=gpu`, `PLAN=cpu`) with the same tokens: **CPU experts are bit-equal to GPU experts** for Bolt in
decode (13,479 expert evaluations moved from GPU to CPU, no token changed). The prefill variants did nothing: every start logged `chunk by chunk, chunk 2048,
group 0, loan limit 2313 MB`, i.e. the tool does not reach the layered plan of the server (the 16K prompt of `tona-server` was layered with 126 slots per layer). So
the bench cannot test the plan; the server must. Next: an A/A of `tona-server` through `scripts/ab.py` (job `g1`).

## Plan of job `g1` (`gpu` queue job, ab.py A/A, same build both arms)

`python3 scripts/ab.py --a det --b det --env TONA_DETERMINISTIC=1 --env TONA_PF_DEBUG=1 --workload server --targets 2000:100,16000:100 --ctx 262144
--drafter none --fixed-rounds --max-rounds 3 --out work/ab/bdet-g1`: 3 rounds, one warm-up start, the arms identical, so any SHA difference is run-to-run.
Scope: 7 server starts about 1 min each. Expectation: SHAs equal in all rounds if the bench result carries over (source: E1/f1); the kv-hot-head file saw
5c55e0ca5fda4040 against another value between two rounds of one arm at 16000 (then at least two SHAs among the 7 starts, the plan lines show the cause). No speed is
claimed (the job is `gpu`, a speed number beside another job would not be valid). Active tooling: `TONA_DETERMINISTIC=1`, `TONA_PF_DEBUG=1`, the shipped profile (set by ab.py).
Decision: unstable = read the plan and the hit counts of the odd start, bisect the input (plan, slots) with `TONA_PF_*` variables in the next job; stable = reproduce
the kv-hot-head conditions (`TONA_KV_STAGE=0` build `kvh-base`) before concluding.

### Result g1 (`work/ab/bdet-g1.md`, `tona-server` through ab.py, same build both arms, 3 rounds + warm-up)

SHA `464308741868de30` (2000) and `356900539592d273` (16000): **stable in all 6 starts of each target**, A = B; expert hits and misses identical per start
(21199 / 14801 and 18693 / 17307). `356900539592d273` is the value of B in block 1 of the kv-hot-head file. Plan lines (`TONA_PF_DEBUG`): `Prefill 15749 tokens:
chunk by chunk, chunk 2048, group 0, loan limit 2313 MB`; 92 expert slots per layer, KV hot head 128000 cells (all KV of the prompt in VRAM). Plain reading:
the instability of the kv-hot-head file is not present in the default configuration of this build. It occurred in the arm `kvh-base` + `TONA_KV_STAGE=0`
(KV cache wholly in pinned RAM, older build) and in `TONA_KV_VRAM_GB=0.1` (B: `fb5a...` in block 1b, `3569...` in block 1). The layered plan is not reachable for
Bolt here (its `pf_layered_need` counts `2 * max_experts * slot_bytes` for the two layer buffers, more than the loan limit of 2.3 GB), so "chunk and group sizes" are
not what differs; what differs in those arms is where the KV cache lives (host-pinned reads in prefill and decode attention).

## Plan of job `f2` (`gpu` queue job, `run2.sh f2`): does the KV placement (hot head size, from the free VRAM) change Bolt's tokens?

Reason: g1 is stable in the default configuration, the instability of the kv-hot-head file sat in arms with the KV cache in RAM or a small hot head, and the hot
head size is derived from the free VRAM (`KvSplit::default_budget`), which is runtime state. Run: `tona-bolt-bench`, 16000 prompt, 2 repeats per process,
`TONA_KV_VRAM_GB` = default (0.59 GB, 128000 cells), 0, 0.05, 0.1, 0.3, and 0 again in a second process (does a RAM-only KV give the same hash twice?); 2000
prompt with 0 and 0.1. Expectation: if placement does not matter, one hash for all (like E1/f1: `40bf709329746081` at 16000); a difference for small hot
heads would mean the attention rounds differently at the split. Tolerance none. Active tooling: `TONA_DETERMINISTIC=1`, `TONA_PF_DEBUG=1`, `TONA_MEGA_INFO=1`,
the shipped profile. The bench also prints the hash of the first decode logits now (new build, rebuilt first as a `cpu` job). Duration about 8 min.
Decision: differing = find where (prefill KV write/stage in RAM, decode attention split) with the logits dump; equal = the placement is innocent, go back to
kvh-base-only behavior and test it with a build of that commit.

### Result f2 (`/var/tmp/claude/bolt-det/f2.out`)

16000 prompt, 2 repeats each: hot head default (128000 cells), 0, 0.05, 0.1, 0.3 GB, and 0 twice: **ids hash `40bf709329746081` and first-logit hash `7e81a64bb9cd5814`
in all 12 repeats**, with 22350 to 24809 expert hits and 3748 to 3870 CPU experts between the configs. 2000 prompt with 0 and 0.1: `1888fd36065a62f8`,
logits `2e460feba1f832ad`. So **KV placement, CPU or GPU experts and the expert cache state do not change a single bit of Bolt's logits** in this build.

## Plan of job `g2` (`gpu` queue job, `run3.sh g2`): are batch widths and the prefill chunk size bit-neutral?

Still open from the problem statement: (a) `TONA_DETERMINISTIC` promises that every pass width up to 4 computes the same bits; the server drafts with width up to 4
(`generator_env.h`), so a Bolt width effect would make the output depend on the acceptance pattern; (b) "the tokens depend on the prefill plan" was a guess.
New build option `--check-widths W,...`: after the prompt, 16 teacher-forced positions in passes of W tokens, hash of every logit row against passes of one token.
Runs: widths 2,3,4,8 at prompt 2000; 2,4 at 16000. Then the prefill chunk (`TONA_PREFILL_CHUNK` 1024, 3072, 4096, 8192; default 2048) at 16000 with `--repeat 1`:
ids hash and first-logit hash against `40bf709329746081` / `7e81a64bb9cd5814`; `TONA_MAMBA_SSD=0` (sequential scan, expected to differ: different rounding,
documented in `prefill_mamba_ssd.hip`, a positive control for the comparison). Expectation: widths 2..4 0 mismatches (design goal and the Coder's
`--batch-check`), width 8 may differ (not claimed under determinism); chunk variants equal (SSD grid is absolute, GEMM rows independent) except `ssd0`.
Tolerance none. Decision: width mismatch at 2..4 = the server's drafting makes Bolt output acceptance dependent, find the kernel; chunk variants differ = the
chunk defines the output, then fix one chunk in deterministic mode. Active tooling: as f2. Duration about 8 min.

### Result g2 (`/var/tmp/claude/bolt-det/g2.out`)

Widths: **0 of 16 positions differ** for widths 2, 3, 4, 8 at 2000 and 2, 4 at 16000 (hash of the full logit row against passes of one token): batch widths are bit-neutral for Bolt.
Chunk: 3072, 4096, 8192 (chunk by chunk) give ids `40bf709329746081`, first logits `7e81a64bb9cd5814` = the default (chunk 2048): the chunk size is bit-neutral in the
chunk-by-chunk plan, as the absolute 64-token SSD grid promises. **`TONA_PREFILL_CHUNK=1024` differs** (`d094d6c639b309cf`, logits `3343aaf58d084fde`), and its log line reads
`layer by layer, chunk 1024, group 8192, loan limit 2615 MB`: the plan became layered. `TONA_MAMBA_SSD=0` differs (`7c7b12522adae48b`, positive control).
So the output depends on whether the prefill runs layer by layer or chunk by chunk (or on chunk 1024 itself; g3 separates the two).
This also explains the kv-hot-head instability: layered needs `pf_layered_need` to fit the loan limit, and the loan limit follows the expert slots per layer, which follow the free VRAM:
the servers of block 1b had 126, 126, 125 slots per layer in the A arm (VRAM 12.92 / 12.87 GB, `work/ab/kvh-b1b-16k-logs`), 122 in the B arm, while the default
bench and the g1 servers had 92 slots (loan limit 2313 MB) and never go layered. A start with a little more free VRAM can cross the fit threshold: another plan, other bits.

## Plan of job `g3` (`gpu` queue job, `run3.sh g3`): which part of the layered prefill changes the bits?

Prompt 16000, `--repeat 1`, `TONA_DETERMINISTIC=1`. (1) `TONA_PREFILL_CHUNK=1024 TONA_PF_LAYERED=0` (chunk by chunk at 1024) and the layered 1024 again: chunk size or layering?
(2) `TONA_PF_LOAN_PCT` 60, 75, 90 (the share of layers whose slots the prefill may borrow, default 50): enlarge the loan limit until the default chunk 2048 goes layered; the
plan line says so. (3) With 90: `TONA_PF_WCACHE=0` (dense weights unpacked per chunk instead of once per layer), `TONA_PF_LOAN_KEEP=0`, both: which feature moves the bits.
Expectation: (1) chunked 1024 = default hash if the chunk size is neutral (source: g2 chunks 3072..8192) and layered 1024 differs again; (2) 60/75/90 layered at some point and differ from
the default; (3) no expectation from a source, the weight cache is the first suspect (a different GEMM route for the unpacked copies). Tolerance none. Decision: the feature that
moves the bits gets a bit-equal route in deterministic mode (or is switched off there). Duration about 8 min.

### Result g3 (`/var/tmp/claude/bolt-det/g3.out`)

Reference (chunk by chunk, chunk 2048 or 1024 or 3072..8192): ids `40bf709329746081`, first logits `7e81a64bb9cd5814`.

| Config | Plan line | ids hash | first logits |
|---|---|---|---|
| `PREFILL_CHUNK=1024 PF_LAYERED=0` | chunk by chunk, chunk 1024 | 40bf709329746081 | 7e81a64bb9cd5814 (equal) |
| `PREFILL_CHUNK=1024` | layer by layer, chunk 1024, group 4096 (g2: group 8192) | 823482ed63fe61ed (g2: d094d6c639b309cf) | ecf47fdf6196bd4f (g2: 3343aaf58d084fde) |
| `PF_LOAN_PCT=60` | chunk by chunk, 2048 (loan limit 2701 MB) | 40bf709329746081 | 7e81a64bb9cd5814 (equal) |
| `PF_LOAN_PCT=75` | **layer by layer, chunk 2048, group 16384** (3358 MB) | e74f008d80235004 | e6fbb8c34d8bac84 |
| `PF_LOAN_PCT=90` | layer by layer, chunk 3072, group 9216 (3990 MB) | 15c1d6981f855854 | 164b0e6b7e068858 |
| 90 + `PF_WCACHE=0` / `PF_LOAN_KEEP=0` / both | the same | 15c1d6981f855854 | 164b0e6b7e068858 (equal to 90) |

Findings: **chunk size is neutral; the layer-by-layer plan is not**, and each layered shape (chunk, group) has its own bits (two different layered plans of chunk 1024 differ from each
other and from chunked). Weight cache and loan keeping do not move bits. The layered plan is chosen when the loan limit (a share of the layers' expert slots, so the free VRAM at start)
suffices for `pf_layered_need`; at the default 92 slots it is not, at 122 to 126 slots (the kv-hot-head runs) it can be, and at the edge (125 against 126 slots) it flips:
that is the run-to-run difference of block 1b. Open: what in the layered order changes the numbers (same data, other order: layers 0 and 1 per chunk, from layer 2 layer-major).

## Plan of job `g4` (`gpu` queue job, `run3.sh g4`): is the layered prefill stable, and which switch moves its bits?

All with `TONA_PF_LOAN_PCT=75` (layered, chunk 2048, group 16384 at 16000; e74f008d80235004 / e6fbb8c34d8bac84 in g3). (1) stability: 3 repeats in one process and 2 in another:
expectation equal to g3's value in all 5 (a different value = a race in the layer buffers or their events, a real nondeterminism of the layered plan). (2) one repeat each with
`TONA_GEMMQ_T128_MIN=0` (always the 64-token expert tile), `TONA_PF_SPLIT_ROWS=0`, `TONA_PF_PAIR=0`, `TONA_PF_MIX=0`, `TONA_PF_UNFUSED=1`: those hashes equal to e74f... = the switch is not the cause;
a hash equal to the chunked reference 40bf... = the switch removes the layered/chunked difference. No expectation from a source for (2): the documented bit-exactness of these switches
(`2026-11-16-prefill-lever7.md` etc.) was shown on the Coder. Active tooling: as g3 plus the named variable. Duration about 7 min.

### Result g4 (`/var/tmp/claude/bolt-det/g4.out`)

`TONA_PF_LOAN_PCT=75` (layered, chunk 2048, group 16384): all 5 repeats over 2 processes give `e74f008d80235004` / `e6fbb8c34d8bac84`: **the layered prefill is itself deterministic** (no race in the layer buffers). `TONA_GEMMQ_T128_MIN=0`,
`TONA_PF_SPLIT_ROWS=0`, `TONA_PF_PAIR=0`, `TONA_PF_MIX=0`, `TONA_PF_UNFUSED=1`: the same hashes as without: none of them is the cause. Layered and chunked are two different, each stable, functions of the prompt, for
the same chunks (2048): the order of the work (layer-major) alone changes the bits. Reading found nothing (same kernels, same boundaries). So bisect by data.

## Plan of job `g5`: where do the two orders first differ? (`gpu` queue job, `run3.sh g5`)

New diagnostic `TONA_PF_HASH=1` (`PrefillRun::hash_residual`, `prefill_run.cpp`): after every (layer, chunk) step the stream is synchronized and an FNV-1a hash of the chunk's residual stream goes to stderr
as `pfhash pos <chunk start> layer <l>: <hash>`. Runs (prompt 6000, 3 chunks of 2048, `--repeat 1`): `h-chunked` (default plan), `h-l75` and `h-l90` (`TONA_PF_LOAN_PCT`, plan lines tell whether layered), `h-l75-b`
(second process: are the hashes stable per line?). Expectation: layer 0 and 1 and all of chunk 0 equal in all runs if the cause is state carried across chunks (Mamba state, KV); the first
differing line names the layer type (Mamba, attention, MoE) of the cause. The sync per step slows the prefill (about 3 x), irrelevant here. Rebuilt first as a `cpu` job; the run is `gpu`.
Decision: the first differing (pos, layer) picks the next, finer probe inside that block.

### Result g5 (`/var/tmp/claude/bolt-det/g5-*.err`, prompt 6000, 3 chunks)

Plan lines: `h-chunked`: chunk by chunk, chunk 2048; `h-l75`: layer by layer, chunk 2048, group 6144 (and `h-l75-b`, a second process: all 264 hashes equal, the layered run is reproducible line by line).
Residual hashes of the chunked and the layered run: **identical for layers 0 and 1 of every chunk, different from layer 2 on, at chunk 0 already** (258 of 264 lines differ; the 6 equal ones are layers 0 and 1
of the 3 chunks). Layer 2 is a Mamba-2 block (NVFP4 `ssm_in`), the same type as layer 0, which is equal. The input of layer 2 of chunk 0 is the same residual (layer 1 equal). What differs for layer 2:
in the layered order the layers 0 and 1 of chunks 1 and 2 run before layer 2 of chunk 0 (`PrefillRun::layer_order`), and the norm of layer 1's output, which feeds layer 2, is written
at the end of layer 1 into `B.mixed16`, a buffer shared by all chunks (`prefill_layer.cpp` `body`: `launch_res_norm16(... res.norm[l + 1], B.mixed16 ...)`). Hypothesis (from reading, unverified until g6): at layer 2 of chunk 0 `B.mixed16`
holds the norm of the last chunk, not chunk 0's. If true the layered plan computes the wrong input in layer 2 for every chunk but the last, and the "difference between plans" is a bug, not rounding.
The Qwen-35 sequence (`arch/qwen35moe/prefill_layer.cpp`) has the same structure.

## Plan of job `g6`: how large is the difference between the layered and the chunked prefill?

Two runs, prompt 6000, `--logits-out` for the first decode pass: default (chunked) and `TONA_PF_LOAN_PCT=75` (layered). Then numpy: top-1, relative difference, KL(chunked || layered). Expectation under the
hypothesis: large (relative L2 of the logits 0.1 or more, KL above 0.1); if only rounding: KL about 1e-3 (the chunked-vs-sequential level of `2026-11-19-bolt-opt2.md`). Decision: large = fix the data flow
of the layered order for Bolt (norm recomputed per chunk at the first layered layer), after which the layered plan must equal the chunked one bit for bit; small = look for the source of rounding.

### Result g6 (`lg-chunked.bin`, `lg-l75.bin`, prompt 6000, first decode logits)

Chunked against layered (`TONA_PF_LOAN_PCT=75`): top-1 **10165 against 1032**, relative L2 of the logits **5.76**, largest absolute difference 65.5, **KL(chunked || layered) = 11.55**. This is not rounding: **the
layer-by-layer prefill of Bolt computes a wrong network** (the hypothesis of g5 holds in effect). Cause: every layer of the layered order (one layer for all chunks of a group, then the next layer) reads its input norm
from `B.mixed16`, which the previous step of the same layer wrote for another chunk; each layer reads the norm of the chunk that ran last (the last chunk of the group), not its own. The per-chunk residual streams `Rp[j]` are right,
the shared norm output is not. The same data flow is in the Ornith sequence (`qwen35moe/prefill_layer.cpp`); the Coder's sequence closes an open combine and norms at the start of the layer, so it is not affected.
Why nobody saw it: at the default slot count the Bolt loan limit (2.3 GB) is below `pf_layered_need`, so Bolt prompts always run chunk by chunk; with more free VRAM (122 to 126 slots) the plan flips to layered at 16K
(the kv-hot-head runs), and the output changes with the slot count, i.e. with the free VRAM of the start, i.e. from run to run. That is the instability of block 1b.

Fix (this branch): `PrefillSequence::resume(run, l)`, called by `PrefillRun::step` whenever the previous step was not layer l-1 of the same chunk; Bolt and Ornith remake the norm of the layer input from the chunk's
residual stream (`launch_res_norm16` without `y`: the same arithmetic as the fused kernel, so the bits are the ones of the chunk-by-chunk run). Cost: one norm kernel per layer and chunk in layered plans only.

## Plan of jobs `g7a` (Bolt), `g7b` (Ornith, new build), `g7c` (Ornith, build before the fix = `work/build/det-old`)

Rebuilt first (`cpu` job). g7a (`run4.sh`, Bolt, `gpu` kind): prompt 6000 with `TONA_PF_HASH=1`: default (chunked), `LOAN_PCT=75` (layered, chunk 2048), `LOAN_PCT=90` (layered, chunk 3072), logits of the first decode pass dumped;
prompt 16000: default, 75, 90 and `TONA_PREFILL_CHUNK=1024` (layered, group 4096/8192 in g2/g3). Expectation: **all layers' residual hashes of the layered runs equal the chunked run's** (same chunk 2048 lines; for chunk 3072 the first-logit
hash), the 16000 runs all `40bf709329746081` / `7e81a64bb9cd5814`, the 6000 chunked logits `b10b2d13ef76a4d5` as before the fix (the chunked path is untouched). Tolerance none; a layered run that differs after the fix
means the norm recomputation is not bit-equal or something else is shared. g7b/g7c (Ornith Q4_K_M, ids from `tona-tokenize` of the prose corpus, shipped Ornith profile): prompt 6000, default, `LOAN_PCT=75`, `LOAN_PCT=90`, first-pass logits dumped:
expectation by the same reading: g7c (old build) layered differs from chunked, large KL if the plan is layered (the log says), g7b all equal to the chunked run. If Ornith is not layered at these settings, the log
line says so and the Ornith part is open. Duration about 15 min for the three jobs.

### Result g7a/g7b/g7c

**Bolt with the fix (g7a):** layered plans (`LOAN_PCT=75`: chunk 2048; `90`: chunk 3072; `TONA_PREFILL_CHUNK=1024`: layered) now give exactly the chunked results: prompt 6000 first logits `b10b2d13ef76a4d5` and ids `7b54e90324581d1d` in all three; prompt 16000
`7e81a64bb9cd5814` / `40bf709329746081` in all four (chunked, 75, 90, chunk 1024). The residual hashes: layered `LOAN_PCT=75` equals chunked in all 264 lines (layer x chunk; before the fix 258 differed); chunk 3072 (`LOAN_PCT=90`) has other chunk boundaries, so its lines cannot be compared by position, its final logits equal bitwise.
Before the fix the same layered runs gave `ab03bed62d82547e`, `e9b5ac525731854b` (6000) and `e6fbb8c34d8bac84`, `164b0e6b7e068858` (16000). So the norm recomputed from the residual is bit-equal to the fused one and the layered Bolt prefill is bit-equal to the chunked one,
whatever chunk and group size: the plan no longer defines the output.
**Ornith (g7b new, g7c old build), prompt 6000:** all six runs equal (`e84f67af2b77428d`), but the log shows `layer by layer, chunk 6144, group 12288`: one chunk, which never leaves the chunk-major order, so this tests nothing about the bug. Next: job g8, prompt 16000 with
chunks that really are several (default plan, `TONA_PF_LAYERED=0` as the chunked reference, `LOAN_PCT=75 LCHUNK=2048`), old and new build, first logits dumped.

## Plan of jobs `g8old`, `g8new` (Ornith, 16000 prompt, `gpu` kind, `run4.sh`)

Is Ornith's layered prefill affected too (same data flow)? Prompt 16000 (several chunks in any plan), three configs: default, `TONA_PF_LAYERED=0` (chunked reference), `TONA_PF_LOAN_PCT=75 TONA_PF_LCHUNK=2048` (layered with chunk 2048 if it fits); old build (`work/build/det-old`) and new.
Expectation: if the plan line says "layer by layer" with more than one chunk, the old build differs from its chunked reference (KL far above 1e-3) and the new one equals it bit for bit; if the plan is chunk by chunk everywhere, nothing is learned and
the Ornith side stays unverified (stated in the report). Duration about 6 min.

### Result g8 (Ornith Q4_K_M, prompt 16000, first decode logits dumped; `g8old.out`, `g8new.out`, `g8-*.bin`)

Plan lines: default `layer by layer, chunk 6144, group 18432` (three chunks, weight cache), `LOAN_PCT=75 LCHUNK=2048`: `layer by layer, chunk 2048, group 16384`. So **Ornith's default prefill of a 16K prompt is layered with several chunks**.

| Build | Config | first logits | ids | against chunked (`TONA_PF_LAYERED=0`) |
|---|---|---|---|---|
| before the fix | chunked | adb0807b8006609d | a89c0ba61ce55df3 | reference |
| before the fix | default (layered, chunk 6144) | 2a07bfb98dc1e089 | 725dae816a7f5d2b | top-1 41253 against 1754, relative L2 1.015, **KL 12.97** |
| before the fix | layered chunk 2048 | 68cc15f7aff7f8f1 | 7f641d3abfac8313 | top-1 41253 against 208306, relative L2 0.991, **KL 15.77** |
| with the fix | chunked | adb0807b8006609d | a89c0ba61ce55df3 | equal |
| with the fix | default (layered) | adb0807b8006609d | a89c0ba61ce55df3 | **bit-equal** (KL 0) |
| with the fix | layered chunk 2048 | adb0807b8006609d | a89c0ba61ce55df3 | **bit-equal** (KL 0) |

(The 6000-token Ornith runs of g7b/g7c were one chunk, so they could not show it.) **Ornith's layer-by-layer prefill had the same defect**: every prompt that the default plan prefilled in more than one chunk (here 16K) was computed
with the wrong input norm in all layers but for the last chunk. After the fix it is bit-equal to the chunk-by-chunk prefill. This changes the default numbers of Ornith for long prompts (to the validated chunk-by-chunk ones).

## Plan of the speed runs (`exclusive` queue jobs, `scripts/ab.py`, A = build of base `0abbbed` (`work/build/det-base`), B = this branch (`work/build/det`))

Question: does the fix cost speed in the default path? Bolt (default plan chunk by chunk, so the code path is identical: expectation 0 +- 1 %) and Ornith (the default plan is layered above 6144 tokens, the fix adds one norm
kernel per layer and chunk: 40 layers x 3 chunks at 16K, a norm over 6144 tokens x 2048 floats = 25 MB read, about 0.1 ms each, expectation prefill -0.3 to 0 %: 12 ms of a 6 s prefill). Jobs: (1) Bolt `--workload prose,longdoc`
(prose 2K/16K, longdoc 64K) ; (2) Bolt code 2K/16K/64K as `--workload server --targets 2000:200,16000:200,64000:100`; (3) Ornith `--workload server --targets 2000:200,16000:200,64000:100`.
Rounds: 3, alternating AB/BA, `--fixed-rounds`. Measured with `--decode` and prefill named. Expectation source: the code difference is the extra norm kernel in layered plans only. Active tooling: none (no
TONA_ variable). Tolerance: +-1 % for Bolt (noise of A/A: `2026-11-19-bolt-opt2.md`, below 0.25 %), +-1 % Ornith prefill. The Ornith decode must be equal (the decode path is untouched).

## Result of the speed runs (`work/ab/bdet-sp{1,2,3}.md`, 3 rounds AB/BA each, A = base `0abbbed`, B = this branch; quiet checks before and after each job: warning for 2.8 GiB VRAM held by the desktop only, `/var/tmp/claude/bolt-det/quiet-*-sp*.txt`)

| Model, workload | prefill B vs A | decode B vs A | Note |
|---|---|---|---|
| Bolt prose 2K / 16K, longdoc 64K (sp1) | -0.00 / -0.03 / -0.02 % | +0.19 / +0.20 / +0.05 % | all CIs span 0 (about +-0.9 %); SHAs equal in every start |
| Bolt code 2K / 16K / 64K (sp2) | -0.01 / -0.01 / -0.03 % | +0.11 / -0.01 / +0.01 % | all CIs span 0; SHAs equal |
| Ornith code 2K (sp3) | -0.24 % [-1.35, +0.88] | +0.06 % | one chunk, no change |
| Ornith code 16K | **-1.17 % [-2.01, -0.33]** (4124 to 4077 tok/s) | -42.7 % (text, see below) | layered: 3 chunks of 6144 |
| Ornith code 64K | -0.21 % [-1.06, +0.63] | -21.4 % (text, see below) | layered |

Bolt: no change in speed, as expected (the code path of the default plan is untouched). Ornith: the added norm kernel costs 0.2 to 1.2 % of the prefill in layered plans (the 1.2 % is a little above the estimate of 0.3 %: 12 norm
launches... of 6144 x 2048 floats each at the memory bandwidth; within the +-1.5 % that was planned for the whole result, a finding to keep). The Ornith decode differences at 16K and 64K are **not speed**: the texts differ (they must, the prefill was
wrong before), and the decode rate follows the text through the drafter: tokens per pass at 16K 4.88 (A: the text after the broken prefill is repetitive, the drafts are accepted) against 2.21 (B), at 64K 1.90 against 2.08 (so at 64K B is *faster per pass*),
2K equal (2.38 both, same SHA). Pass time is not the quantity that changed. Nothing was slower in the decode path (untouched code). The Coder was not built differently (no change in `qwen4exp`): goldens below.

## Gates (the final build, after the refactor into `resume_input` and the formatting; `/var/tmp/claude/bolt-det/goldens.out`, `gates.log`)

`just check` (pre-commit hooks, openmp, clang-tidy of the changed files): green. `ctest -L unit` (71 tests, among them the new `test_prefill_order`): green. `just test-py`: 681 passed, 1 skipped (torch). Coder goldens on this build: decode gpu plan, decode cpu
plan (csv, rust, explain), prefill (long), prefill dflash (long), server (all cases): **identical**. `test_gpu_fattn`, `test_gpu_attn_wmma`, `test_gpu_mamba`, `test_gpu_moe_latent`, `test_gpu_bolt_quant`: exit 0.
`scripts/bolt-determinism.sh` (Bolt, 6000 tokens, 5 plans x 2 repeats): all 10 runs logits `b10b2d13ef76a4d5`, ids equal; on this run only `layered90` was really layered (the free VRAM of the moment gave a smaller loan limit: the
plan follows the free VRAM, which is the reason the bug showed up as run-to-run differences; g7a covered the layered 75 and chunk-1024 plans with `TONA_PF_HASH`). Ornith, 16000 tokens: all 10 runs equal, 8 of them layer by layer.

## What stays open

- The decode of Bolt was never the source (CPU = GPU experts bit for bit, widths 1 to 8 equal, cache state and KV placement neutral); only the prefill plan was. Ornith's plan still follows free VRAM (now without effect on the output).
- Plan independence beyond chunk by chunk against layer by layer: chunk sizes 1024 to 8192 are bit-neutral (g2, g7a). Not tested: staged KV (`TONA_KV_STAGE`, layered plan with a cache in RAM) for Bolt at 64K+ and Ornith with the weight cache on beyond what g7b/g8 ran (weight cache was on in the Ornith default plan and equal).
- The Ornith layered prefill costs 0.2 to 1.2 % more prefill time (one norm per layer and chunk). A per-chunk norm buffer would avoid it at 6144 x 2048 x 2 bytes per chunk of extra workspace; not done.
- Earlier Ornith or Bolt measurements of prompts that were prefilled layer by layer in several chunks (Ornith above 6144 tokens in one piece, e.g. long ab.py prompts; the quality and agentic runs that sent long fresh prompts) were computed with the wrong network; their decode speeds (acceptance) and quality numbers should be read with that in mind. Prompts that the server fed in pieces below the chunk size were not affected.
