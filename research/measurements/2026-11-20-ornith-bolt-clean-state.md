# 2026-11-20: Ornith and Bolt, clean absolute state after the layered-prefill fix (#175)

Branch `worktree-<worktree>`, commit `936dfdc` (main with the fix of `59986a2`), build `work/build/clean175`
(this worktree; `tona-server`, `tona-bolt-bench`). Logs, scripts and ab.py output: `/var/tmp/claude/clean175/`.
No engine code changed in this ticket; every GPU job goes through `scripts/gpu_queue.py --kind exclusive`.

## Why

Before the fix (`2026-11-20-bolt-determinism.md`) every prompt that the default plan prefilled layer by layer in several
chunks (Ornith above about 6K tokens; Bolt only with more than about 120 expert slots free) ran with a wrong input norm.
The decode tok/s and the tokens per pass after long prompts in the Ornith measurements of today came from degenerate
text (16K: 4.88 tokens per pass instead of 2.21). The A/B ratios of those runs hold (both arms computed the same), the
absolute decode numbers do not. This note measures the absolute state once, on the fixed build.

## Test conditions

`python3 tools/hw_report.py` (08:18, before the first run, `/var/tmp/claude/clean175/hw.txt`):

```
- CPU: AMD Ryzen 9 9950X 16-Core Processor, 16 cores / 32 threads
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 3.4 GiB held by others before the run, 1 % busy before the run
- RAM: 60.5 GiB, 43.6 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.3.4-200.fc44.x86_64 (x86_64)
- Driver and toolkit: hip 7.16.26385-0000000
- TONA_* variables: none set
- Load average (1, 5, 15 min): 17.84, 7.54, 3.89 (a build of another contributor was running)
- Commit: `936dfdc` (clean tree)
```

`quiet_check.py` at planning time (08:18): do not measure (other programs 74.8 % of the CPU threads, load 19, 3.4 GiB
VRAM held). Each job repeats the check before and after (`quiet-before-*.txt`, `quiet-after-*.txt`); the exclusive queue
job starts only when no other GPU or CPU job runs. A warning at the start is reported with the result. VRAM held by
others and the slot count of every server start are taken from the ab.py logs.

## Plan

### Question and decision

What are decode and prefill rates (named, with tokens per pass) of Ornith and Bolt on the default setup, now that
long-prompt text is the real text? Result goes into the README ("Ornith and Bolt") and BENCHMARKS.md; the old
absolute numbers are replaced. No decision on code depends on it.

### Job 1: Ornith (`job-ornith.sh`)

Ornith-1.5-35B Q4_K_M, default server setup: ctx 262144, MTP head from the file (`--drafter none`), the shipped expert
profile, KV hot head at its default. `scripts/ab.py` as an A/A (same build in both arms: the spread comes for free,
and the arms are two starts per round).

- `server` workload, targets `2000:200,16000:200,64000:100,128000:100`, 2 to 3 rounds (noise.toml: decode sd 0.23 to
  0.42 %, prefill 0.02 to 0.09 %; ab.py stops when every metric is clear, minimum 2 rounds). Size: the four context
  points of the README; 200/100 decode tokens are 1 to 2 s of decode at 40 to 110 tok/s, enough for a stable rate (the
  earlier runs of today used 100 and 50 and spread under 1 %).
- `prose` workload (2K and 16K prose prompts, 400 decode tokens), same setup, 2 to 3 rounds.

Expectation (source: the hot-head file `2026-11-20-kv-hot-head.md`, Ornith decode with default budget, 94.4 to 143.7
tok/s at 16K, 35.1 to 76.1 at 64K, 14.6 to 41.8 at 128K; tokens per pass of those texts 4.88 at 16K, 1.90 at 64K, from
`2026-11-20-bolt-determinism.md`; clean tokens per pass 2.38 / 2.21 / 2.08 at 2K / 16K / 64K): the pass rate stays, the
tokens per pass change, so decode = old pass rate x clean tokens per pass.

| Rate | Expected | Tolerance |
|---|---|---|
| decode@2000 | 110 tok/s (tokens per pass 2.38, unchanged: one chunk) | +-8 % |
| decode@16000 | 65 tok/s (143.7 / 4.88 x 2.21 = 65; 2.21 tokens per pass) | +-20 % |
| decode@64000 | 83 tok/s (76.1 / 1.90 x 2.08; 2.08 tokens per pass) | +-20 % |
| decode@128000 | 41 tok/s (41.8 at unknown tokens per pass, assumed 1.9 to 2.1) | +-30 % |
| prefill@2000 / 16000 / 64000 / 128000 | 3,000 to 3,500 (first request) / 4,100 / 2,900 / 2,300 tok/s (hot-head and attention files: 4,077 at 16K, 2,965 at 64K; 2K and 128K from the same files) | +-10 % (2K first request +-20 %) |
| A against B (spread) | 0 | each decode within +-1 %, prefill within +-1 % (2K prefill +-3 %) |

A result outside the tolerance is a finding: it is read against the plan, not rounded into it.

### Job 2: Bolt (`job-bolt.sh`)

Bolt Nemotron 3.5 Bolt 74B-A9B 24 GB file, default server setup (ctx 262144, no drafter, shipped profile). A/A as above.

- `prose,longdoc` workloads (2K and 16K prose, 64K document, 400 and 200 decode tokens), 2 rounds. Size: ab.py needs 2
  rounds at least; the known spread of Bolt (`2026-11-20-bolt-determinism.md`: CIs about +-0.9 %) is small and the layered
  plan is rarely reached (122+ slots per layer), so 2 rounds suffice.
- `server` workload, targets `2000:200,16000:200,64000:100`, 2 rounds.

Expectation (source: `2026-11-19-bolt-opt2.md`, `2026-11-20-kv-hot-head.md`, `2026-11-19-prefill-attention.md`): Bolt
decode 41 tok/s at 2K (+-12 %), decode after 64K about 30 (KV hot head +22 % over the earlier 25; +-25 %); prefill about
1,300 to 1,500 tok/s at 2K and 16K, 1,757 at 64K (+-20 %). Bolt's text does not change by the fix when the plan is not
layered; with free slots it was layered, so tokens per pass may differ from the old rows (Bolt runs with no drafter:
1 token per pass). A against B within +-1 % (prefill +-2 %).

### Active tooling

Defaults only: no `TONA_*` variable, no profiler, no deterministic mode, `--load-log` off (Linux default). Exit 75 of
the desktop guard: requeue the same job and note the lost round here. A GPU fault stops everything and goes to the
coordinator.

## Results

Jobs (exclusive queue): `clean175-ornith` (08:22 to 08:43), `clean175-prose-bolt` (08:48 to 09:36). The first Ornith job ran its
`server` part and then stopped: `ab.py` looked for the prose corpus under this worktree's `work/corpus` (not present); the
prose workloads were run again in the second job with `--prose-corpus` pointing at the corpus of the main checkout (same file,
same tokens). Full ab.py reports (per round, expert cache counters, SHAs, tokens per pass):
`2026-11-20-ornith-bolt-clean-state-ab-{ornith-server,ornith-prose,bolt-prose,bolt-server}.md`.
Quiet checks (`/var/tmp/claude/clean175/quiet-*`): before and after every job only the warning "3.4 to 3.6 GiB of 15.9 GiB VRAM held
by others" (the desktop; once "swapping, 5 pages in 1 s" before the Bolt job). The load of the planning check (17.8) came from another
contributor's build; the exclusive job started after it.

**Deviation from the plan.** The `prose` and `longdoc` workloads set their own server context (32768 and 98304; docs/contribute/measurement.md);
`--ctx 262144` does not change that. So the prose rows (Ornith `prose`, Bolt `prose` and `longdoc`) are at those contexts with more
free VRAM (Ornith 104 slots per layer, Bolt 88 and 76), not at the service default. The `server` rows are at 262144 (Ornith 74 slots
per layer, Bolt 70; the desktop held 3.5 GiB, more than in earlier runs of today, so fewer slots than the 91 to 99 of those runs).

### Ornith (Q4_K_M, MTP head from the file, expert profile, KV hot head at default; A/A, 3 rounds, median of the rounds)

| Workload | Decode tok/s | Tokens per pass | Prefill tok/s | Spread of the rounds (decode / prefill) |
|---|---:|---:|---:|---|
| server 2K (code) | 114.3 | 2.38 | 2,582 | 114.3 to 114.4 / 2,580 to 2,586 |
| server 16K | 95.5 | 2.21 | 4,082 | 95.4 to 95.5 / 4,078 to 4,083 |
| server 64K | 64.7 | 2.08 | 2,964 | 64.7 / 2,963 to 2,965 |
| server 128K | 47.5 | 2.17 | 2,192 | 47.4 to 47.5 / 2,191 to 2,192 |
| prose 2K (ctx 32768) | 116.5 | 2.36 | 2,682 | 116.0 to 116.8 / 2,673 to 2,756 (first-request noise) |
| prose 16K (ctx 32768) | 101.2 | 2.20 | 4,089 | 101.2 / 4,089 |

A against B (same build): every decode within +-0.2 %, prefill within +-0.2 % except prefill at 2K (-0.03 % `server`, -0.97 % `prose`
with a round of -2.9 %: the first request of a start). SHAs equal and stable in A and B at all points.

### Bolt (24 GB file, no drafter so 1.00 tokens per pass, shipped profile; A/A, 2 rounds)

| Workload | Decode tok/s | Prefill tok/s | Spread (decode / prefill) |
|---|---:|---:|---|
| prose 2K (ctx 32768) | 40.3 | 1,417 | 40.3 to 40.4 / 1,416 to 1,418 |
| prose 16K (ctx 32768) | 39.4 | 1,615 | 39.3 to 39.4 / 1,615 to 1,616 |
| longdoc 64K (ctx 98304) | 36.7 | 1,524 | 36.7 / 1,523 to 1,524 |
| server 2K (code) | 40.1 | 1,415 | 40.1 / 1,414 to 1,416 |
| server 16K | 38.6 | 1,578 | 38.5 to 38.6 / 1,572 to 1,584 |
| server 64K | 36.3 | 1,429 | 36.3 / 1,425 to 1,434 |

A against B: decode within +-0.2 %, prefill within +-1.0 % (largest round: -1.03 % at 16K, -0.81 % at 64K, both in round 2). In
the `server` run the expert cache hits and misses of A and B differed in one start each (e.g. 2K: 34459/37541 against 34337/37663)
while the output SHAs were equal at every point. With 69 to 88 slots per layer the layered Bolt plan (122+ slots) was not reached
in these runs, so the fixed norm does not enter them.

### Against the plan

| Rate | Expected | Measured | Verdict |
|---|---|---|---|
| Ornith decode 2K | 110 +-8 % | 114.3 | within |
| Ornith decode 16K | 65 +-20 % | 95.5 | **outside (+47 %)** |
| Ornith decode 64K | 83 +-20 % | 64.7 | **outside (-22 %)** |
| Ornith decode 128K | 41 +-30 % | 47.5 | within (+16 %) |
| Ornith prefill 2K | 3,000 to 3,500 +-20 % | 2,582 | **outside** (the plan took the 2K figure from loosely related files; the 4,000-class rates at 16K are right) |
| Ornith prefill 16K / 64K / 128K | 4,100 / 2,900 / 2,300 +-10 % | 4,082 / 2,964 / 2,192 | within |
| Bolt decode 2K | 41 +-12 % | 40.1 to 40.3 | within |
| Bolt decode after 64K | about 30 +-25 % | 36.3 to 36.7 | outside (+21 to +22 %; the source was an older value, before the KV hot head) |
| Bolt prefill 2K / 16K | 1,300 to 1,500 +-20 % | 1,415 / 1,578 to 1,615 | within |
| Bolt prefill 64K | 1,757 +-20 % | 1,429 (code), 1,524 (document) | outside for code (-19 %, at the edge), within for the document |

Finding: the expected Ornith decode at 16K and 64K was built as "old pass rate x clean tokens per pass". That was wrong in both
directions. The pass rate behind the old 16K figure (143.7 tok/s / 4.88 = 29 passes/s) is much lower than the clean one (95.5 / 2.21 =
43 passes/s), and at 64K it is the other way round (old 76.1 / 1.90 = 40, clean 64.7 / 2.08 = 31). So the degenerate text changed not
only the acceptance but also the pass time (cause not examined here; the old runs also had other slot counts and desktop VRAM). The
old absolute decode numbers of Ornith after 16K and longer prompts are not to be reused, not even corrected by tokens per pass.
