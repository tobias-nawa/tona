# Multi-architecture refactor R1 to R3: production A/B of the server hot path (2026-11-15)

RX 9060 XT (gfx1200), Zen 5. Branch `142-r3-server` (R1 model layer, R2 registry, R3 server neutralization, cumulative) against main 64a8204
(`work/build/base`, built by R0). Design: `docs/architecture/model-architectures.md`; this is the check the coordinator added for tickets
that change per-token or per-pass server code.

## Plan (written before the run)

**Question.** Did the neutralization of the server (the generator reads vocabulary size and model width from `ModelBase::dims()` instead of
`Model::hp()`, the chat format and the output parser are virtual objects resolved once per request, the thinking-budget texts come from the
format) change production decode speed?

**Why this should not change anything.** Checks C1 (all 217 kernels identical) and C2 (the op programs byte-equal, R1) hold, so the GPU work
per pass is the same. The host changes in the per-pass path are: `m_.dims().vocab` instead of `m_.hp().vocab` in `verify.cpp`,
`prefill.cpp`, `generator.cpp` (one load from another member offset), and one virtual call per streamed piece (`OutputParser::feed`) in the
HTTP route instead of a direct call. No virtual call goes through `ModelBase` in the token loop.

**Scope.** `scripts/ab.py --workload set` (14 requests, 11 code and 3 text, MTP drafter, `--ctx 32768`), A = `base` (main), B = `142-r3`,
2 rounds (alternating order, discarded warm-up start of A), `--load-log on`. Rates: decode tok/s of the code and text requests (the rate the
workload measures; prefill is not a metric of `set`). Expected duration: about 14 requests x 2 arms x 2 rounds plus 5 server starts of the
55 GB model, roughly 40 minutes.

**Expectation (with tolerance).** B against A: decode within -0.5 % to +0.5 % (source: `bench/noise.toml`, decode-code sd of the paired
difference 0.065 % with df 2, so unused by ab.py for the decision, and the mechanism: identical machine code and programs, one offset differs).
Decision: a loss beyond the noise (below -0.5 % in both rounds) is not merged; find the cost in the host loop, fix, repeat. A result inside the
expectation ends the check.

**Active tooling.** `--load-log on` (our load and the system's per server start, sums only). No `TONA_*` variable besides what `ab.py`
sets (expert profile, `TONA_DIAG` as the tool's default); no `TONA_DETERMINISTIC` (it caps drafts at 3). Test conditions and quiet check
before and after the run are in the next section; the machine is shared through `scripts/gpu_queue.py`, no build ran during the measurement.

## Test conditions

Before the run (`tools/hw_report.py`):

- CPU: AMD Ryzen 5 9600X 6-Core Processor, 6 cores / 12 threads
- GPU: AMD 1002:13c0, gfx1036, 2.0 GiB VRAM, 0.02 GiB held by others before the run, 0 % busy before the run, integrated
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 1.14 GiB held by others before the run, 2 % busy before the run
- RAM: 60.5 GiB, 54.8 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.2.15-200.fc44.x86_64 (x86_64)
- Driver and toolkit: rocm 10.1.0
- TONA_* variables: `TONA_GPU_ARCH=gfx1200` `TONA_GPU_LOCK_HELD=1` `TONA_HIPCXX=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/lib/llvm/bin/amdclang++` `TONA_HIPFLAGS=-x hip --offload-arch=gfx1200 -isystem <repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/include` `TONA_HIPLDFLAGS=-L<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/lib -lamdhip64 -Wl,-rpath,<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/lib`
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 2.2, 3.92, 4.56
- CPU use by other programs over 0.5 s: 0.2 % of all threads, swap in use 1.63 GiB
- sandbox: no
- Commit: `4caacd0` (clean tree)
- Model: `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf`, 27.6 GiB (recognized by name and size: toolchain/models.toml)

Quiet check before the run (warning; the only finding is 1.14 GiB of VRAM held by others, CPU use by other programs was 0.2 % of all threads; measured
anyway, the A/B is paired with alternating order, and the available resources are those of the test conditions above):

- Quiet check (2026-11-15 10:26:27): warning
  - warning: 1.14 GiB of 15.9 GiB VRAM are held by others

After the run:

- CPU: AMD Ryzen 5 9600X 6-Core Processor, 6 cores / 12 threads
- GPU: AMD 1002:13c0, gfx1036, 2.0 GiB VRAM, 0.02 GiB held by others before the run, 0 % busy before the run, integrated
- GPU: AMD Radeon RX 9060 XT, gfx1200, 15.9 GiB VRAM, 1.14 GiB held by others before the run, 0 % busy before the run
- RAM: 60.5 GiB, 54.8 GiB available before the run
- OS: Fedora Linux 44 (Workstation Edition), kernel 7.2.15-200.fc44.x86_64 (x86_64)
- Driver and toolkit: rocm 10.1.0
- TONA_* variables: `TONA_GPU_ARCH=gfx1200` `TONA_GPU_LOCK_HELD=1` `TONA_HIPCXX=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/lib/llvm/bin/amdclang++` `TONA_HIPFLAGS=-x hip --offload-arch=gfx1200 -isystem <repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/include` `TONA_HIPLDFLAGS=-L<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/lib -lamdhip64 -Wl,-rpath,<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel/lib`
- cgroup limits of this shell: none set
- Load average (1, 5, 15 min): 1.82, 2.32, 3.6
- CPU use by other programs over 0.5 s: 0.5 % of all threads, swap in use 1.63 GiB
- sandbox: no
- Commit: `4caacd0` (clean tree)
- Model: `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf`, 27.6 GiB (recognized by name and size: toolchain/models.toml)

Quiet check after the run (`--compare` against the one before):

- Quiet check (2026-11-15 10:32:30): warning
  - warning: 1.14 GiB of 15.9 GiB VRAM are held by others
- Quiet check before and after the run: warning -> warning, unchanged

## Result (`scripts/ab.py` report, unchanged)

A: `base` env -; B: `142-r3` env -. Alpha 0.05, margin 0.5 %, rounds 2 of at most 2, stop: 2 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| decode-code: Decode tok/s (bench-set code) | 93.7 / 93.4 | 93.6 / 93.7 | 93.6 (93.4-93.7) | 93.7 (93.6-93.7) | +0.11 % [-0.74, +0.96] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 68.2 / 67.8 | 68.0 / 68.0 | 68.0 (67.8-68.2) | 68.0 (68.0-68.0) | +0.00 % [-0.88, +0.88] | open (undecided at the last round) |

| Round | Order | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | -0.11 % | -0.29 % | 145 |
| 2 | BA | +0.32 % | +0.29 % | 145 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| decode-code | +0.11 % | -0.5 to +0.5 % | within the expectation |
| decode-text | +0.00 % | -0.5 to +0.5 % | within the expectation |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 162.4 | 11.0 | 3.8 | 0.0 | 0.0 | 3.1 | 28802.6 | 0.1 | 0.6 | 47 | 0.0 | 0.0 | - | 0.0 | - | - | 20.3 |
| 2 | A | 162.7 | 10.4 | 3.8 | 0.0 | 0.0 | 4.0 | 28541.2 | 0.3 | 0.0 | 47 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.5 |
| 3 | B | 163.0 | 10.3 | 3.6 | 0.0 | 0.0 | 3.6 | 28545.1 | 0.0 | 0.0 | 47 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.5 |
| 4 | B | 162.5 | 10.2 | 3.5 | 0.0 | 0.0 | 3.7 | 28535.5 | 0.0 | 0.0 | 47 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.5 |
| 5 | A | 162.1 | 10.3 | 3.6 | 0.0 | 0.0 | 3.7 | 28604.7 | 0.0 | 0.0 | 47 | 0.0 | 0.0 | - | 0.0 | - | 0.1 | 20.5 |


Runtime: 6.0 min for 2 rounds (2.4 min per round); a fixed 3 rounds would take 7.2 min (+2.4 min saved).

## Reading against the plan

Both rates are inside the expectation (-0.5 to +0.5 %): decode code +0.11 %, decode text +0.00 % (B against A, two rounds, alternating order).
ab.py calls both "open" because two rounds with a spread of 0.3 % cannot show equality within the margin of 0.5 % at its error budget; no round
shows a loss beyond the expectation (round 1, order AB: -0.11 % and -0.29 %; round 2, order BA: +0.32 % and +0.29 %; the sign changes with the order, a
start-order effect is unchecked and would need more rounds to separate from noise). The plan's decision rule (a loss beyond the
noise in both rounds is not merged) does not trigger. The A/B is a guard of the refactor and makes no speed claim, so it has no row in BENCHMARKS.md.
