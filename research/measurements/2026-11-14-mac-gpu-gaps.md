# Mac: where the GPU waits between the command buffers of decode

Issue #108. In the server A/Bs the GPU share of our process is 94 to 95 % in fast starts
([2026-11-14-mac-ple-copy.md](2026-11-14-mac-ple-copy.md)); a decode cycle (target pass plus the MTP head's drafts)
takes about 100 ms ([2026-11-14-mac-ple-prefetch.md](2026-11-14-mac-ple-prefetch.md)), so the GPU waits about 5 to
6 ms per cycle. This probe splits that wait by cause before anything is changed.

## Plan (written before the run)

**Probe switch.** `TONA_GPU_GAPS` (`engine/backend/metal/gpu_gaps.h`, Metal only; empty or 0 = off, nothing recorded;
1 = one stderr line per 1000 target passes, N > 1 = per N). Every decode command buffer is recorded with the host times
it was prepared from, committed and waited on, and its `GPUStartTime`/`GPUEndTime` (the host clock). The backend calls
whose command buffers are recorded: `forward` (target pass), `mtp_fill`, `mtp_step`, `commit` (the DeltaNet replay
after a partial accept). Each gap between two recorded command buffers splits into:

| Part | From, to | Where the code lives |
|---|---|---|
| wake | GPU end until the CPU returns from `waitUntilCompleted` | Metal (driver, wait) |
| backend | host work of the backend after the wait and before the next buffer's inputs (MTP picks, h copies, NaN count) | Metal backend (`runner*.mm`); the same jobs exist in every backend |
| outside | between two backend calls: generator, sampling, acceptance, draft decisions | shared host code (`engine/server/`) |
| inputs | writing the next buffer's inputs and encoding it (n-gram copy, history copy, MTP embedding) | Metal backend; the n-gram copy helpers are shared |
| submit | commit until the GPU starts the buffer | Metal (driver, queue) |

Gaps over 1 s (between requests) and gaps across unrecorded work (prefill, state copies) are left out. A smoke run
(`tona-check` with the MTP golden file, target passes only, not a result): busy 43 to 44 ms, idle 1.0 ms per pass
(2.2 %): wake 0.16, inputs 0.67, submit 0.15 ms.

**Prior art.** llama.cpp (`ggml/src/ggml-metal/ggml-metal-context.m`, master, read 2026-11-14): the calling thread
encodes and commits the first nodes of a graph at once, and while the GPU runs them `n_cb` threads encode the rest into
their own command buffers ("optimal values for n_cb are 1 or 2" on M1 Pro and M2 Ultra); the graph is left to compute
asynchronously, the CPU waits only in `synchronize` on the last buffer. MLX (`mlx/backend/metal/device.cpp`, main):
command buffers are committed when they hold 40 to 50 ops or MB and a new one is opened at once, without a CPU wait;
order between encoders by `MTLFence`, between streams by events (`encodeSignalEvent`/`encodeWait`); completion
handlers instead of waits; the CPU blocks only in `synchronize()`. Our decode waits after every command buffer
(`waitUntilCompleted`), because the host reads the logits (sampling, MTP picks) before the next one can be written.

**Run.** `scripts/ab.py --workload set`, DM-T + MTP-164, A/A with `TONA_GPU_GAPS=500` in both arms (a line per 500
target passes), `--fixed-rounds --max-rounds 2`, `--load-log on`; about 25 min. Not a speed comparison: the starts
supply the gap lines.

**Expectation.** Idle 4 to 7 ms per target pass (cycle), 4 to 7 % of the GPU's time, roughly: wake and submit 0.3 ms per
command buffer at 4 to 6 buffers per cycle (target, commit, fill, steps) = 1.2 to 2 ms; inputs 0.7 to 1.5 ms (the target
pass as in the smoke run, the MTP head's embedding); backend 0.5 to 1.5 ms (picks over the head's logits); outside 1 to
2 ms (sampling and acceptance in the generator). If one part is ≥ 2 ms it is the first to attack.

**Active tooling.** `TONA_GPU_GAPS=500` in both arms, the load log; defaults otherwise (`TONA_PLE_PREFETCH` 2).

## Result: 3.5 to 4.9 ms idle per cycle, spread over five parts, none ≥ 2 ms

Conditions: before ok, after "do not measure" (GPU 97 % busy, the run's own server; 5.0 % other CPU) and 20 s later ok;
log in [2026-11-14-mac-gpu-gaps.log](2026-11-14-mac-gpu-gaps.log). Report and load table:
[2026-11-14-mac-gpu-gaps-server-aa.md](2026-11-14-mac-gpu-gaps-server-aa.md) (A/A code +0.26 % [−0.55, +1.07], text
+0.19 %, within ±3 %). 9 server starts, 2 lines each (500 target passes per line), 18 lines; ms per target pass (one
decode cycle: target pass, commit, fill, steps):

| | Median | Range |
|---|---|---|
| GPU busy | 94.9 | 92.2 to 113.1 |
| GPU idle | 4.0 (4.1 %) | 3.5 to 5.2 (3.5 to 4.9 %) |
| wake (GPU end to the CPU back from the wait) | 0.50 | 0.49 to 0.52 |
| backend (host work of the backend after a wait) | 1.06 | 0.80 to 1.51 |
| outside (shared host code between backend calls) | 0.65 | 0.61 to 0.72 |
| inputs (writing and encoding the next buffer) | 1.25 | 1.02 to 1.46 |
| submit (commit to GPU start) | 0.62 | 0.56 to 1.40 |

Idle by step, median ms per cycle: target→fill 0.91, step→target 0.87, step→step 0.81, fill→target 0.70, fill→step
0.38, target→commit 0.26, commit→fill 0.13. The first line of start 1 (busy 113 ms, submit 1.40) is the cold start.

**Finding: inside the expectation (4 to 7 ms), at its low end, and no part reaches 2 ms.** The waits are spread over
about five command buffers per cycle at 0.1 to 0.3 ms each per part. The whole idle time bounds the gain of any
overlap: about 4 % of decode. The GPU share of the load log (93 to 95 %) and busy plus idle (99 ms) agree with the pass
period measured in [2026-11-14-mac-ple-prefetch.md](2026-11-14-mac-ple-prefetch.md).

**Where each part lives (for the PC):**

| Part | ms | Code | Transfer to the PC |
|---|---|---|---|
| outside | 0.65 | shared: generator, sampling, acceptance, draft decisions (`engine/server/`) | the same code runs there; worth measuring on the PC |
| backend | 1.06 | Metal backend host work (MTP picks over the head's rows, h copies, NaN count; `runner*.mm`) | the PC backend has its own; same jobs, other code |
| inputs | 1.25 | Metal: input writes and encoding (`pass.mm`); the n-gram copy in it is shared (`backend/ple_prefetch.h`) | encoding is Metal only; the n-gram copy is already ported |
| wake, submit | 1.12 | Metal driver: a CPU wait after every command buffer | none: the PC's megakernel waits inside the kernel |

**Candidates (not started; the coordinator chooses):**

1. MTP drafting without the CPU between the head's passes: the pick (argmax over the head's rows) on the GPU and the
   steps chained in one command buffer removes fill→step and step→step (1.2 ms per cycle) and their backend picks:
   about +1.2 % decode; bit identity of the picks to be shown.
2. Encoding cost of the inputs (1.25 ms): split it first (n-gram copy 0.26 ms of it, the rest encoding and history
   copies); reusable encodings (indirect command buffers) or a smaller history copy, about +0.5 to +1 %.
3. Wake (0.5 ms): a shared event with a short spin instead of `waitUntilCompleted`, at most +0.3 to +0.5 %.

## Decisions on the candidates (coordinator, 2026-11-14)

**Candidate 1 as written (drafting without the CPU between head passes): rejected.** The chain has a host decision
after every step: `Generator::chain_push` (`engine/server/draft.cpp`) turns the pick's q into a calibrated p
(`mcal.predict`) and stops when the next draft would no longer raise the expected rate. Chaining steps on the GPU ahead
of that decision runs one head pass in vain at the end of every chain (several ms) to save gaps of about 0.27 ms; moving
the decision onto the GPU would duplicate the generator's calibration and cost tables in kernels.

**What the "backend" part is.** `Runner::mtp_pick` (`runner_mtp.mm`, Metal backend code, not shared): a host scan over
the 49,681 rows of the truncated head (41,489 listed, 8192 dynamic) for the argmax and the sum of `exp(l - max)` in
double. Micro probe of the same loop: 321 to 337 us per pick; about 3 picks per cycle are most of the 1.06 ms.

**1a (the host scan on several threads, bit-identical): rejected.** Micro probe, 400 picks per row, NaN and tie cases
included, token and q bit-identical in all (0 mismatches):

| Threads | us per pick |
|---|---|
| serial (today) | 321 to 337 |
| 1 / 2 / 4 / 6 / 8 | 372 to 378 / 264 to 280 / 206 to 225 / 208 to 228 / 211 to 244 |

Bound: the sum in double in row order (51 us, not parallel if bit-identical), exp in double (about 5 ns per row, 71 us
on 4 threads), 12 us per parallel region. Gain about +0.4 % decode, and libomp's threads spin after each region
(`KMP_BLOCKTIME`, 200 ms), against the rule "work first".

**1b chosen: the reference's GPU pick ported** ([2026-11-14-mac-mtp-gpu-pick.md](2026-11-14-mac-mtp-gpu-pick.md)). The
reference never had the host scan: `op_mtp_pick` (`engine/gpu/mega_op_mtp.h`) reduces the rows on the GPU in float with
`__expf`, and the host reads 4 floats (`Mega::pick`). So Metal's double q was never the reference's q either.

## The "inputs" part split (coordinator: item 2, a probe extension)

`TONA_GPU_GAPS` now also reports, of the inputs before each target pass (time from the start of `Runner::run` until
commit): the copy of the token history (`std::vector` from the slot's tokens plus the pass's), the n-gram rows
(`ple::rows_at`, the read-ahead hint, the row copy into the staging buffer) and the attention positions
(`write_positions`, the QSA cells); the rest of "inputs" is the encoding of the pass and the MTP head's own inputs.

### Plan (written before the run, 2026-11-15)

**Scope.** `scripts/ab.py --workload set,server --targets 8000:100 --ctx 32768`, DM-T + MTP-164, A/A on the build of
a3018cb plus the probe extension, 2 fixed rounds (about 30 min), `TONA_GPU_GAPS=500`, `--load-log on`, defaults
otherwise (GPU pick and read-ahead level 2 on).
**Expectation.** inputs about 1.25 to 1.3 ms per cycle (the earlier runs); n-gram rows 0.25 to 0.35 ms (the PLE copy
with read-ahead, 0.24 to 0.31 ms in 2026-11-14-mac-ple-prefetch.md, plus row ids and hint); history under 0.05 ms
(8000 tokens are 32 KB); positions under 0.1 ms; the rest, encoding and MTP inputs, about 0.8 ms. A/A: speed equal
within the spread.
**Active tooling.** `TONA_GPU_GAPS=500` in both arms; the load log.

### Result: of 1.29 ms inputs, 0.26 ms n-gram rows, 1.02 ms encoding and MTP inputs, the rest nothing

Conditions: before ok; after "do not measure" (GPU 98 % busy, the run's own server), 20 s later ok; log
[2026-11-15-mac-inputs.log](2026-11-15-mac-inputs.log), report
[2026-11-15-mac-inputs-server-aa.md](2026-11-15-mac-inputs-server-aa.md). A/A inside its spread (decode code −0.13 %,
text −0.73 %, @8000 −0.51 %, all open), answers identical (SHA 727ff658ae742fa7).

Per cycle, medians over 9 starts (range), ms:

| Line | inputs | history | n-gram rows | positions | rest (encoding, MTP inputs) | idle total |
|---|---|---|---|---|---|---|
| code | 1.29 (1.26-1.41) | 0.00 | 0.26 (0.24-0.28) | 0.00 | 1.02 | 3.35 |
| text | 1.29 (1.26-1.47) | 0.00 | 0.25 (0.22-0.27) | 0.00 | 1.04 | 3.40 |
| 8000 | 1.13 (1.09-1.30) | 0.00 | 0.23 (0.21-0.28) | 0.00 | 0.88 | 2.58 |

(0.00: below 0.005 ms per cycle.) The other parts with the GPU pick on: wake 0.51, backend 0.35, outside 0.59 to 0.66,
submit 0.59 to 0.60 (code line).

**Reading against the plan.** inputs and n-gram rows as expected; history and positions below the expectation (they
cost nothing); the rest is about 1.0 ms, above the 0.8 expected: **encoding is now the largest single host part
between two command buffers** (about 1 % of the cycle). Not split yet: how much of it is the target pass's encoding
(about 40 layers of dispatches, `Pass::encode`) and how much the MTP passes' inputs (the embedding row dequantized and
copied `n_hc` times, the h copy, in `Runner::mtp_pass`).

**Candidates (not measured):**
1. The split of the rest into target encoding and MTP inputs (probe, minutes).
2. Encoding the next target pass while the GPU still runs the MTP head: buffers are read at execution, so tokens,
   rows and positions can be written after encoding; only the width is unknown before acceptance (indirect dispatch
   sizes or one encoding per width).
3. Indirect command buffers (`MTLIndirectCommandBuffer`): the pass encoded once and replayed, the width from a buffer.

**Transfer to the reference PC (coordinator): none.** The megakernel does not encode per pass; the PC's host time
between passes is a separate item there.

### Plan of the second inputs run: the rest split (written before the run, 2026-11-15)

The probe now also reports the MTP passes' inputs (embedding rows dequantized and copied, h copied, `Runner::mtp_pass`)
and splits the remaining encoding by the command buffer it precedes: target passes against the others (MTP head,
commit). **Scope:** as the first inputs run, A/A, 2 rounds, about 30 min, `TONA_GPU_GAPS=500`, `--load-log on`.
**Expectation:** MTP inputs + encoding target + encoding other = 1.02 ms (code line) within ±0.1; MTP inputs about
0.1 ms (about 3 head passes, each a dequantized row of 4096 floats copied `n_hc` times plus h, tens of us); target
encoding about 0.5 ms (one pass of about 40 layers and several hundred dispatches per cycle); other encoding about
0.4 ms (about 3 head passes). A/A within the spread.

### Result of the second inputs run: the target pass's encoding is 0.82 ms per cycle

Conditions: before ok; after "do not measure" (GPU 98 % busy, the run's own server), 20 s later ok; report
[2026-11-15-mac-inputs2-server-aa.md](2026-11-15-mac-inputs2-server-aa.md). A/A within its spread (decode code
+0.00 %, text +0.56 %, @8000 +0.72 %, all open), answers identical.

Per cycle, medians over 9 starts (range), ms:

| Line | inputs | n-gram rows | MTP inputs | encoding target | encoding other | history, positions |
|---|---|---|---|---|---|---|
| code | 1.28 (1.26-1.31) | 0.27 (0.25-0.30) | 0.02 | 0.82 (0.82-0.83) | 0.16 (0.16-0.17) | 0.00 |
| text | 1.28 (1.27-1.31) | 0.25 (0.23-0.28) | 0.02 | 0.84 (0.84-0.85) | 0.15 (0.15-0.16) | 0.00 |
| 8000 | 1.11 (1.09-1.13) | 0.23 (0.21-0.24) | 0.01 | 0.75 (0.74-0.75) | 0.11 (0.11-0.12) | 0.00 |

**Reading against the plan.** The sum matches (1.28 against 1.29 in the first run). The split differs from the
expectation: the target pass's encoding is 0.82 ms (expected 0.5), the MTP head's encoding only 0.16 (expected 0.4),
the head's inputs 0.02 (expected 0.1). One target pass per cycle, so **encoding one target pass costs about 0.8 ms of
host time while the GPU waits**: the largest host part between two command buffers.

**Prior art for taking it off the critical path.** llama.cpp's Metal backend encodes the first `max(64, 10 %)` nodes
of a graph on the calling thread, commits them, and encodes the rest while the GPU already runs them
(`ggml/src/ggml-metal/ggml-metal-context.m`, `n_main`: "the first n_main nodes are encoded and submitted for
processing directly by the calling thread; while these nodes are processing, we start n_cb threads to enqueue the
rest"). Apple's indirect command buffers reuse an encoding across submissions (developer.apple.com, "Indirect command
encoding"), with dispatch sizes from a buffer for a varying width; no measured compute numbers found.

**Next (coordinator: build what takes the time off the critical path):** the early commit first (the pass split into
two command buffers on the same queue after the first layers, the same kernels in the same order, so bit-identical),
then indirect command buffers or encoding ahead if encoding still shows.
