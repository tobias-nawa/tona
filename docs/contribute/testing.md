# Tests and coverage

Golden checks, kernels and speed proofs: [verification.md](verification.md).

## Where tests run

| Test | Needs | CI | Locally |
|---|---|---|---|
| C++ unit tests (ctest label `unit`) | a compiler | `engine-cpu-tests` | `just test` |
| the same with ASan and UBSan | nothing | `engine-sanitizers` | `just sanitize` |
| Python tests of gates and tools (`tools/tests`) | pinned tools | `coverage` | `just test` |
| installer tests (`installer/tests`) | nothing | `installer` in `check.yml` | `just test` |
| tests with the model (`model_shapes`, `cpu_experts`, `chat_vs_jinja`, `chat_tokens`, tokenizer and dequantization against llama.cpp) | model, partly llama.cpp | no | `just test-gpu` |
| golden checks, `expert_bitexact`, speed | GPU and model | no: the maintainer sets `gpu-golden` | [verification.md](verification.md) |

Unit tests need no model, GPU or network and write only into a temporary directory. CI (`tests.yml`) runs them in a
pinned `fedora:44` container (GCC 16) with `-DTONA_MARCH=x86-64-v3`; the default is `znver5`. Other CMake options:
`TONA_SANITIZE`, `TONA_COVERAGE` (gcov at `-O0`). `just test` and `just sanitize` build in `work/build/cpu` and
`work/build/sanitize` with the CI flags (packages: `openssl-devel`, for sanitizers `libasan`, `libubsan`).

`TONA_DIAG=1` in the shell makes `toolchain/env.sh` write an allocation trace per GPU process and a GPU dump on a fault
into `work/diag` (no measurable cost; up to ~45 GB per fault; newest dump kept, traces 7 days). See
[environment.md](../use/environment.md).

## Writing a test

- Add the file to `engine/tests/`, register `tona_unit_test(<name> SOURCES ... LIBS ...)` in `engine/CMakeLists.txt`.
  `tests/test_support.h` has `EXPECT`, `TempDir` and a GGUF writer.
- Test behavior, not implementation.
- A value read once per process (`static`) needs one process per setting (see `request_options`, `env_budget`).
- Skipped on some machines: exit 77 and `SKIP_RETURN_CODE 77` (see `uring`).
- A bug you find: write the test, mark it `DISABLED` (ctest) or `xfail` (pytest), report it.
- Python: `tools/tests/test_*.py`; scripts with a hyphen are imported with `load_script` (conftest.py).

## Coverage

```sh
just coverage                  # build, unit tests, pytest, reports and floor check
just coverage origin/main      # also the changed lines against that base, as in a pull request
```

`tools/gates/coverage_gate.py` (CI too). Reports in `work/build/coverage/` (`cpp.xml`, `python.xml`, HTML). C++: line
coverage of `engine/` host code from the `unit` tests, without `engine/tests/`, `engine/third_party/`, `engine/gpu/`
and server code that needs ROCm. Python: `tools/` and `scripts/`.

| Gate | Rule |
|---|---|
| changed lines | at least 80 % of new and changed lines covered (diff-cover against the PR base) |
| total (ratchet) | C++ and Python line coverage not below `tools/gates/coverage-floor.json` |
| floor | a PR must not lower a value in `coverage-floor.json` |

At 1.5 points or more above the floor the job prints the value to commit: the CI summary value cut to one decimal minus
0.5. Never lower the floor; add the test or change less code.

## Speed comparison: `scripts/ab.py`

Use it for every speed claim (AGENTS.md rule 2). It stops when the result is statistically clear (method: header of
`scripts/ab_stats.py`).

```sh
source toolchain/env.sh
scripts/ab.py --a merged --b w88 --b-env TONA_DIAG=1 --workload short,long,set \
    --expect 'decode:-0.5..+0.5%' --expect 'prefill:-0.5..+0.5%' --expect-source 'A/A of 2026-11-02-ab-tool.md'
```

| Option | Meaning |
|---|---|
| `--a`, `--b` | builds (`work/build/<name>` or a directory) |
| `--env`, `--a-env`, `--b-env` | variables per arm; your shell's `TONA_*` are not passed on |
| `--workload` | names from [measurement.md](measurement.md#workloads) |
| `--expect`, `--expect-source` | mandatory for every metric: the plan (saved as `<out>-plan.md`) and the check "within / OUTSIDE the expectation: finding" |
| `--why-size` | mandatory: why these lengths answer the question |
| `--min-rounds` / `--max-rounds` | 2 / 6; order alternates AB, BA; no round is dropped |
| `--margin` | 0.5 % band for "equal" |
| `--alpha` | 0.05 error budget over the planned looks |
| `--primary decode` | only the named metrics decide the stop |
| `--no-warmup`, `--round-warmup auto\|on\|off` | the discarded start before round 1 / before each measured start (on by default on macOS) |
| `--no-noise`, `--noise FILE` | ignore / replace `bench/noise.toml` |
| `--no-expert-profile`, `--a-env TONA_EXPERT_PROFILE=none` | drop the shipped profile (`toolchain/profiles/<model id>.bin`) in both arms / one ([architecture/expert-tiers.md](../architecture/expert-tiers.md)) |
| `--port` | default 8097, never 8662 |
| `--load-log auto\|on` | per server start: CPU/GPU time of ours and the system's, page-ins, swap, backup (sums only; `scripts/ab_load.py`, Linux `scripts/ab_load_linux.py` ~33 ms of a core per sample; Linux `auto` stays off until an A/A shows no disturbance) |
| `--no-lock`, `TONA_GPU_LOCK_HELD=1` | when an outer script holds the lock |

- Every round starts a fresh `tona-server` per arm in `systemd-run --user --scope` with `MemoryMax` (RAM minus 6 GiB)
  and no swap, stopped by PID. macOS: no systemd, a guard thread ends the PID past the limit (`scripts/ab_rss.py`).
- `TONA_SPEC_OFF=1` in one arm gives the non-speculative baseline.
- `TONA_DETERMINISTIC=1` caps drafts at 3 (`engine/server/generator_env.h`): use it only to compare output SHAs, never
  for decode speed or drafting; say which mode ran. Pool tokens per pass (tokens / passes).
- Workloads with the same `--ctx` and environment share a server start (short, mid, long). `server` takes
  `--targets TOKENS[:DECODE],...`; `bench-set` runs the set at `--ctx`; `huge` takes hours per round.
- Corpus: `scripts/make_corpus.py` from one fixed commit; with deterministic mode the SHAs of the 30K and 130K requests
  are compared with `bench/sha.toml`.
- Expert cache hits and misses per request come from `GET /status` (`totals.expert_hits`, `totals.expert_misses`).
- Output: Markdown and JSON next to `--out` (default `work/ab/`).

| Decision | Meaning |
|---|---|
| B faster / B slower | the interval of `mean(d)` excludes 0 (`d = 100 * (B / A - 1)` per round); inside the margin it is marked "real, but inside the margin" and a loss is never a draw |
| equal | `mean(d)` proven inside +-margin (TOST) |
| open | neither; at the round limit: undecided |

`bench/noise.toml` holds the known round-to-round noise per workload and metric; with at least five earlier runs one
round can decide, but only if it lies inside the plan's expected range. A round more than four table standard
deviations from the median and outside the range is marked "outlier" and cannot decide. After a run `ab.py` writes
`<out>-noise.toml` and asks whether to update `bench/noise.toml` (update after A/A runs, never with a changed workload).

### The queue and the desktop guard

- `scripts/gpu_queue.py --kind <exclusive|gpu|cpu> --label <x> -- COMMAND` runs COMMAND under the lock in arrival order
  (`--status` shows the line). `exclusive` (default; speed, `ab.py`) holds the machine, `gpu` (quality, goldens) only
  the GPU, `cpu` (builds, CPU runs) only the CPU; `gpu` and `cpu` run side by side, a `cpu` job must not start GPU work.
  `golden.sh`, `golden-server.sh`, `gpu-status.sh` and `ab.py` use it (lock `TONA_GPU_LOCK`, default `work/gpu.lock`).
  Never a bare `flock`.
- Desktop guard (`scripts/desktop_guard.py`): a `gpu` or `exclusive` job waits until every connected monitor is off
  (DPMS in `/sys/class/drm`; no monitor or a locked screen counts as away) and is stopped within a second when one comes
  on (exit 75; `ab.py` shuts its servers down). `TONA_DESKTOP_GUARD=off` only with the owner's explicit consent.

## Determinism of the prefill plans: `scripts/bolt-determinism.sh`

With `TONA_DETERMINISTIC=1` one prompt must give the same logits and tokens in every prefill plan and repeat. The script
runs `tona-bolt-bench` (Bolt, or Ornith with `BOLT_MODEL`, `BOLT_IDS`, `BOLT_PROFILE`) through the plans, prints the
hashes and exits 1 on any difference; needs model and GPU (queue). To find where two plans or builds part, run both with
`TONA_PF_HASH=1` and compare the `pfhash` lines.

## The mini model

Reference architecture, random weights (12 layers, 64 experts, full vocabulary, every quant type of the real model), 183
MB, loads in 0.1 s.

```sh
python3 tools/get_model.py mini                   # work/models/mini/mini.gguf and mini-mtp.gguf; also mini-ornith, mini-bolt
tona-gpu-gen work/models/mini/mini.gguf "Write a function" -n 32 --mega --check 4
GOLDEN_MODEL=mini scripts/golden.sh record        # references in work/golden/mini/
GOLDEN_MODEL=mini scripts/golden.sh compare       # also GOLDEN_PLAN=cpu and GOLDEN_MODE=prefill
GOLDEN_MODEL=mini scripts/golden-server.sh compare   # both APIs, MTP head of the mini model
scripts/extra-slots-check.sh work/build/engine       # extra expert slots in the prefill workspace: same answers with and without
python3 tools/mini_token_check.py compare --build work/build/cpu   # CPU token check on all three mini models, no GPU (`just test` runs it after the build)
```

The mini models are built from the headers in `data/mini-models/` (tokenizer, template, metadata and tensor
directory of the real models, 1.6 to 2.3 MB each, xz, under the models' own licences: [README](../../data/mini-models/README.md)), so no big model is needed. Building the four files takes 4 s;
the same seed gives the same bytes.

**Token check.** `just test` builds `tona-ref`, builds the mini models and compares the greedy token ids and the
logits of every step of two prompts (128 tokens each) with `data/mini-models/expected/cpu.json`. It takes about 26 s on
4 threads plus 28 s for a cold build. CI runs it as `mini-tokens`. The result does not depend on the thread count or on
`-march` (tested: 1, 2, 4, 16 threads; `x86-64-v3` and `znver5`).

**Changing a header or the expected tokens.** Both change in a reviewed pull request only.

```sh
python3 tools/make_mini_model.py --export-header coder   # also coder-mtp, ornith, bolt; needs the model in work/models/
python3 tools/mini_token_check.py record                 # after a build: just test builds work/build/cpu
```

Record references only with a current build (`golden.sh` exits 2 on a stale binary; `scripts/build-fresh.sh`).
`--arch nemotron_h_moe` writes a mini Bolt (`work/models/mini-bolt/`); with `--f32`
llama.cpp and the CPU reference compare logit by logit (`research/measurements/2026-11-19-bolt-b1-b7.md`).

- **For:** determinism, host logic (scheduler, API parsing, stash and restore, prompt cache), stress tests, shape-dependent
  host paths; a run takes seconds.
- **Not for:** speed, quality, long contexts, the DFlash drafter, accepted MTP drafts, real expert reuse, scale-dependent
  kernel behavior; those run the recipes of [measurement.md](measurement.md) on the big model.

### The synthetic model in full size

`tools/make_synth_model.py` writes the reference shapes with random weights and the expert types of a compression
variant (`--variant a`; `--experts` default 128, 12.4 GB; about a minute). For speed and shape-dependent kernel paths
before a real file exists, not for quality or expert reuse. On macOS `tona-bench` measures it.

**Forcing expert misses:** `tona-gpu-gen --slots N` (N at least 10) or `TONA_VRAM_GB` limits the slots per layer (10
slots: 71 % hits); `TONA_PLAN=cpu` runs pinned experts on the CPU; tokens stay equal. For the server the limit is the
VRAM in use plus 0.5 MB per slot plus about 3 MB; `TONA_VRAM_LOG=1` prints the slots. Stash and restore need one slot
(`TONA_SEQS=1`).

**Golden check times:** mini model about 48 s together. Big model: 27 s (prefill, DFlash2), 27 s (gpu plan), 26 s (CPU
plan), 188 s (server), 350 s (prefill with `GOLDEN_PROMPT_NB=1`); prefill at width 4 about 157 s
(`research/measurements/2026-11-05-scale-run.md`, `2026-11-02-golden-loads.md`).
