# Measurement policy: which runs, how many, how large

A fixed recipe per kind of change. Refines rule 2 of [AGENTS.md](../../AGENTS.md); checks: [verification.md](verification.md);
quality checks: [backends/README.md](../backends/README.md#checks). Every measurement file opens with `tools/hw_report.py` and
`tools/quiet_check.py` before and after; measure only on a quiet machine
([optimize-for-your-hardware.md](optimize-for-your-hardware.md)).

## Principles

1. **Plan first:** recipes, expectation with tolerance and source, active tooling; say why a run deviates.
2. **Smallest run that answers the question:** a `--ctx` cost shows with a short prompt, an attention cost needs a long
   one.
3. **Deterministic counters before rates** (expert hit rate, MB reloaded, pinned GB, kernel time per pass); a tok/s rate
   confirms.
4. **Cheapest recipe first, stop at the first failure.**
5. **The big model where scale matters** (long context, memory pressure, long sessions).
6. **One GPU block per change** (`TONA_GPU_LOCK`, default `work/gpu.lock`).
7. **Sums, no program names** (available RAM and VRAM, desktop VRAM, swap, load, CPU and GPU share);
   `tools/gates/measurement_privacy.py` rejects a `"top"` field in `research/measurements/*.jsonl`.
8. **A measured gain is dropped only after a fresh search** for a cheaper route (papers, PRs, issues:
   `tools/papers/scan.py`, [research/papers.md](../../research/papers.md)) and an attempt to remove its cost. The file
   records: "not taken because X; searched on DATE (topics, window): found Y / nothing new; tried Z".

## Models

| Model | Path | Use |
|---|---|---|
| big | `work/models/coder-iq1_m/` (58.4 GB, `toolchain/models.toml`; drafters in `work/models/draft/`) | every recipe unless it names tiny |
| tiny | `work/models/mini/` ([testing.md](testing.md#the-mini-model); `GOLDEN_MODEL=mini`) | determinism, host logic, stash and restore, prompt cache, API, stress; never speed or quality |

For host code (C2, C3) run R3 on tiny first (about 48 s); R3 on big still decides.

## Workloads

Named workloads of `scripts/ab.py`; prompts from `scripts/make_corpus.py` (one commit) and `scripts/bench-set.py`.

| Name | Requests | Prompt | Decode | Server `--ctx` | Measures |
|---|---|---|---|---|---|
| `short` | 2: summarize code, write a unit test | 2K tokens each | 400 | 140000 | decode tok/s, expert cache hits/misses per request |
| `ctx` | `short`, once per context | 2K | 400 | 140000 and 262144 (service default) | `--ctx`-dependent costs: buffers, pinning, slots |
| `mid` | 1: summarize code | 30K | 200 | 140000 | prefill tok/s at full groups, decode after |
| `long` | 1: summarize code | 130K | 200 | 140000 | prefill and decode where attention dominates; bit-exact SHA |
| `set` | 14: 11 code, 3 text | bench-set | up to 400 | 32768 | decode of the production path (MTP drafter) |
| `prose` | 2: explain a slice of the prose corpus (`scripts/make_prose_corpus.py`) | 2K and 16K | 400 | 32768 | Bolt: prefill of new text, decode of prose |
| `longdoc` | 1: report on a long document | 65536 | 200 | 98304 | long-document prefill and decode (Bolt: attention in 8 layers) |
| `huge` | 1 each: summarize code | 512K and 1M | 200 | 1048576, `TONA_SEQS=1` | prefill and decode past 262K |

## Recipes

| Recipe | Model | What runs | Rounds |
|---|---|---|---|
| R0 gates | none | `just check` | 1 |
| R1 unit | none | `just test-cpu`; `just test-py` when Python changed | 1 |
| R2 kernels | none | `scripts/isa-diff.py <base build> <your build>`: `all kernels identical` | 1 |
| R3 golden | big | `scripts/gpu-status.sh`: the checks `tools/select_checks.py` names (all five when undecided) | 1 |
| R4 bit-exact | big | `scripts/ab.py` with `TONA_DETERMINISTIC=1`, `mid` and `long`; SHAs equal `bench/sha.toml` | 1 |
| R5 speed | big | `scripts/ab.py`, workloads below, base against yours | `ab.py` decides |
| R6 quality | big | [backends/README.md](../backends/README.md#checks): Δ-NLL, top-1, task benchmark when the rule says | as there |
| R7 long context | big | `scripts/ab.py` workload `huge`; `bench/needle --hard` at the targeted lengths, one text each | 1 per arm |
| R8 pressure | big | 20 requests, 10 under page-cache pressure, `TONA_DIAG=1`: `scripts/r8_pressure.py --build <build> --pressure-files '<glob of big unused files>'` (RAM only, capped, port 8091) | 1; a second only if the first misses the expectation |

| The change affects | R5 workloads |
|---|---|
| decode, no attention change | `short`, `set` |
| decode with attention, KV or indexer | `short`, `long`, `set` |
| prefill | `mid`; plus `long` for attention or indexer |
| allocation depending on `--ctx` | `ctx` |

## Classes of change

`tools/select_checks.py` names the class; when undecided the higher class applies.

| Class | What changed | Recipes |
|---|---|---|
| C0 | docs, `research/`, comments, `*.md` | R0 |
| C1 | `scripts/`, `tools/`, `bench/` feeding no golden check | R0, R1 |
| C2 | host code off the hot path (API parsing, installer, updater, CLI) | R0, R1, R3 |
| C3 | host code on the hot path (generator, scheduler, stash, prompt cache, memory planning, pinning) | R0, R1, R3, R5; R4 when stash or prompt cache changed |
| C4 | device-code refactor, no number change | R0, R2; kernels not identical: as C5 |
| C5 | device code that changes machine code | R0, R1, R3, R4, R5 |
| C6 | numbers change on purpose | R0, R1, R6, R5, then R3 with re-recorded references |
| C7 | long context (limits, RoPE/YaRN, indexer, top-k, KV layout, `TONA_SEQS`, past 262K) | C5 or C6, plus R7 |
| C8 | stability (stalls, memory pressure, host memory type, time limits, recovery) | C3, plus R8 on the final build |

## How many rounds

**Rates (tok/s):** `scripts/ab.py` alternates the builds and stops when every rate is decided (faster, slower, equal
within the 0.5 % margin; a stability fix may cost up to 2 %). With known noise (`bench/noise.toml`) one round may decide
if it lies inside the plan's expectation; otherwise 2 to 6 rounds; "undecided" after six is reported. Details:
[testing.md](testing.md#speed-comparison-scriptsabpy).

**Counts (slow requests, faults, aborts):** zero events in *n* trials bound the rate below 3/*n* at 95 % (0 in 20: below
15 %; 0 in 40: 7.5 %; 0 in 100: 3 %). To show a fix lowers a rate from *p*₀ to *p*₁, state *n* in the plan so the
expected counts differ clearly. A rate near 1 % is not hunted with repetitions; `TONA_DIAG=1` and scale runs catch it.

## Scale run

At least every three days and before every release, on `main`, with a plan. Recipes: all five golden checks (prefill
with `GOLDEN_PROMPT_NB=1`, 359 s), R4, R5 with `set` and the curve `mid`, `long`, `huge` (one build), R7 needle at 128K
and 1M, R8, and the peaks at the service default context (host RSS, GTT, VRAM, pinned GB). Expectation: the previous
scale run with tolerances; every number outside gets an issue. File: `research/measurements/<date>-scale-run.md`, one row
in BENCHMARKS.md.

## Before a release

The scale run, R6 on the release candidate, and three runs of every number the README publishes, with spread.
