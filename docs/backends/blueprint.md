# Backend blueprint

Order of work and traps for a Tona backend on a GPU that has none yet (path B of
[optimize-for-your-hardware.md](../contribute/optimize-for-your-hardware.md)). An agent works through the phases without asking, except at
[Where to ask the human](#where-to-ask-the-human). Interface and rules: [backends/README.md](README.md). Platform pages:
[CUDA](cuda.md), [other AMD GPUs](hip-other-amd.md), [Metal](metal.md), [Vulkan](vulkan.md),
[Intel Xe2](intel-xe2.md).

Read first: [AGENTS.md](../../AGENTS.md), [backends/README.md](README.md), `engine/backend/backend.h`, [docs/hardware/README.md](../hardware/README.md)
and your class file, [measurement.md](../contribute/measurement.md), [verification.md](../contribute/verification.md), [reference-platform.md](reference-platform.md).

A backend implements `tona::backend::Backend`: one call computes a whole forward pass; the backend owns forward pass, prefill,
state, slots, drafts, memory plan and expert strategy. Everything else is shared. The reference (`engine/gpu/`, HIP, megakernel) is
the yardstick, not a template. A backend starts at **tier 3** ([backends/README.md](README.md#tiers)): experimental, named owner, off by
default.

## Phases

Every phase ends with a gate; do not start the next before it is met and written down. Every measurement file opens with
`tools/hw_report.py` output and the plan (AGENTS.md rule 2) and runs on a quiet machine
([quiet check](../contribute/optimize-for-your-hardware.md#steps)). Prefer the smallest run that answers the question
([measurement.md](../contribute/measurement.md#principles)).

| Phase | Work | Gate |
|---|---|---|
| 0 Inventory | the table below, one file per question under `research/measurements/<date>-<name>-<topic>.md`; ceiling written as "the ceiling is X, we are at Y, the next step is Z" | every row has a file or a stated reason it does not apply; watchdog limit and behavior after a timeout measured or cited |
| P Proposal | `research/<nn>-<name>-backend-proposal.md`, modeled on `research/40-mac-backend-proposal.md`: a decision table (proposal, disadvantage), measured starting point, order of work, expected rates, risks | **the human decides**; the agent stops after the proposal and its PR text |
| M0 Kernels | throwaway benchmarks in `bench/<name>/`: expert matvec per quant type at B = 1, 2, 4, 8; dense matvec at B = 1 to 8; (linear-)attention step at short and long context; dequant bit-exact against `engine/model/quant.cpp` | every kernel has a measured time and share of bandwidth; the resident-pass estimate is rewritten from measured parts |
| M1 Resident | the `Backend` implementation (`forward`, `commit`, `reset`, `recover`, slots, `save_state`/`load_state`, `stash`/`unstash`; `prefill` through `forward` first; MTP and DFlash may throw, [contract](README.md#drafts-optional)) on a model that fits resident | checks 1 to 3 below pass; decode measured with spread against the M0 estimate; the reference untouched or measured before and after (`isa-diff`, `bench-set`) |
| M2 Offload | only where the class file says memory is short: expert slots, pinned RAM, SSD (pin rules of `engine/backend/budget.h`), a miss path (CPU compute or GPU wait, with a **cap** and **recovery**), prediction only after measuring its recall (`tools/cache_sim/`); cache size by measurement under background load (a larger Mac cache cut decode from 13.9 to 2.3 tok/s, `research/38-moe-on-mac.md`) | tokens identical to the resident run (`TONA_DETERMINISTIC=1`); no aborted pass or watchdog event in R8 (20 requests under pressure, `TONA_DIAG=1`, [measurement.md](../contribute/measurement.md#recipes), rule of three for zero events); rates with spread against the estimate |
| M3 Speculation, prefill | MTP head (`has_mtp`, `mtp_fill`, `mtp_step`) at B = 2 to 8; prefill with `prefill_chunk() > 0` and its own matrix-unit path | bit-identity at all widths including drafts; a path that rounds differently only without the switch and with the Δ-NLL result; tokens per pass (code and text separately), decode and prefill rates with spread |
| M4 More models | the backend reads shapes from the GGUF; license check ([research/42-model-licenses.md](../../research/42-model-licenses.md)); update the tier list in [docs/hardware/](../hardware/README.md) | M1 checks per model; a BENCHMARKS.md row per model and rate |

Pull requests: one topic each, a phase being a natural unit, [template](../../.github/PULL_REQUEST_TEMPLATE.md); the agent prepares
the text and hands it to the human ([step 12](../contribute/optimize-for-your-hardware.md#steps)).

**Phase 0 inventory** (port the micro-benchmarks of `research/measurements/2026-10-22-micro.md` to `bench/<name>/`):

| What | Result |
|---|---|
| Memory bandwidth, read (long and short buffers) | GB/s, share of vendor peak |
| Caches (L2, last-level) | GB/s per working-set size |
| Compute peaks of the units the model needs (matrix f16, int8, fp8/fp4; vector FMA) | sustained TFLOPS or TOPS, with clock |
| Dispatch / launch cost, back to back and with a sync | us per dependent launch, per round trip (megakernel or many launches) |
| In-kernel sync (grid barrier, CPU to GPU flag in host memory) | us per barrier; forward progress guaranteed? |
| GPU clock after an idle gap of 0.1 to 30 ms | kernel time against gap (Mac: 3.3x slower for ~16 ms after 30 ms idle) |
| **Watchdog and timeout** | limit in s, what happens, whether the queue survives ([watchdogs](#gpu-watchdogs-and-timeouts)) |
| Host link, pinned against pageable copies | GB/s per direction and both together |
| CPU memory bandwidth and ISA (AVX-512, AVX2, NEON) | GB/s (reference: 72 on two CCDs, 58 on one) |
| SSD: sequential, random 2 MB at queue depth 1 and 16 | GB/s, latency (method of `research/measurements/2026-11-11-mac-ssd.md`) |
| Memory budget next to the desktop | GB for your class ([docs/hardware/](../hardware/README.md#how-the-files-estimate)) |

Estimates follow [docs/hardware/README.md](../hardware/README.md#how-the-files-estimate) (bytes per pass, bandwidth, factor 0.58).

**M1 checks**, cheapest first, stop at the first red ([backends/README.md](README.md#checks)):
1. `tona-check MODEL.gguf "PROMPT" -n N --backend NAME` against `cpu/ref_*.cpp` (f32): top-1 and logit deviation per step.
2. `TONA_DETERMINISTIC=1`: every pass width 1 to `BMAX` gives bit-identical logits for a token alone, in a speculative pass and next
   to another slot ([Bit-identity](#bit-identity-and-quality)).
3. `tona-gpu-gen --multi-check` with `TONA_MULTI_DUMP` and `scripts/nll-compare.py`: upper end of the 95 % interval of the mean
   Δ-NLL below 0.01 nats.
4. Speed: decode at least 70 % of the bandwidth bound of the pass; if lower, say what limits it with a measurement.

## Layout

```text
engine/backend/<name>/
  AGENTS.md            rules of this backend (template below)
  CMakeLists.txt       one library, built only with -DTONA_BACKEND_<NAME>=ON
  <name>_backend.cpp   the Backend implementation (host code)
  kernels/             device code, split into headers of one translation unit; no file over 500 lines
  tests/               unit tests without GPU and model, ctest label `unit`
bench/<name>/          the M0 micro-benchmarks
research/measurements/<date>-<name>-*.md   one file per run
docs/hardware/<class>.md                   updated in the same pull request
```

- **Build option default OFF.** Without it the engine builds and runs as before; CI builds without it.
- **Selection through the registry** (`engine/backend/registry.h`, `--backend NAME|auto`, `TONA_BACKEND`): each backend registers a
  name, probe and factory explicitly from `main` (no static initializers). `auto` takes the first backend whose probe passes and
  lists every candidate with its reason otherwise. The reference stays statically linked and bit-identical.
- **Static linking while a platform has one backend.** Loading with `dlopen` comes with the first binary that carries optional
  backends: one exported C symbol `tona_backend_plugin_v1()` returning `{abi, build_id, registry entry}`, a refused load on any
  `build_id` mismatch, `$ORIGIN/../lib/tona/libtona_backend_<name>.so` as search path, no `dlclose`. Whoever builds it also builds
  the loader; requirements: (1) the reference path is unchanged (`isa-diff`, `bench-set`); (2) a library that cannot load leaves a
  clear message and the reference or CPU path usable; (3) `info().name` says which backend runs (`"cuda-sm89"`); (4) a stale
  library is refused; (5) the installer still ships one package ([backends/README.md](README.md#selection)).
- **Per-backend `AGENTS.md`** template:

````markdown
# AGENTS.md: <name> backend

Read the root [AGENTS.md](../../../AGENTS.md) and [docs/backends/blueprint.md](../../../docs/backends/blueprint.md) first.

## Platform
<GPU family, ISA/arch targets, driver and toolkit versions, OS, tier (3), owner (name or handle)>

## Build
<the cmake option, the toolchain, how to run the tests without the GPU>

## Safety
- Watchdog: <limit in seconds, source, measurement file>. Wait cap: <value, variable>. Recovery: <what happens>.
- Never <call that blocks while the GPU waits on the CPU>.
- Risky first runs (new device code, waits on the CPU): warn the user first.

## Memory
<budget rule, desktop reserve, pinning rule, variables>

## Checks that must pass before a change here
<check commands from backend.md and blueprint, bit-identity, quality rule>

## Do not
- change shared code (`model/`, `server/`, `cpu/`, `backend/`) without a before/after on the reference;
- copy third-party code without license check and origin note.
````

## What is shared and never touched

From [backends/README.md](README.md#split): `model/` (GGUF loader, tokenizer, quant formats), `server/`, `cpu/ref_*.cpp`, `tools/`,
`scripts/`, `bench/` of others, `engine/backend/backend.h`, the goldens in `work/golden/` and `bench/sha.toml`. Use them; do not
reimplement them.

If shared code must change (a new quant type, a generator hook, a field in `Stats` or `Info`):
1. Make it a separate pull request or commit with a reason independent of your backend.
2. Measure the reference before and after: `scripts/isa-diff.py` prints `all kernels identical`; `scripts/bench-set.py` with
   alternating builds (`scripts/ab.py`), same tokens, speed within the spread; the goldens `python3 tools/select_checks.py` names
   ([verification.md](../contribute/verification.md#which-checks-a-change-needs)).
3. State "effect on the reference" with numbers; a slower reference is refused (rule 1).
4. `backend.h` keeps every existing method (add, do not reshape); a method your backend cannot serve throws.

## What to watch out for

### The reference never gets worse

Rule 1 of [AGENTS.md](../../AGENTS.md). Anything per pass or per token is measured on the reference before and after.

### Bit-identity and quality

- With `TONA_DETERMINISTIC=1` every pass width computes identically ([backends/README.md](README.md#bit-identity)), because the generator
  verifies drafts by comparing: fixed reduction order (no atomic float add in a reduction that feeds logits), the same dot format
  at every width.
- Without the switch a wide pass may round differently, but the quality check applies ([backends/README.md](README.md#checks)): Δ(-log p)
  with the 95 % interval rule, KL and top-1, the task benchmark once for a new kind of change.
- Disagreement with `cpu/ref_*.cpp` (f32, `--check N`) beyond the format's quant noise is a bug.
- A deliberate change of numbers (lower precision attention, a lossy approximation) is a lossy lever: [ASK](#where-to-ask-the-human).

### GPU watchdogs and timeouts

On the reference a hung kernel triggers a GPU reset that takes the desktop with it ([AGENTS.md](../../AGENTS.md#gpu-safety-reference-backend),
`research/09-gpu-reset.md`). Measure your platform in phase 0 in a throwaway process and write limit, source and behavior into the
backend's `AGENTS.md`.

| Platform | Limit | Scope and aftermath | Mark |
|---|---|---|---|
| AMD, Linux (reference) | `lockup_timeout` 2 s per drm job; KFD user queues are no drm job; KFD and MES have own timeouts (queue preemption 9 s, MES fence 2.1 s), then queues are reset | a reset takes the desktop with it; the megakernel pauses after 100 ms (`TONA_MEGA_YIELD_MS`); `research/23-waiting-in-kernel.md`, `research/09-gpu-reset.md` | from kernel source, not measured |
| macOS, Metal | an event wait of 4.5 s completes, 5 s aborts with `kIOGPUCommandBufferCallbackErrorTimeout`; waiting counts as running | one timeout leaves the queue usable; repeated ones can end in `accessRevoked`; `2026-11-11-mac-timeout.md` | measured |
| Windows, any API | TDR 2 s for a task the scheduler cannot preempt; five hangs in a minute bug-check | GPU reset, video memory purged, applications recreate the device | [V] [Microsoft Learn](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/timeout-detection-and-recovery) |
| NVIDIA, Linux, display on the same GPU | old notes: a few seconds with a display, none without; `cudaDevAttrKernelExecTimeout` reports it | Wayland on current drivers **unchecked**; query the attribute and test | [S] forum threads; [V] [CUDA runtime types](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html) |

Rules for every platform:
1. **No blocking call while the GPU waits on the CPU** (any synchronize, allocation or copy that needs the queue to drain); list
   them in the backend's `AGENTS.md`.
2. **A wait has an upper limit well below the watchdog** and the CPU ends it. Reference: pause after 100 ms. Mac: `TONA_METAL_WAIT_MS`
   100 ms, 50 times below 5 s. Name cap, variable and margin.
3. **Split long work** so no submission reaches the limit (Mac: split at 1 s, margin 4.5x).
4. **Recovery:** an aborted pass leaves the state as before (write KV and recurrent state after the pass or restore the generator's
   checkpoint); the generator repeats the pass; `recover()` makes the state usable. After a real timeout: no retry loop, recreate the
   queue once, a second error within a minute stops the backend with a clear message (`EXIT_GPU_STALLED`, `engine/backend/stall.h`).
5. **Forward progress between blocks is not guaranteed everywhere.** A grid barrier or spin on another block's flag needs all blocks
   resident: cooperative launch (CUDA, HIP) guarantees it, Vulkan and Metal do not (OOPSLA 2021 [S]; a fused megakernel measured
   1.8x slower than separate dispatches on an M4 Pro, arXiv 2610.31479). Then the barrier becomes a dispatch boundary.
6. **Test a risky run in a throwaway process and warn the user** ([ASK](#where-to-ask-the-human)).
7. **Clock after idle:** the Mac GPU ran up to 3.3x slower for about 16 ms after an idle of 30 ms or more
   (`2026-11-11-mac-overlap.md`); measure after gaps of 0.1 to 100 ms; a short busy kernel while a request is parsed hides it.

### Memory planning per class and the desktop

- Budget = capacity minus what desktop and OS keep, read from the driver or OS, not the runtime's "free memory": the reference reads
  `/sys/class/drm/card*/device/mem_info_vram_used` because `hipMemGetInfo` does not see the desktop
  ([reference-platform.md](reference-platform.md#what-runs-next-to-the-engine)). Mac `recommendedMaxWorkingSetSize` is a ceiling,
  not a plan.
- Keep a desktop reserve: 1.4 GB plus `TONA_VRAM_RESERVE_MB` (1024) on the reference.
- Use `TONA_RAM_GB`, `TONA_VRAM_GB` and the pin rule in `engine/backend/budget.h`. Allocate large host buffers once at start and pin
  them; pageable sources and re-registering host memory caused queue evictions on AMD
  (`research/measurements/2026-10-23-pinned-buffers.md`).

### Expert strategy

Decide by measurement in this order: (1) all experts resident; (2) hot ones resident, the rest on the CPU (host bandwidth: DDR5 72
GB/s on the reference, 58 on one CCD, DDR4 about 40 [K]); (3) the rest streamed over the link or from the SSD. The hit-rate curve
(33 % resident about 85 % hits, 67 % about 94 %) is a provisional llama.cpp anchor, unchecked below 33 %; replace it with the cache
simulation on your route traces (`tools/cache_sim/`). Dropping or substituting an expert on a miss (`TONA_MISS_DROP`) is a lossy lever.

### Quant formats

`engine/model/quant.cpp` is the reference of every format; your device dequant must be bit-identical on random blocks before a kernel
uses it. A platform with another native dot format (int8, fp8, fp4) converts activations on the fly and must pass the quality rule.
A format without a fast path may be dequantized to f16 in registers at the cost of bandwidth: measure.

### Third-party code

Adapting a kernel is welcome ([AGENTS.md](../../AGENTS.md#approach)). Check the license (Tona is MIT; MIT, BSD and Apache-2.0 are
compatible with an origin note; GPL and unknown need the human), write the origin note in the file header (project, file, revision,
license, what changed), keep vendored code under `engine/third_party/` whole and unmodified. Never paste code from a source you
cannot name.

### Measuring and gates

AGENTS.md rule 2 and [measurement.md](../contribute/measurement.md) apply. **No program names in any file, log, commit or PR** (sums only);
`tools/gates/measurement_privacy.py` rejects a process list in `*.jsonl`. No build while a measurement runs; one GPU block per change
(`TONA_GPU_LOCK`); files above ~100 MB go to `work/`. The code gates ([Code gates](../../AGENTS.md#code-gates)) hold for GPU code
too: split device code into headers of one translation unit and pass many kernel arguments as one struct. A toolchain the gates do
not know (`.mm`, `.cu` for clang-tidy) is reported in the pull request; the gate is not weakened.

## Where to ask the human

The places of the playbook ([optimize-for-your-hardware.md](../contribute/optimize-for-your-hardware.md#where-the-agent-asks-the-human-and-only-there)),
and only these:

| ASK | When |
|---|---|
| Decisions | the proposal (kernel route, pass structure, memory plan, loading) after phase 0 |
| Downloads | a model or toolchain above 5 GB: size and target folder, wait for a yes; into `work/` or `/var/tmp` |
| Risky GPU tests | first run of new device code, a kernel that waits on the CPU, a timeout test, anything that can reset the GPU |
| Licenses | a model or third-party file with an unknown or non-MIT-compatible license |
| Lossy levers | lower quantization, dropped or substituted experts, a lossy approximation |
| A busy machine | measure only on a quiet machine; the user decides what to close |

Everything else the agent decides and records in the measurement file.

## Sources

[backends/README.md](README.md), `engine/backend/backend.h`, [reference-platform.md](reference-platform.md), `research/09-gpu-reset.md`,
`research/21-backends.md`, `research/23-waiting-in-kernel.md`; Mac notes `research/38-moe-on-mac.md`, `39-mac-engines-2026-11.md`,
`40-mac-backend-proposal.md`, `41-mac-reconciliation.md`, `research/measurements/2026-11-11-mac-*.md`; source marks:
[docs/hardware/README.md](../hardware/README.md#source-marks).
