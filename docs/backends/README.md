# Contributing a backend

Interface and rules. Phases for a new GPU backend: [blueprint.md](blueprint.md). Reference platform:
[reference-platform.md](reference-platform.md). A backend computes the whole forward pass with its own kernels and
expert strategy; the rest is shared.

## Ground rules

1. **The reference never gets worse**; it takes no consideration for other backends.
2. **One call, one whole pass**: no single ops (matmul, attention) in the interface.
3. **Measure, do not guess**: several runs with spread, one file per run under `research/measurements/`.

## Split

| Shared | Per backend |
|---|---|
| `model/`: GGUF loader, tokenizer, quant formats | forward pass and prefill |
| `server/`: API, chat template, generator, planner, prompt cache, sampling, speculation | drafts (MTP head, DFlash) |
| `cpu/ref_*.cpp`: f32 reference | state, slots, checkpoints |
| `tools/`, `scripts/`, benchmarks | memory plan, expert cache, kernels |

Interface: `engine/backend/backend.h` (`tona::backend::Backend`). Reference backend: `engine/gpu/` (`gpu::Mega`). New
backends: `engine/backend/<name>/`, one library each.

## Selection

One binary, backend chosen at start.

- **Registry** (`engine/backend/registry.h`): `main` registers every linked backend explicitly, in the order `auto` tries
  them. `tona-server --backend NAME|auto` (or `TONA_BACKEND`); each backend has a probe saying whether the machine can
  run it and why not. Options: `backend::CreateOptions` (`engine/backend/options.h`).
- **Architectures and weight types:** an entry declares its architectures, per architecture the ggml weight types it has
  kernels for, and its tier (`ArchSupport`). `auto` takes the first backend whose probe passes and which lists the
  architecture. Before `create` the registry names the first tensor without a kernel ("blk.1.ffn_up_exps.weight is
  IQ2_XXS; hip-gfx1200 has no kernel for it for nemotron_h_moe"); embedding tables are not checked. `tona-server
  --list-backends` prints the matrix. A new quant type is declared in the same change as its kernel.
- **Linked statically:** `hip-gfx1200` on Linux (ROCm toolchain), `metal` on macOS (`-DTONA_BACKEND_METAL=ON`). `dlopen`
  comes with the first optional backend (planned: one exported C symbol per library, an exactly matching build id,
  `tona_model`/`tona_cpu` shared).

Virtual calls cost about 30 per pass of 20-60 ms: not measurable.

## Interface contract

Except `info()` right after loading, all calls come from one worker thread; no locks.

### Pass

1. `forward(tokens, nb)`: nb tokens (1 <= nb <= `BMAX` = 8) from `pos()` of the active slot; logits `[nb][vocab]` valid
   until the next call.
2. Exactly one `commit(m)`, 1 <= m <= nb: the first m tokens are accepted, the state is as if only they had run.
3. `forward_multi(slots, n, k, tok)`: one pass over k slots; then per slot `use_slot(s)` and `commit(m)`.

### Prompt

`prefill(tokens, n, nll)` from `pos()` (with `nll`: -log p per token). `prefill_chunk()` > 0: own chunked prefill; 0: the
generator uses `forward`.

### Slots

`seqs()` conversations (up to `SMAX` = 4), each with its own state; `use_slot(s)` selects the active one.

### State

- `save_state()` / `load_state()` / `has_state()`: one checkpoint per slot; `snap_tokens()`, `snap_tokens_of(slot)` say how
  far it reaches. `save_state` before `commit` is an error.
- `stash(HostSnap&)` / `unstash`: checkpoint to RAM and back (`HostSnap` content is the backend's); `snap_fits` checks a
  stored state.
- `reset()` to position 0; `recover()` after an exception in a pass.

### Drafts (optional)

`has_mtp()` (false without an MTP head; the generator asks first); DFlash only if loaded; what a backend cannot do
throws.

### Counters and information

`info()`: name (`"hip-gfx1200"`), memory and split, for `/status` and the log. `stats()`: unused fields stay 0.

### Bit-identity

With `TONA_DETERMINISTIC=1` every pass width computes bit-identically (alone, with drafts or beside another
conversation). Without it a backend may round wide passes differently; quality is then measured.

## Checks

1. **Correctness:** `tona-gpu-gen MODEL.gguf "PROMPT" --check N` compares the first N steps' logits with `cpu/ref_*.cpp`.
2. **Quality over long texts:** `tona-gpu-gen --multi-check` with `TONA_MULTI_DUMP=FILE`, 10,000 tokens each of code and
   shell, then `scripts/nll-compare.py REFERENCE.txt VARIANT.txt` (method `research/measurements/2026-10-26-oea.md`;
   for long-context changes also the teacher-forced tail at 32K, `2026-10-29-indexcache.md`). Report KL and top-1.
   Rule on the 95 % interval of the mean Δ(-log p), per text:
   - upper end below 0.01 nats: pass;
   - mean below 0.01, upper end at or above: undecided; more tokens or the task benchmark (`bench/threshold/`);
   - mean at or above 0.01 nats: fail.

   Calibration: `research/measurements/2026-10-30-threshold-main.md`. A new kind of change (context extension,
   lower-precision attention, merged selections) also runs the task benchmark once.
3. **Tasks:** `scripts/bench-effort.py medium` (23 tasks; finds gross failures only, the reference varies by 1-2 of 46).
4. **Speed:** `scripts/bench-set.py` (one conversation) and `scripts/bench-parallel.py URL 2`, several runs, alternating
   with the comparison version; name every rate (decode or prefill).
5. **Reference unchanged:** `bench-set` on the reference before and after any shared change: same tokens, speed within
   spread.

`tona-server` and `tona-check` (any backend against `cpu/ref_*.cpp`) select through the registry; `tona-gpu-gen` knows
only the reference.

## Tiers

Evidence (goldens, quality check, benchmarks) is per model: a pull request names the model(s) it measured, the rest are
"not measured" and can be taken over in the issue.

- **Tier 1, reference:** maintainers own the hardware; release-blocking; all gates, goldens and benchmarks per change.
- **Tier 2, maintained:** a named maintainer with the hardware; CI on their hardware or a runner they provide; checks
  pass. A regression blocks that backend's release notes, not the reference.
- **Tier 3, experimental:** a named maintainer and docs, no CI, no guarantee; off by default (build option); removed
  without a maintainer.

### Time limits

Procedure: [maintainer.md](../maintain/maintainer.md#4-backend-contribution). A broken backend never blocks a reference release.

| Step | Limit |
|---|---|
| Tier 3 to tier 2 | 3 weeks stable, public CI on the owner's hardware, checks above passed, and a second person who can build it or a public CI job that builds and tests the documented steps on a GitHub-hosted runner of that platform |
| Owner counts as active | a commit, review or reply in the last 90 days, and CI green |
| Reply to a mention | 14 days |
| No reply in 14 days and no activity for 90 days | marked `orphaned` and announced |
| `orphaned` for 60 days without a new owner | demoted or removed in one PR; the code stays in history |
