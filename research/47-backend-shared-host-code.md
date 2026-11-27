# Shared host code across backends: audit of engine/gpu (reference) and engine/backend/metal

Question: do we write host code twice that every platform (HIP reference, Metal, later CUDA) has to rewrite? Intended
split ([docs/backends/README.md](../docs/backends/README.md#split)): compute kernels per platform, everything else shared.

Scope: C++ host code only, no kernels. Read-only audit of the tree at `main` (422030c); nothing was built or run, so
every statement about speed below is an argument from the code path, not a measurement. Line numbers are of that
commit.

## Summary

- The shared layer today (`engine/backend/`): `env.h`, `budget.h`, `nan_guard.h`, `rope_yarn.h`, `idx_share.h`,
  `stall.h`, `options.h`, `registry`. These are used by both backends and are fine.
- Metal (about 3,000 lines, of which about 1,000 are host logic outside kernel encoding) copied the reference's host
  logic in five places instead of calling shared code, because the shared layer has no home for them yet:
  PLE context hash, MTP truncated vocabulary and ring, MTP bookkeeping, stash/snapshot layout arithmetic, small
  numeric helpers.
- The PLE context hash exists four times (CPU reference, old HIP `Forward`, HIP prefill, Metal). The MTP vocabulary
  list builder is byte-for-byte the same in two files.
- Copies have already drifted (section "Drift found"): Metal never prints NaN warnings, its MTP vocabulary scan is
  not per slot and not reset, the reference sets `hlast_pos_` where Metal does not.
- Not yet duplicated but about to be: expert cache policy, expert profile file, expert selection (OEA, union,
  ranking). They sit in `engine/gpu/`, partly tied to `gpu::Mega`. Metal's milestone M2 ("no memory budget or expert
  streaming yet") would copy them. Cheapest time to move them is before M2.

## 1. Duplicated or near-duplicated host logic

Notation: R = reference (`engine/gpu/`, `engine/cpu/`), M = `engine/backend/metal/`. "Neutral" = needs no HIP or
Metal type. "Hot" = runs per pass or per token.

### 1.1 PLE n-gram context and row hash

Same arithmetic four times: context of `ple_ngram` tokens (a missing predecessor or anything before an EOS becomes
EOS), then per gram `mixed = ctx[0]*mult[0] ^ ctx[j]*mult[j]`, row `= mixed % head_vocab + head_offset`.

| Where | Lines |
|---|---|
| R, CPU reference | `engine/cpu/ref.cpp:134-160` (`ple_context`, `ple_lookup`) |
| R, old HIP path | `engine/gpu/forward.hip:265-290` (`Forward::ple`) |
| R, HIP prefill | `engine/gpu/prefill_entry.cpp:229-254` (`ple_rows`, rows for a chunk) |
| M | `engine/backend/metal/host_math.h:31-57` (`ple_rows`) |

Same: everything except where the context tokens come from (a vector, `tokens_`, a window of a sequence) and what is
done with the row (dequantize it, gather over io_uring, memcpy from the mapping). Neutral: yes, it needs only
`Hparams`. Only the dequantize/gather step is per platform.

Proposal: `engine/model/ple.h` (next to `Hparams`, no backend dependency):

```cpp
// Table rows of token at index i of seq (seq[0..i] is the history). out[heads].
inline void ple_rows_at(const Hparams& h, const int32_t* seq, int64_t i, int64_t* out);
```

Context on the stack (`int64_t ctx[8]`, as `prefill_entry.cpp:236` does today), no allocation. The reference's
prefill calls it per token of a chunk; it is host code that runs while the GPU works on the previous part, and an
inlined header function is the same code as today's local copy, so no cost. Metal currently allocates two vectors per
token (`host_math.h:33,42`); the shared version is cheaper for Metal.
Rule 1 check: pure integer function, bit-exact by construction. Tests: `test_metal_host.cpp` (rows) plus a check that
all four old call sites give the same rows on the mini model before the old copies are deleted.

### 1.2 MTP truncated vocabulary: list and dynamic ring

| Piece | R | M |
|---|---|---|
| Fixed list from `mtp_vocab.inc`, `TONA_MTP_VOCAB`, `TONA_MTP_VOCAB_N`, dedupe, bounds | `gpu/mega_build_mtp.cpp:159-176` | `metal/mtp_vocab.mm:14-34` (same code) |
| `TONA_MTP_DYN` rows appended, `in_head` map | `gpu/mega_build_mtp.cpp:190-193,197` | `metal/mtp_vocab.mm:41-45` |
| Round-robin `take` of a context token into a free row | `gpu/mega_mtp.cpp:116-123` | `metal/mtp_vocab.mm:55-64` |
| `sync`: scan new context tokens, then `next` | `gpu/mega_mtp.cpp:111-125,141` | `metal/mtp_vocab.mm:66-72` |
| Row mapping row -> token, `mtp_token(r)` | `gpu/mega.h` (`mtp_vocab_`), `mega_mtp.cpp:288` | `metal/mtp_vocab.h:22` |

Same: everything in the table. What differs is the row copy: R copies from the GPU copy of the LM head or converts
to IQ4_NL (`mega_mtp.cpp:94-107,126-140`), M memcpy's from the mapping (`mtp_vocab.mm:61`). Neutral: yes (the row copy
is a callback).

Proposal: `engine/backend/mtp_vocab.h`:

```cpp
class MtpVocabMap {                       // plain C++
public:
    MtpVocabMap(uint32_t vocab, const char* list_file, size_t list_max, size_t dyn);
    size_t rows() const; size_t n_static() const; int32_t token(size_t r) const;
    // Rows to (re)load for the new context tokens: calls load(row, token) for each; scan is per slot.
    template <class Load> size_t sync(const std::vector<int32_t>& tokens, int32_t next, size_t& scan, Load&& load);
    void head_rows(...);                  // the static rows, in order, for the initial fill
};
```

The reference keeps its two-piece device copy and its map upload by recording the rows `load` was called for (it
already works with "first and n" in `mega_mtp.cpp:115,127`). `sync` runs once per `mtp_fill`, that is once per
decode round, and is a loop over the new tokens: the shared version is the same loop. No cost.
Rule 1: the reference must stay bit-identical in vocabulary and ring order (output tokens, `stats.mtp_dyn`); a test
that replays a token stream through old and new ring is enough.

### 1.3 MTP bookkeeping (which h is known, what to catch up)

| Piece | R | M |
|---|---|---|
| Where is the target's h of position q (last pass rows, saved row, unknown) | `gpu/mega_mtp.cpp:146-152` (`mtp_h_of`) | `metal/runner_mtp.mm:17-25` (`trunk_h`) |
| Catch-up loop: chunks of at most BMAX from `min(valid, pos)` to `pos + (next>=0)`; token is `tokens[p+b]` or `next`; stale count; the last chunk wants the head | `gpu/mega_mtp.cpp:252-265` | `metal/runner_mtp.mm:98-112,132-139` |
| After fill: keep h of pos-1, `mtp_valid = pos`, `hlast_pos` | `gpu/mega_mtp.cpp:266-271` | `metal/runner_mtp.mm:140-147` |
| Step precondition and position `mtp_pos + mtp_nb`, context full check | `gpu/mega_mtp.cpp:291-296,311-313` | `metal/runner_mtp.mm:150-159` |
| Invariants on load_state/unstash/reset: `mtp_valid = min(valid, pos)`, `mtp_nb = 0`, `hlast_pos` | `gpu/mega_state.cpp:60,83-84,184,199` | `metal/runner.mm:242,269`; `slot` fields `runner.h:22-26` |
| Fields | `SeqHost` (`gpu/mega_seq.h:23-34`) | `Slot` (`metal/runner.h:17-27`) |

Same: the state machine and the chunking. Different: where h lives (device pointer vs host vector) and how a chunk is
run.

Proposal: `engine/backend/mtp_track.h`, a plain struct per slot that owns the integers and decides, the backend
executes:

```cpp
struct MtpTrack {
    int64_t valid = 0, pos = 0, hlast_pos = -1; int nb = 0;       // as mtp_valid_, mtp_pos_, ..., mtp_nb_
    enum class Src { None, Pass, Saved, Unknown };
    Src h_of(int64_t q, const PassWindow& fw, Src* ...) const;    // fw = {pos, n} of the last target pass
    template <class Run> void catch_up(int64_t pos, int32_t next, const std::vector<int32_t>& tokens, Run&&);
    void after_fill(int64_t pos, bool h_saved); void on_load(int64_t pos, ...); void on_reset();
};
```

Hot? `h_of` and the catch-up loop run once per MTP fill and per chunk (about once per decode round), integer work
only. As templates they inline. The risk is not speed but touching `SeqHost`, which `swap_seq` (`mega_seq.cpp:72-83`)
swaps field by field and which `Mega` reads as plain members all over; see ranking.

### 1.4 Snapshot layout arithmetic, stash, snap_fits

| Piece | R | M |
|---|---|---|
| Pool bytes of a snapshot `(pos/idx_block + 1) * idx_dim * 4` | `gpu/mega_state.cpp:104,140,150,156` (four times in the reference alone) | `metal/seq.h:34` (`pool_rows`), `seq.mm:89,109,122`, `runner.mm:282-283` |
| KV bytes `pos * kv_row`, per QSA layer | same four places | `metal/seq.h:33`, `seq.mm:89,109,122` |
| `stash_bytes` | `gpu/mega_state.cpp:136-142` | `metal/runner.mm:279-284` |
| `snap_fits` core: `pos` in 1..max_ctx, `tokens.size() == pos`, state size, kv and pool sizes | `gpu/mega_state.cpp:144-152` | `metal/seq.mm:119-123` |
| Order in `HostSnap`: state, then per QSA layer kv, pool | `gpu/mega_state.cpp:107-118,159-165` | `metal/seq.mm:81-117` |
| Fields set after stash: tokens, pos, commit, parity | `gpu/mega_state.cpp:100` | `metal/runner.mm:276` |

Neutral: yes (sizes only; the copy itself is per platform: guarded pipelined D2H on R, memcpy from unified memory on M).

Proposal: `engine/backend/snap_layout.h`:

```cpp
struct SnapLayout { size_t state; size_t dring; uint32_t n_qsa; size_t kv_per_pos, raw_per_pos, pool_row; uint32_t idx_block;
    size_t kv_bytes(int64_t pos) const; size_t pool_bytes(int64_t pos) const; size_t total(int64_t pos) const;
    bool fits(const HostSnap&, int64_t max_ctx) const; };
```

Not hot (per request). The reference's formulas stay as they are numerically; this is an extraction. Pairs with the
checkpoint file code in `server/checkpoint_file.cpp:117` which already compares sizes from `state_bytes()` etc.

Slot bookkeeping that is the same but trivial: validation of `commit(m)` (`gpu/mega_serve.cpp:393-396`,
`metal/runner.mm:117-120`), slot index check (`gpu/mega_seq.cpp:102`, `metal/runner.mm:63-66`), forward argument
checks (`gpu/mega_serve.cpp:145-147`, `metal/runner.mm:100-106`), "save_state/mtp_fill/... before commit"
(`gpu/mega_state.cpp:49,65`, `metal/runner.mm:59-61`), tokens/pos update on commit. A `SlotBook` (tokens, pass,
pos, snap tokens/pos/valid, and the validators) would carry these. Per pass cost: a few integer compares and one
`vector::insert` as now.

### 1.5 Small numeric helpers

| Helper | Copies |
|---|---|
| -log p of a token from a logit row (max, sum exp in double, log) | `metal/backend.mm:155-160`, `metal/runner.mm:221-226`, `tools/tona_check.cpp:203` (three in two backends alone; R computes it on the GPU `gpu/prefill_mtp.cpp:52`) |
| NaN -> -FLT_MAX, +-inf clamp | `metal/runner_mtp.mm:87` re-implements `nan::clean_one` (`backend/nan_guard.h:25`) |
| MTP pick: argmax over rows, skipped free rows, probability `1/sum exp(l - max)` | `metal/runner_mtp.mm:80-95` on the host; R does it on the GPU (`OP_MTP_PICK`) and only maps (`gpu/mega_mtp.cpp:84-89,284-289`) |
| RoPE cos/sin for a position, with YaRN | `metal/host_math.h:18-28` = `cpu/ref.cpp:46-57` (same formula; `rope_yarn.h` holds the table, not the angle) |
| Indexer cells of a QSA layer (`n_pool`, `n_top`, tail) | `metal/host_math.h:66-74` = `cpu/ref.cpp:331` = `gpu/forward.hip:222` |
| Row bytes of a ggml type | `gpu/matvec_tables.h:89-110` (switch), `cpu/experts.cpp:255` (second switch), `metal/weights.mm:67-79` (via `ggml_type_info`, no switch); also `bytes_per_expert` twice in R (`gpu/mega_experts.cpp:23`, `gpu/mega_prefetch.cpp:19`) |
| IQ4_NL quantizer used for the truncated head | `gpu/mega_mtp.cpp:24-69`, plain C++ with no GPU type, in the wrong directory (belongs next to `model/quant.cpp`); a CUDA backend with `TONA_MTP_HEAD=iq4` would copy it |
| Env parsing with the same name | `TONA_SEQS`: R `gpu/mega_build_slots.cpp:20` (default 2), M `metal/backend.mm:30` (default 1), same clamp; `TONA_MTP_VOCAB`, `_VOCAB_N`, `_DYN` parsed in both (1.2) |

All neutral, none hot beyond what they are today (NLL helper: per prompt token of a perplexity run only; pick: per
MTP step, over 41,489 + 8,192 rows on the host for Metal, which is a Metal cost, not a sharing cost).

Proposal: `engine/backend/host_logits.h` (`neg_log_p`, `clean_logit` re-export, `pick_argmax_prob`), `model/quant.h`
(`row_bytes(type, n)` built on `ggml_type_info`, plus `quantize_iq4_nl`), `engine/backend/rope_yarn.h` (add
`cos_sin(n_rot, pos, base, out)`), `model/qwen4exp.h` (`qsa_cells`).

### 1.6 Diagnostics and NaN reports

`nan::report` has callers only in R (`gpu/mega_serve.cpp:195-196`, `gpu/mega_select.cpp:130-135`,
`gpu/mega_mtp.cpp:77-87`, `gpu/prefill_moe_run.cpp:97-107`, `gpu/forward_moe.hip:88`, `mega_prefetch.cpp:244`).
Metal counts NaN router logits on the GPU (`metal/pass.mm:103`, `pass_layers.mm:165`) and returns the count in
`stats.nan_rows` (`backend.mm:78,151`), but never calls `nan::report` and never sets `stats.nan_reported`. AGENTS-level
rule in `nan_guard.h`: every NaN is reported. So the Metal backend is silent today where the reference prints a
warning once per request. Fix: after a pass, `nan::report("router logits", n, where)` in `Runner::run`. A shared
`report_router_nan(stats, n, layer, pos)` helper would also unify the three slightly different message texts in R
(`mega_select.cpp:132-137`, `prefill_moe_run.cpp:101-105`, `mega_serve.cpp:195`).

The rest of the diagnostics are per platform: Metal's `Profiler` (`metal/profile.*`, command buffer times),
R's `TONA_BAR_PROF`, `TONA_MTP_PROF` (`mega_mtp.cpp:190-209`), `gpu/mega_dump.*`, `trace_path`.

### 1.7 Expert cache, expert profile, selection (not duplicated yet)

Metal has none of this (`metal/AGENTS.md`: "No memory budget or expert streaming yet (M2)"). In the reference:

| Piece | File | Neutral? |
|---|---|---|
| Expert profile file (magic `0x50584549`, counts per layer and expert) load and write | `gpu/mega_experts.cpp:27-40` (+ `write_profile`) | yes (reads and writes a file) |
| Initial cache order: experts of a layer by profile count, stable | `gpu/mega_experts.cpp:42-72` (the HIP copy is one call) | order yes, copy no |
| ClassLRU replacement | `gpu/class_lru.h` (header says "Host code only") | yes, no includes of HIP |
| Router softmax and ranking with NaN rules | `gpu/route_rank.h` | yes |
| CPU cost estimate for the planner | `gpu/cpu_cost.h` ("Host side only, no HIP") | yes |
| Selection: top-k per token, union, OEA, slots, miss split | `gpu/mega_select.cpp`, `mega_select_cpu.cpp`, `mega_select.h` | logic yes, but it takes `Mega&` and `LayerCache&` and reads `g_.stats_`, `g_.pos_` |
| Pin budget from `TONA_RAM_GB` | `gpu/mega_experts.cpp:70-` uses `backend/budget.h` | already shared |
| Eviction credit, KFD counter, wait clock, GTT budget | `gpu/evict_counter.*`, `wait_clock.h`, `gtt*.h`, `guarded_copy.*` | AMD only |

Hot: selection runs per layer per pass while the kernel waits at the router (`serve_pass`, `mega_serve.cpp:157-200`).
A shared selection class must not add indirection or allocation per layer. Today `ExpertSelect` holds `std::vector`s
and is constructed per layer per pass (`mega_select.h:60-77`); that cost exists now. Making the logic
backend-neutral means passing an interface of a few plain structs (router logits pointer, cache state by reference,
stats by reference), not virtual calls per expert. Check with `bench-set` before and after, per `docs/backends/README.md`.

Proposal: move the pure pieces now (`class_lru.h`, `route_rank.h`, `cpu_cost.h`, profile read/write and ordering
into `engine/backend/expert_policy.h`), and split `ExpertSelect` into "decide" (pure, neutral, takes spans) and
"apply" (per platform, takes the decisions). Do it before Metal M2 starts.

## 2. Drift already visible

These are differences found by reading; none was run.

1. NaN reports: absent in Metal (1.6).
2. MTP vocabulary scan: R keeps `mtp_scan_` per slot (`SeqHost::mtp_scan`, `mega_seq.cpp:81`) and resets it on
   `load_state` and `reset` (`mega_state.cpp:84,200`). M keeps one `scan_` in the single `MtpVocab`
   (`mtp_vocab.h:32`), resets it only if it is larger than the history (`mtp_vocab.mm:68`), and `Runner::reset` and
   `load_state` do not touch it. With several slots, or after a reset to a longer sequence, tokens before `scan_`
   are not offered for a dynamic row. Effect: a lower MTP acceptance, not wrong output (the target verifies). Size of the
   effect: not measured.
3. `hlast_pos` after a fill: R sets `hlast_pos_ = pos_ - 1` unconditionally (`mega_mtp.cpp:269`), also when
   `mtp_h_of` returned "unknown"; M sets it only when h was found (`runner_mtp.mm:142-145`). Whether R can reach the
   unknown case there is unchecked; if it can, it would read a stale `mtp_hlast_` row instead of counting a stale
   entry. Needs a look with the shared code in hand.
4. `valid_row`: R maps an invalid pick to row 0 and reports (`mega_mtp.cpp:85-89`); M returns an empty pick
   (`runner_mtp.mm:90`). Both are safe; the behaviour differs.
5. `TONA_SEQS` default: 2 (R) against 1 (M). Intentional probably (memory), but it is two places with the same name.

## 3. What is right to keep per platform

- Everything that encodes work for the device: `Pass` and its layer functions (`metal/pass*.mm`), the megakernel
  program builder (`gpu/mega_build*.cpp`, `mega_op_*.h`), prefill drivers (`gpu/prefill_*`).
- Memory placement and transfer: pinned and GTT memory, io_uring row gathers, the guarded pipelined copy and its
  deadlines (`gpu/guarded_copy.*`, `backend/stall.h` is the neutral part: constants and `CopyStalled`), no-copy
  Metal buffers and page-aligned windows (`metal/host_math.h:76-98`, `weights.mm`), KV placement.
- Rollback mechanism: R commits lazily (`pending_commit_`, `parity_`: the kernel applies the rollback in the next
  pass), M (since 2026-11-12, a3a8f2a) replays only the DeltaNet recurrence of the m kept tokens, inside commit() instead of the next pass (`runner.mm:117-132`). Only the bookkeeping around it is shared (1.4).
- Concurrency model: R waits on the CPU inside the kernel (`serve_pass`) and needs the yield protocol and the
  eviction credit; M has no CPU waits inside a command buffer.
- Pass-cost tables (`metal/backend.mm:43-53`, the reference's in the server): measured per platform. Cosmetic point:
  the reference's table lives in `server/generator.h` as the fallback instead of in the reference backend's own
  `pass_costs()`; moving it there makes every backend treat the interface the same way, no runtime change.
- `kernel_type` map from ggml type to kernel name (`metal/weights.mm:13-19`): a table of the kernels that exist.
- Profilers of command buffers or megakernel steps.
- `runtime_env.h` is AMD/libgomp specific although it sits in `backend/`; harmless, but a CUDA backend should not
  inherit its assumptions (name it `hip_runtime_env.h` or leave a note).

## 4. Ranking

Ranked by lines removed or kept out, by the risk of the copy drifting (section 2), and by what a CUDA backend would
otherwise copy. Effort and risk are for the change itself; time is not part of the ranking.

| # | Merge | Lines in copies | Drift seen | CUDA saves | Effort | Risk to R |
|---|---|---|---|---|---|---|
| 1 | PLE rows (`model/ple.h`) | 4 x about 25 | none | all of it | small | very low: pure integer function, compare rows old/new on the mini model |
| 2 | MTP vocabulary map (`backend/mtp_vocab.h`) | about 70 in R and 75 in M | yes (item 2) | all of it | small to medium | low: ring order and `stats.mtp_dyn` must stay identical; one-time host loop per fill |
| 3 | Snapshot layout and slot bookkeeping (`snap_layout.h`, `SlotBook`) | about 4 + 4 formulas, 2 validators | no, but 4 copies inside R | stash/checkpoint arithmetic, validation | small (layout) to medium (SlotBook, `SeqHost` swap) | low for layout; medium if `Mega` adopts `SlotBook` (touches the members `pos_`, `tokens_`, ... used all over) |
| 4 | MTP tracking (`backend/mtp_track.h`) | about 45 in R and 60 in M | yes (items 3, 4) | the whole fill/step state machine | medium | medium: touches `SeqHost`, `mtp_h_of`, `mtp_fill`; per round integer work only, no speed risk if templated |
| 5 | Expert policy before Metal M2 (`class_lru.h`, `route_rank.h`, `cpu_cost.h`, profile I/O, split `ExpertSelect` into decide/apply) | about 1,300 lines in `gpu/`, none copied yet | n/a | the largest item for any backend that streams experts | medium to large | medium: hot (per layer per pass); `bench-set` before and after, TONA_DETERMINISTIC tokens identical |

Also worth doing in one pass, almost free: the small helpers of 1.5 and the Metal NaN report of 1.6 (a quality
fix for Metal today). `neg_log_p` (3 copies), `row_bytes` (3 implementations), `quantize_iq4_nl` to `model/`.

Order I would take: 1, then the helpers and the NaN report, then 2, then 3 (layout first), then 4 together with fixing
the drift items, then 5 before M2 begins. Rule 3 applies to every step: refactors of compute code must give
identical tokens; items 1 to 4 do not touch kernels, so the check is tokens identical with `TONA_DETERMINISTIC=1` on
the reference and `tona-check` on Metal.

## 5. Method note

Both sides were read in full for: `backend/metal/{seq,runner,runner_mtp,mtp_vocab,host_math,weights,profile,backend}`,
`gpu/mega_{seq,state,mtp}.cpp`, parts of `gpu/mega_serve.cpp`, `mega_build_mtp.cpp`, `mega_experts.cpp`,
`mega_select*.cpp`, `prefill_entry.cpp`, `forward.hip`, `cpu/ref.cpp`, and all of `engine/backend/*.h`. Metal's
`pass*.mm` were read only for host inputs (`write_inputs`, `write_positions`, router NaN counter); grouping of experts
per pass in Metal is a GPU kernel (`moe_group`, `pass_layers.mm:167-170`), so there is no host copy of it in either
backend (the reference groups by union on the CPU in `ExpertSelect::build_union`; the two are different algorithms
for different devices).
