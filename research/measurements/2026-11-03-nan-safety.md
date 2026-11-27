# NaN safety of the argmax and top-k paths (#102)

Date: 2026-11-03. Origin: #72, where one NaN router logit in the prefill left the start index `1 << 30` of a maximum
unchanged and the scatter addressed 4 GiB away (page fault in `k_moe_sum`).

## Plan

- Scope: every place that forms a maximum or a top-k over values that come from the model (logits, scores) and uses
  the result as an index. Found by `grep` for `max_element`, `partial_sort`, `nth_element`, `sort`, `1 << 30`,
  `-INFINITY` start values and `__shfl_xor` argmax reductions, then read one by one.
- Rule: NaN counts as the smallest value, +-inf as the largest and smallest finite value (`ie::nan::clean_one`,
  `logit_clamp` on the device); finite values stay bit-identical. Every NaN is counted and reported once per request
  (`ie::nan::report`), never only under `IE_DIAG`.
- Tests: CPU paths as unit tests (`test_nan_guard`, `test_sampling`), GPU paths as ctest label `gpu`
  (`gpu_route` for the prefill router, `gpu_mtp_pick` for the MTP head).
- Bits: all goldens identical (`scripts/gpu-status.sh --own --build work/build/w100`).
- Speed: only the megakernel hot path changes (`op_host_wait`, `op_route`, `op_mtp_pick`). Expectation: within the
  noise of `scripts/ab.py` (decode +-0.5 %), because the additions are one wave reduction per router step and no new
  barrier. Plan of the runs is written before they start (section "Speed").

## List

"Memory/UB": can the site violate memory or invoke undefined behaviour with NaN or inf in its input?
Behaviour with a whole row of NaN, one NaN, +inf, -inf, before the fix.

| # | Site | Before | Memory/UB | Fix |
|---|---|---|---|---|
| 1 | `gpu/prefill_moe.hip:15` `route_best` (prefill router) | NaN: all comparisons false, `be = 1 << 30` (fixed in #72) | yes | `pf_clean_logit`, counter `kRouteNanWord` (#72) |
| 2 | `gpu/mega_op_mtp.h` `mtp_pick_part` / `mtp_pick_merge` (MTP draft) | NaN row never wins: block with only NaN returns row `n`, merge start `1 << 30`; the host indexes `mtp_vocab_[r]` with it. +inf: `exp(inf - inf)` = NaN sum | yes (host read outside `mtp_vocab_`, token outside the vocabulary) | `logit_clamp`, first valid row always wins, NaN count in bit 30 of the part row and in `out[3]`; host `valid_row` |
| 3 | `gpu/mega_op_moe.h` `route_softmax`, `route_wave_topk`, `route_union` (MTP router on the GPU) | NaN: `v > -2` false, `be = E` (or an unused slot >= E); `u_of[be]` and `slot[nu]` write outside LDS. +inf: NaN exponentials | yes (LDS) | clamp before the softmax; NaN count per wave into the pick buffer |
| 4 | `gpu/mega_op_wait.h` `op_host_wait` -> `gpu/mega_select.cpp:131-137` (`rank_token`), `:169`, `:187` (target router, CPU side) | NaN: `max_element` NaN, `p` all NaN, `partial_sort` with a comparison that is no strict weak order | yes (UB in `partial_sort`) | the GPU clamps the router logits and the prediction before the copy into the mailbox, counts NaN into `nan_router`/`nan_pred`; the CPU reports (`mega_serve.cpp`). `mega_select.cpp` is not touched, see below |
| 5 | `gpu/mega_select.cpp:256` (`predict_stats`, IE_PREDICT_STATS only) and `gpu/mega_prefetch.cpp:143` (`vote_predicted`) | same as 4, comparator `r[a] > r[c]` | yes (UB) | the input is clamped by the GPU (4); `vote_predicted` also compares through `clean_one` |
| 6 | `gpu/mega_prefetch.cpp:243` (CPU-computed predicted router) | same | yes (UB) | `nan::clean` before the sort, report |
| 7 | `gpu/mega_select_cpu.cpp:174` `choose_fill` (sort by weight sums) | the weights come from 4; NaN only if 4 is not clean | yes if 4 failed | covered by 4 |
| 8 | `server/sampling.h` `make_dist_rows`, `sample` | NaN row: no candidate, `d.tok` empty, `sample` reads `d.tok[0]`. +inf: `exp(inf - inf)`, NaN distribution | yes (empty vector) | `clean_one`, first valid token as fallback, report |
| 9 | `server/sampling.h` `argmax` (greedy; `verify.cpp`, `generator.cpp`) | `max_element` returns the index 0 for a leading NaN, silently | no (index valid) | NaN skipped, report |
| 10 | `cpu/dflash.cpp:284` top-k of the selector | `upper_bound` with an inconsistent order; index valid, `cand` filled | no | row cleaned before, report |
| 11 | `cpu/dflash.cpp:188` attention softmax | NaN propagates, no index | no | none (value only) |
| 12 | `cpu/ref.cpp:326` block scores, `:409`/`:416` router | `partial_sort` with NaN: UB | yes (UB) | `nan::clean`, report |
| 13 | `gpu/forward_moe.hip:86-94` router, `gpu/forward.hip:252` block scores (class `Forward`) | same | yes (UB) | `nan::clean`, report |
| 14 | `tools/ie_ref.cpp:71,83` | `max_element`, `partial_sort` | yes (UB) in the tool | `argmax`, comparator through `clean_one` |
| 15 | `gpu/topk_dev.h`, `gpu/mega_op_topk.h`, `gpu/prefill_topk.hip` (QSA block selection) | radix selection on the bit pattern as `uint32_t`: a total order, the number of cells is exactly `n_top` | no | none |
| 16 | `gpu/mega_op_idx.h` (indexer scores) | `fmaxf(sum, 0)` removes NaN heads; input of 15 | no | none |
| 17 | `tools/ie_gpu_gen*.cpp`, `ie_gpu_gen.h` `argmax` (`max_element`) | index valid, result may be wrong | no | none (test tools) |
| 18 | `gpu/mega_serve.cpp:27`, `mega_abort.cpp:82`, `mega_extra.cpp:83`, `prompt_cache.cpp`, ... | sorts of times, sizes, file names | no | none |

Site 4: `gpu/mega_select.cpp` itself (`rank_token`, `:169`, `:187`, `:256`) is done by #101 (`gpu/route_rank.h`, `Stats::nan_rows`, merged locally in 9563c46); this branch does not touch it. The GPU clamp below makes the input safe in addition.

Site 4 is the main decode path (target router -> CPU ranking). `engine/gpu/mega_select.cpp` is not changed by this
branch: the clamp happens before the data reach it. A defence there (the comparator through `ie::nan::clean_one`, and
`rank_token` calling `ie::nan::clean` on `r` before `max_element`) is three lines and left to whoever owns the file
for #101.

## Speed

Plan (written before the runs). Changed hot path: the megakernel (`op_host_wait` at every router step: clamp of the
copied logits and one wave reduction; `op_mtp_pick`, `op_route`: clamp plus one block reduction per step). The
prefill, the weights and the memory layout are untouched.

- Scope: `scripts/ab.py --a merged --b w100 --workload short,set --env IE_DETERMINISTIC=1`. `--a` is the build of
  main 9563c46 (the merge of #101, own build in a worktree). `short` measures decode of the plain target path (router step
  every layer, 2 requests, decode@2000), `set` the production path with the MTP drafter (pick and route per draft step,
  decode-code and decode-text). Prefill is not changed by this branch, so it is not measured for speed; the SHA of
  every text must be equal in both arms (bit-exact).
- Expectation: decode within -0.5 % .. +0.5 % (the noise table of `bench/noise.toml` for `decode@2000`: sd 0.23 %,
  `decode-code`: sd 0.07 %). Source: the added work per router step is ~16 additional ALU operations per thread and
  one wave reduction (a few hundred ns of ~1 ms per layer step is the upper bound, i.e. < 0.1 %), and the MTP pick adds
  two block reductions per draft step (~1 us of ~100 us).
- A result outside the range is a finding, not noise.
- Active tooling: `IE_DETERMINISTIC=1`, no `IE_DIAG`, no profiler. No build during a measurement.
- ISA: `scripts/isa-diff.py` between the two builds. Expected: only `k_mega` differs (the changed steps); all
  other kernels identical.

### Result (2026-11-03, `scripts/ab.py`, A = main 9563c46 in its own worktree build, B = this branch)

Six rounds, alternating AB/BA, `IE_DETERMINISTIC=1`. Raw report: `work/ab/2026-11-03-nan-safety-ab.md` (not in git).

| Metric | A median | B median | B vs A (mean) | Expected | Verdict |
|---|---|---|---|---|---|
| prefill@2000 | 1349.3 | 1348.5 | -0.01 % | -1..+1 % | within |
| decode@2000 (target path) | 70.6 | 70.4 | -0.26 % | -0.5..+0.5 % | within, all six rounds negative or 0 |
| prefill@2000:write | 1570.5 | 1569.8 | -0.04 % | -1..+1 % | within |
| decode@2000:write | 96.3 | 96.0 | -0.23 % | -0.5..+0.5 % | within |
| decode-code (MTP) | 85.5 | 85.3 | -0.19 % | -0.5..+0.5 % | within |
| decode-text (MTP) | 68.7 | 68.5 | -0.22 % | -0.5..+0.5 % | within |

SHAs of both arms are equal and stable; expert hits and misses are identical. The decode shift is small but
systematic (same sign in nearly every round): the cost of one clamp and one wave reduction per router step plus the
changed register allocation of `k_mega` (below), about 0.2 % of decode. It is inside the plan and below the 2 % that a
stability fix may cost (AGENTS.md rule 1 is about quality and speed of the reference; the owner decides).

ISA (`scripts/isa-diff.py`, A against B): code differs in `k_mega<1|4|8>`, `op_host_wait`, `op_route`, `op_host_y`
(register allocation only); LDS (`group_segment_fixed_size` 32024) and VGPR count unchanged; SGPR spills of
`k_mega<4>` 117 -> 179, of `k_mega<8>` 247 -> 179 (VGPR spills 181 -> 141). All other kernels identical.

`scripts/gpu-status.sh --own` names two more checks for a megakernel change ("long-context run; pressure series"): not
run here, left to the coordinator.
