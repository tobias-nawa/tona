# Ornith-1.5-35B-A3B on Tona: Rust and Go coding quality, paired against the Coder (2026-11-19, #143, #165)

## Plan (written before the run)

**Question.** On llama.cpp, Ornith solved 8/22 Rust exercises against the Coder's 2/22 (paired 6:0, p = 0.031;
`research/measurements/2026-11-19-rust-go-ornith-bolt.md`). That run had two confounds: llama.cpp applies `min_p` 0.05
by default and Tona has no `min_p` (research/71 row 1), so Ornith got a tail cut the Coder did not. Here both models run
on Tona with the identical sampler. The run also checks whether Tona's Ornith implementation reaches llama.cpp's quality
(an engine check).

**Decision (one per result).** Ornith on Tona is "recommended for <language>" if the paired net against the Coder is at
least +5 and the exact McNemar p is below 0.05 (fixed beforehand, research/71 checklist item 1). If Ornith-on-Tona is
clearly below llama.cpp Ornith on the same exercises, that is an engine finding to localize (N7 cross-check), not noise.

**Arms.** Ornith: `tona-server` (this branch, build `work/build/ornith-tona`), Q4_K_M file, **with its MTP head** (O4,
section below; the maintainer asked for it before the eval, like the Coder runs with its drafter), history drafts as
for the Coder, prompts through the decode kernel in passes of up to 4 tokens (no Ornith prefill kernels yet).
Speculation does not change the distribution of the output (the drafts are verified by the target with min(1, p/q) and
the residual), so the quality question is the same with and without it; it changes the speed and the random draws.
Coder: the existing run in `/var/tmp/claude/w170rg/coder` (Tona, IQ1_M, MTP). Reference for the engine check: llama.cpp
Ornith in `/var/tmp/claude/w170rg/ornith` (Q4_K_M, q8_0 KV, `--reasoning off`, effective `min_p` 0.05), stopped by hand
after 9 of 39 Go exercises, so the Go comparison with it covers those 9 only.

**Settings identical to the Coder arm.** Sampling `{"temperature":1.0,"top_p":0.95,"top_k":20,"seed":1165}`, 4 attempts,
`--max-tokens 16384`, server `--ctx 65536`, 40 min wall cap per exercise, the harness `bench/polyglot` of the main
checkout at commit `6ec3df5` (parser fallback for replies without a FILE header; read-only; the worktree that held the
uncommitted fallback during the Coder run was merged and removed, the committed code is the same file), the driver
`/var/tmp/claude/w170rg/driver.py` (copy with the path of the harness changed) with its
futility stop (net < +2 at half of a language stops that language; reported as "stopped at n/N"). Thinking off: the
request carries no thinking field, Tona's server default is off (`--thinking` not given), and the Ornith chat template's
generation prompt for thinking off is `<think>\n\n</think>\n\n`, which is what Tona's Qwen3 format renders (checked against
the template inside the GGUF). Same as the Coder run. Not sent: `min_p` (Tona has none), `repeat_penalty`.
Order: all 22 Rust, then 39 Go.

**Scope.** Rust 22 exercises, Go 39, 4 attempts each, one sample per exercise. Power: with n = 22 the rule needs 6 wins
to 0 losses; a null result means "not shown". Expected duration (from the smoke test below: about 5 min for an exercise
that failed all 4 attempts at about 3,000 tokens each): Rust about 2 h, Go about 3 h.

**Expectation (with tolerance, source).** Ornith-on-Tona Rust pass within 4 attempts 8/22 plus or minus 3 (llama.cpp
8/22, Wilson 20 to 57 %). Paired net against the Coder +6 plus or minus 3. Go: no sound expectation, only 9 llama.cpp
exercises exist (7 of 9 passed, the first 9 in the order); I expect 30 to 60 % of 39. Speed (by-product, not an `ab.py`
measurement): decode 110 to 150 tok/s per request with the MTP head and history drafts (smoke with the head: 113 to 150;
without: 63 to 89), prefill 180 to 200 tok/s (1559 tokens in 8.1 s, unchanged by the head). A result outside these is a
finding.

**Active tooling.** No `TONA_DIAG`, no profile file for the experts (none exists for Ornith), no draft log, MTP head on
(`TONA_NO_MTP` unset), default environment otherwise; server under `systemd-run --user --scope -p MemoryMax=54G -p MemorySwapMax=0` on port 8092.
Queue job `--kind gpu`. Test conditions (`tools/hw_report.py`, `tools/quiet_check.py` before and after) are in the
output directory and below.

**Checklist (research/71) answers.** 1 plan first: this section. 2 same prompt, parser and sandbox for every arm; harness
not committed in its worktree (dirty), so the commit is named below and the sandbox image digest is not pinned (research/71
row 10). 3 parser: smoke replies had `files_parsed` = `src/lib.rs` in all 4 attempts. 4 sampling: every field the Coder
run sent is sent; the effective values are Tona's request values (no hidden defaults beyond `min_p` absent). 5 thinking
off, sampling from the thinking-mode card rows (research/71 row 2; same for both arms). 8 rates named below. 9 per arm:
truncations, no-file replies, loop/timeouts, early stops reported below. 10 ten raw replies read, below.
12 not comparable to published numbers: own prompt, 4 attempts, Q4_K_M vs IQ1_M.

## O4: the MTP head of Ornith (this branch)

The Ornith GGUF carries its MTP layer (`blk.40`: `nextn.eh_proj`, `enorm`, `hnorm`, `shared_head_norm`, a gated
attention layer and a MoE with 256 experts, Q4_K gate and up, Q6_K down). `engine/gpu/arch/qwen35moe/mtp.cpp` builds it as
the third program of the same kernel: embedding of the token and h of the target go through `OP_MTP_IN` and eh_proj, then
the attention layer with its own int8 KV, routing on the GPU (`OP_ROUTE`), all 256 experts in VRAM (0.52 GB), the shared
expert, `shared_head_norm` and the full LM head of the target (the Coder's truncated head is the business of its scorer;
the full head costs about 0.4 GB per draft step). The host side (`mega_mtp.cpp`, slots, state) is the shared one; it now
asks `mtp_on_` instead of the Coder's `MtpHead` pointer. The server names no architecture: the model declares its
drafter files (`ModelBase::drafters()`), and the generator uses whatever head the backend reports (`has_mtp()`).
`TONA_NO_MTP=1` skips building the head (used for the comparisons below).

Checks (section 6 of `docs/architecture/model-architectures.md`):

- C1 machine code (`scripts/isa-diff.py`, baseline build of the commit before this work, `1db5a2d`): only the Ornith
  kernels changed (the `k_mega<.., Qwen35MoeProfile<..>>` instantiations and `op_fattn_prep<Ornith config>`), every
  Coder kernel is identical. The Ornith kernel now carries the MTP ops: vgpr spills 59 to 70 at width 8 (47 to 40 at
  width 1), sgpr spills 150 to 237 at width 8 (83 to 80 at width 1). Not timed separately.
- C3 Coder goldens on this build: `scripts/golden.sh compare` (csv, rust, explain) identical; `scripts/golden-server.sh
  compare` (all cases: both APIs, streaming, tools, sampling, thinking, reuse, ssd, hit) identical.
- C6 `just check` green including `arch_isolation` and clang-tidy (23 files); unit tests 67 of 67.
- N4, greedy under `TONA_DETERMINISTIC=1`, with against without the head: three short chat requests (code and prose), all
  three identical to the character; the first three attempts of a polyglot exercise (Rust, `react`: 2.2k to 2.5k tokens
  each at up to 11k tokens of context, 3.7 to 4 tokens per pass with history drafts) identical. The fourth attempt
  differed from the eighth token on; the prompts of that attempt were cut differently (948 against 949 prompt tokens
  computed, 10,731 against 10,730 reused). Followed up in the next section: it is not the head.

- Acceptance (server log, "tokens per pass", greedy, code prompts): 3.0 tokens per pass without history drafts (about 2
  accepted drafts per pass), 2.1 on a prose prompt; with history drafts on retries up to 7.5 tokens per pass.

Prior art:

- [llama.cpp PR 29761](https://github.com/ggml-org/llama.cpp/pull/29761), status: merged. MTP for the Qwen3.8 architecture
  with its own block, full trunk LM head, own KV. Adopted: the shape (input `[enorm(e) | hnorm(h)]`, one layer with own KV, the
  target's head), the Coder's program in `arch/qwen4exp/mtp.cpp` as the template. Not adopted: the Coder's QSA block and
  hyper-connection streams (Ornith has plain gated attention and one stream).
- [llama.cpp PRs 31031 and 31037](https://github.com/ggml-org/llama.cpp/pull/31031), status: open. Trimmed draft head
  (32,768 rows instead of 248,320): +2 to +9 % tok/s for 1 to 5 % lower acceptance. Not adopted yet: the first version uses
  the full head; this is the next tuning item (the Coder has its own list and scorer).
- [llama.cpp PR 27210](https://github.com/ggml-org/llama.cpp/pull/27210), status: open. Adaptive draft depth. Not adopted: the
  generator's calibration of the width is the shared one.
- [FastMTP, arXiv 2509.18362](https://arxiv.org/abs/2509.18362), status: read earlier. One head used recursively: acceptance
  falls over steps 1 to 3 without tuning. Yardstick for the acceptance of a head that was not tuned for chaining.
- [vLLM issue 37435](https://github.com/vllm-project/vllm/issues/37435), status: open. The MTP layer did not get the target's
  YaRN override (acceptance 0 above the native length). Ornith's native context is checked by the shared start code
  (`rope_base` of the target; the MTP layer shares it, same attention config).

Search: tools/papers/scan.py topics v1; topics: speculative; terms: MTP, multi-token prediction, drafter; window:
2026-02-19..2026-11-18 (9 months, arXiv only; 573 hits, 9 not in papers.md, none about wiring a built-in head), plus
research/papers.md and papers-screened.tsv (the entries above) and the Coder's program as the template. GitHub was not
re-scanned (read from `research/papers.md`).

## Smoke test with the head (queue job `ornith-tona-smoke-mtp`)

First attempt (before the head, no drafter): the server loaded the file and then failed the first request with
`gpu::Mega: no MTP head` (the server required a drafter option at start, and the generator fed an MTP head that does
not exist). Fixed on the branch (commits `5504b33`, `2c626ab`: the model declares its drafter files, `has_drafter()` in
the generator); logs in `/var/tmp/claude/w170rg/ornith-tona/attempt1`. Without the head the numbers were (kept in
`nomtp-smoke`): decode 63 to 79 tok/s over one exercise (1.00 tokens per pass), prefill 1559 tokens in 8.0 s (195 tok/s),
3739 in 20.9 s (179 tok/s). With the head:

- Load 4.7 s (page cache warm), VRAM 13.94 GB used by the server (14.00 without the head), 154 expert slots per layer (162
  without), 20.4 GB pinned host memory, KV int8 0.75 GB in RAM at ctx 65536 (the head's own KV, 0.19 GB at this context, is
  not in that figure).
- Desktop VRAM held by others before the run: 1.31 GiB of 15.9 GiB. Quiet check before: warning, VRAM 1.31 GiB held by
  others and a little swap activity (5 pages in 1 s); measured anyway.
- One exercise (rust/react, 4 attempts): prefill 1559 tokens in 8.1 s (192 tok/s), 5011 in 29.3 s (171 tok/s), 466 in
  3.7 s, 221 in 2.0 s; decode 146, 150, 137, 127 tok/s per request at 4.9, 7.2, 7.5, 7.7 tokens per pass (the retries
  repeat most of the previous code, so history drafts hit). Not an `ab.py` measurement.
- Four attempts of an exercise use under 20k tokens of conversation, so no exercise comes near the 40 min cap: no stop for
  kernel work.

## Is the difference in attempt 4 the head? (job `ornith-split-test`)

Question: does a prompt that reaches the model by another route (turn 2 after a decoded reply that the cache holds, "warm",
against turn 2 computed from nothing, "cold") change a greedy answer, with and without the head, and does the Coder do the
same? Setup: a 15.5k-token conversation (three source files and a question), turn 1 limited to 120 tokens, turn 2 asks for
a list (300 tokens), `TONA_DETERMINISTIC=1`, one server start per request path. Stop rule: this one pair of requests per
arm and nothing more. Result (answers compared as text):

| pair | result |
|---|---|
| Ornith with head, warm, against Ornith without head, warm | equal |
| Ornith with head, warm, against Ornith with head, cold | equal |
| Ornith without head, warm, against Ornith without head, cold | **differ from character 49** |
| Ornith with head, cold, against Ornith without head, cold | differ from character 49 (same as the line above) |
| Coder (its MTP), warm, against Coder, cold | **differ from character 288** |

Reading: three of the four Ornith runs agree, and the head is not in the odd one (the run without it is). The answer
depends on how the same prompt reached the model, for the Coder as well. That the head produces a different answer is
excluded by the first row (identical, 553 characters) and by the three identical attempts above; why the cold run
without the head differs from the others is **not established**. Candidates, not tested: which experts sit in the VRAM
cache (154 slots with the head, 162 without, and the cache history differs between warm and cold), and how the prompt is
cut into passes. Either is rounding that depends on the path, not a drafting error. Decision: nothing to fix in the head;
the open question (which of the two causes) is a finding for the engine check, to be settled with the layered prefill
(its cut and expert residency are fixed by design).

## O5 (server, chat): what was found

`tests/chat_vs_jinja.py` against Ornith's own GGUF template: 4 of 9 cases were equal with the Coder's renderer. The Coder's
format adds "Reasoning effort is set to ..." lines to the system message when thinking is on; Ornith's template has none.
New `ChatFamily::Qwen35` (the same renderer without those lines), `tona-chat --model FILE` takes the family from the model,
and the gate runs for the Ornith file too (`chat_vs_jinja_ornith`, `chat_tokens_ornith`): 9 of 9 byte-equal. The thinking-off
prompt the evals use was equal before, too. `--list-backends` shows `qwen35moe` (tier 2, F32 Q4_K Q6_K). `general.sampling.*`
of the GGUF (T 1.0, top_p 0.95, top_k 20) equals the Coder's GGUF and the server's thinking-on table, so using it would
change nothing for thinking on and would replace the thinking-off table (0.7 / 0.8 / 20) for both models: not done, a
decision for the maintainers. No catalogue entry: the licence note in `research/42-model-licenses.md` says the linked
`LICENSE` file is missing and the base model is not stated.

## Results: Rust on Tona, stopped by hand at 12 of 22 (incomplete)

The coordinator stopped the run at 11:50 on the maintainer's instruction (the direction is clear; the quality decision
comes from one agentic eval on the finished implementation). Go was not run. Not a verdict under the fixed rule (needs all
22). Ornith on Tona with the head, 12 exercises, 39 attempts, 1 truncated reply, no reply without a parsed file, no
loop or timeout:

- Pass within 4 attempts: 4 of 12 (Wilson 14 to 61 %); within 1 attempt: 1; within 2: 4 (acronym, say, grade-school in
  attempt 2, nucleotide-codons in attempt 1).
- Paired against the Coder on the same 12: only Ornith 4, only Coder 0 (exact p = 0.125, net +4). The Coder passed none of
  these 12.
- Paired against llama.cpp Ornith on the same 12: only Tona 3, only llama.cpp 2 (p = 1.0); llama.cpp passed 3 of the 12
  (grade-school, book-store in attempt 4, variable-length-quantity), Tona 4. No sign of a quality drop of the Tona
  implementation against llama.cpp on this sample; the sample is too small to say more. The sampler differs (llama.cpp
  with `min_p` 0.05, Tona without).
- Test conditions: `hw_report.full.txt` and `quiet_before.full.txt` in `/var/tmp/claude/w170rg/ornith-tona/`: 1.29 GiB
  VRAM held by others, 1 % GPU busy, quiet check warning (the VRAM held), measured anyway; commit `cee9d52` (clean
  tree). No quiet check after (the job was stopped from outside). CPU builds ran beside the run (a second build
  directory for the chat and gate work): this was a quality run, so allowed by the queue rules; its decode and prefill
  numbers below carry that caveat.

### Speed by-product (not an `ab.py` measurement; with the MTP head and history drafts, Q4_K_M, KV int8, ctx 65536)

From the server log of this run, 41 requests with at least 200 output or 200 computed prompt tokens, one request per attempt
(prompts 1.5k to 34k tokens, the longest answers hit the 16,384-token limit):

- Decode (tokens generated per second of generation, per request): median 113.5 tok/s, p10 81, p90 178, min 39 (a
  16,384-token answer at 34k context), max 227. Without the head, in the smoke test and in the identity runs: 63 to 89 tok/s.
- Prefill (computed prompt tokens per second, prompt through decode passes of up to 4 tokens, no prefill kernels): median
  173 tok/s, p10 114, p90 193, min 82 (16k tokens computed at 18k to 34k context).
- Slots per layer 154, VRAM 13.94 GB used by the server, 20.4 GB pinned, desktop VRAM held by others 1.29 GiB.

## Prefill kernels for Ornith: where we stand (not implemented in this ticket)

The ceiling: prefill is compute bound (`docs/backends/reference-platform.md`: WMMA f16 93 TFLOPS measured). The model has about 3 B
active parameters, so 6 to 7 GFLOP per prompt token: 93 TFLOPS / 6.5 GFLOP = about 14,000 tokens per second as the
arithmetic ceiling, a realistic target half of that (WMMA efficiency of the Coder's prefill, to be taken from its
measurements). Streaming the experts is not the limit: one layer-major sweep moves the 20.4 GB that are not in VRAM at most
once, 0.4 s at 55 GB/s, spread over the tokens of a group. Where we are: 173 tok/s (median, above), that is about 1.2 % of
the ceiling; the gap is a factor of 40 to 80. A 20k-token conversation costs about 2 minutes of prefill today and would
cost 3 to 5 seconds at the target.

What the existing prefill driver offers and what is missing (read, not changed):

- `PrefillSequence` is the per-architecture seam; the Coder's is `arch/qwen4exp/prefill_layer.cpp` (201 lines).
- Still Coder-shaped, not yet behind that seam: `PrefillRun` takes `model_as<Qwen4ExpModel>` in its constructor and in
  the plan (`pf_dims(const Hparams&)`, `prefill_plan.cpp:168`), its buffers (`PfBufs`: hyper-connection streams, the
  injections, the drafter features, PLE), `head()`, `finish()` and the MTP entries of the prompt (`prefill_mtp.cpp`, QSA
  MTP head), and `Mega::pf_[l]` descriptors that only the Coder's program fills for the dense matrices.
- Reusable for Ornith: the DeltaNet prefill (`prefill_dn.hip`; the output gate differs: silu), `prefill_fattn.hip` for the
  unindexed gated full attention (F1; written for exactly this, not yet run against the oracle), `prefill_res.hip` (F2),
  the shared MoE streaming (`prefill_moe_run.cpp`: router, expert batches, double buffers). To check: the dequantizing GEMM
  (`gemm_dequant.hip`) has instantiations for the Coder's types; Ornith's Q4_K gate and up and Q6_K down matrices of the
  experts and the dense Q4_K / Q6_K weights may need `k_dequant` and the expert GEMM for types 12 and 14.
- Work breakdown (each step testable alone): (1) `pf_dims` and the workspace measure behind `PrefillSequence`; (2) Ornith's
  plain-stream phase buffers and `PfDeltaNet` / `PfGatedAttn` / `PfMoe` descriptors from `Qwen35MoeProgram`; (3) the layer
  class; (4) types 12 and 14 in the dequantizing GEMM; (5) head and the MTP entries without the Coder's kernels (the
  KV-only MTP program already catches the prompt up); (6) checks: N2 layer against the oracle at 128 tokens, 4K, 32K once,
  N7 against llama.cpp-hip, decode after a kernel-prefilled prompt against decode-prefilled (rule to state first: the
  same top-1 on the whole answer under a fixed split, deviation as the Coder's prefill golden allows). The shared files it
  touches (all in `engine/gpu/`): `prefill_run.h/.cpp`, `prefill_plan.cpp`, `prefill_entry.cpp`, `prefill_mtp.cpp`,
  `gemm_dequant.hip`, `mega.h` (descriptors); the Bolt line shares `gemm_dequant.hip` and the prefill driver.

Prior art for the prefill (not searched afresh; `research/papers.md` has the entries the layered prefill already cites):
layered prefill arXiv 2510.08055, FreeToken 2608.16157, MoE-CORE 2610.01950 (the driver's design); chunked gated-delta-rule
prefill is the Coder's `launch_dn_wave` already.

## Gates on the final tree (commit before this note)

`just check` green (pre-commit hooks, `arch_isolation`, openmp gate, clang-tidy on 28 files); 67 unit tests green plus the
two Ornith chat tests; `scripts/isa-diff.py` against a build of `1db5a2d`: every kernel identical except the Ornith kernel
instantiations; Coder goldens (`golden.sh compare`, `golden-server.sh compare`, all cases) identical on the final build.
