# B1 (#163): what the fixed ~11.5 ms of a decode cycle are

Host Ryzen 9 9950X (changed from the 9600X of the older profiles), RX 9060 XT gfx1200, Coder IQ1_M, MTP head.
Branch `<worktree>`, build `work/build/w163` (main 0fd36c4 plus the A2/A3 changes and `TONA_HOST_TRACE`).

## Plan (written before the run)

**Question.** The cycle model of `/var/tmp/claude/r9700/tona-scaling.md` splits a 48.3 ms decode cycle (3.98 tokens,
42.1 ms verify pass, 4.4 ms draft, 1.8 ms other host) and calls about 11.5 ms "fixed": ~1,000 small steps, CPU round
trips, host time, the fixed part of the MTP steps. It is a model from older GPU profiles (2026-11-12-verify-pass.md,
TONA_MEGA_PROF, block 0 stamps) and says nothing about what the CPU does between the GPU's steps. Which host
segments exist, how long is each, and which are removable bit-exactly?

**Prior art read first** (technique, window 2025-2026): Hazy Research megakernel for Llama-1B (2025-05-27,
hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles: the next instruction's weights are loaded while the previous
one stores; per-counter dependencies instead of a grid barrier); MPK (arXiv 2512.22219: pre-load phase of the next
task overlaps the compute of the current one); TensorRT-LLM one-model speculative decoding (whole draft loop in one
graph, device-side feedback, no host round trip per draft step); vLLM Model Runner V2 (GPU-side prep consumes the
rejection-sampling result, no CPU sync, -6.3 % TPOT); LLVM gfx12 split barriers (signal, independent work, wait).
None states bit-exactness; all of them only move time, not the arithmetic.

**Scope.** One server start (port 8091, MTP, ctx 32768, built-in expert profile as in production), 3 prompts (2 code,
1 text) twice, 400 tokens each, greedy: the first three requests warm up and are skipped in the report. Two starts:
(1) `TONA_HOST_TRACE=<file>`: timestamps of the host side per cycle; (2) `TONA_MTP_PROF=1`: GPU time per step of the MTP
program. Expected duration: model load ~3 min, decode 6 x 400 tokens ~ 1.5 min per start.

**Why this size.** The question is a breakdown (means per segment), not a speed difference: ~100 cycles per request,
~300 after warm-up per start; a segment of 0.1 ms has a spread of a few percent over that many cycles. No ab.py round
count applies because nothing is compared; the trace adds ~150 marks (about 20 ns each) per cycle and no syscall
except one write per cycle, so it should not move the cycle by more than 0.1 ms (checked by the decode rate of
the traced run against the 92 / 67 tok/s of 2026-11-15).

**Expectation.** Cycle 48.3 +- 3 ms (code ~92 tok/s, text ~67 tok/s as in the reference; the 9950X may lower the host
parts). From the model: verify pass 42.1 ms of which host.wait round trips 48 x ~15 us = 0.8 ms + selection 0.6 ms,
draft 4.4 ms for ~3.5 steps (~1.15 ms each, 0.3 ms of it fixed), other host 1.8 ms. A host segment above 0.3 ms per
cycle that is not in this list is a finding.

**Active tooling.** `TONA_HOST_TRACE` (run 1) or `TONA_MTP_PROF=1` (run 2) only; no `TONA_MEGA_PROF` (its read-back
per pass would add to the host time); no `TONA_DETERMINISTIC`.

## Test conditions

`tools/hw_report.py` (2026-11-17 22:01, commit d56090f, clean tree): Ryzen 9 9950X, RX 9060 XT gfx1200, 60.5 GiB RAM,
kernel 7.3.4, hip 7.16. Quiet check before: warning "1.27 GiB VRAM held by others" and "1 page swapped in 1 s", load
average 1.0, CPU use by others 0.1 %; after: warning, 1.27 GiB VRAM held by others. Engine build `work/build/w163`
(= main 1eaa6d2 plus A2, A3 and the trace). Server on 8091 through the queue, kind exclusive. First attempt (before the
rebase) failed at start with a GTT out-of-memory (another program held host memory), the job was requeued; one job was
stopped by the queue and requeued (desktop guard); rounds were not affected. Raw files (not in the repository):
`/var/tmp/claude/163/prof1/` (host_trace.txt 86,738 marks, server logs, `report_skip250.txt`).

## Results

Requests (decode, 400 tokens, greedy, first round = warm-up): code 96.5 / 98.5 and 97.3 / 98.8 tok/s, 5.0 tokens per
pass; text 61.6 / 64.5 and 62.3 / 65.1 tok/s, 2.5 tokens per pass (decode rate with the trace on; the reference of
2026-11-15 was 92 / 67 on the old CPU, so the trace costs nothing visible). Second round only (251 cycles, mixed code and
text, mean cycle 43.8 ms, sd 11.5 because of the varying width), `tools/host_trace_report.py --skip 250`:

| segment (what runs) | ms per cycle | count per cycle | us each |
|---|---:|---:|---:|
| GPU: from the end of the CPU experts of layer l to the router of layer l+1 (rest of layer l, start of l+1) | 25.87 | 47 | 550 |
| CPU experts of the layer (GPU runs the hit experts meanwhile) | 10.01 | 48 | 209 |
| MTP draft steps, launch to synced (GPU kernel 907 us by TONA_MTP_PROF, plus ~70 us) | 3.85 | 3.93 | 979 |
| after the last router: GPU finishes layer 47 and head.out, logits copy | 2.27 | 1 | 2271 |
| CPU selection at a router (router_arrived to selected) | 0.53 | 48 | 11.1 |
| launch of the pass to the first router (layer 0 up to the router) | 0.47 | 1 | 474 |
| **host only**: verification after the logits (post) | 0.31 | 1 | 307 |
| **host only**: embeddings and PLE rows of the pass (stage_embeddings) | 0.30 | 1 | 300 |
| **host only**: MTP step staging and launch (mtp_begin to launched, synced to begin) | 0.11 | 3.9 | 29 |
| other marks | 0.04 | | |

Findings.
- The host-only share of the cycle is 0.7 ms (1.6 %), not the 1.8 ms "other host" of the cycle model (that figure was
  the remainder after subtracting the passes, so it contained GPU tails). A launch costs the host 13 us, a draft step
  35 us in total; the round trip of a draft step is not where the MTP fixed cost is.
- The MTP step is bandwidth-bound: 907 us in the kernel against 266 MB / 315 GB/s = 0.84 ms; the fixed part is about
  0.07 ms per step, 0.27 ms per cycle, not the ~1.3 ms of the model. Largest steps: mtp.head 155, mtp.qsa.in 143, mtp.eh 104,
  mtp.dyn 78, mtp.qsa.out 63, mtp.gate_up 59, mtp.down 33; ~20 steps below 25 us (about 0.2 ms together).
- The remainder of the fixed cost is inside the verify pass kernel (~1,000 steps), as the 2026-11-12 profile says: 36 ms
  of the 43.8 ms. Per layer 760 us of GPU time at this width. Host ideas ranked by what is measured here:
  1. in-kernel step latency (descriptor and first-weight loads after each grid barrier; Hazy "no bubbles", MPK pre-load):
     the only item that can reach ms; bit-exact by construction (loads move, arithmetic does not);
  2. selection 0.53 ms: 11 us per router at width 5.5, hidden behind nothing (the GPU waits in host.wait);
  3. embeddings 0.30 ms and post 0.31 ms: serial on the critical path, overlappable with the draft steps / logits copy.

## Next levers, with size estimates (derived, not yet measured; ms per 43.8 ms cycle at width ~5.5)

| Lever | What | Estimate | Basis |
|---|---|---:|---|
| 1. Step latency inside the verify kernel | ~1,000 steps per pass (~21 per layer); after each grid barrier a block loads the op descriptor, then the task, then the weights: three dependent memory latencies before the first byte of weights. Issue the loads of step o+1 (descriptor, first rows) during step o (Hazy "no bubbles", MPK pre-load); bit-exact, only load time moves | 1 to 2 ms (2 to 5 %); 1 to 2 us on each of ~1,000 steps | per-step fixed 8.7 us on 194 dense steps (cycle model, 2026-11-12 profile): the ceiling is that 1.7 ms plus ~0.9 ms for the expert steps; code generation of `k_mega` is sensitive (spills), so an experiment with a rollback |
| 2. Fewer steps per layer | fuse element-wise neighbours (mix, moe.act, moe.sum, dn.conv/norm, qsa.prep/score/merge) so a step and its barrier disappear; ~4 us per removed step (0.75 us barrier + ramp-up + tail) | 0.6 ms for 3 steps per layer (144 steps) | small steps cost 3 to 20 us each (MTP profile above); a fusion that changes a rounding step is not bit-exact, so each needs the golden check |
| 3. CPU selection at a router | 11.1 us x 48 = 0.53 ms on the critical path (the GPU waits in host.wait); faster selection (no allocation in `split_misses`, rank only what changes) | 0.2 to 0.3 ms | 8.9 to 11 us per router in this trace; floor near 4 to 5 us (3.2 us at width 1, 2026-11-02-mega-steps.md) |
| 3b. GPU routing for resident experts | all experts in VRAM (24 GB and larger cards): no CPU decision, no mailbox round trip; `OP_ROUTE` exists for the MTP head | ~1.0 ms on a card without CPU experts (host.wait 1.2 to ~0.15 ms per cycle, cycle model 3.6) | does not apply to the 16 GB reference, where 12 % of the experts are CPU or GPU-direct and need the host plan |
| 4. Host-only | parallel argmax rows (0.31 ms) and parallel embeddings (0.30 ms), branch `163-b1host` | up to 0.6 ms, expected 0.3 to 0.5 after the OpenMP wake-up | segments above |
| 5. GPU waiting for the CPU experts | after each router the GPU runs the resident experts while the CPU computes the rest (209 us per layer here); where the CPU is slower the GPU waits in host.y | 0.8 to 1.3 ms of waiting per cycle (cycle model); a tighter per-layer split moves part of it | planner constants and the measured CPU cost; a separate experiment |

Order by size and risk: 1 (largest, one experiment), 4 (built, measuring), 2, 3, 5. Only 3 and 4 are on the host side:
the 11.5 ms of the cycle model are mostly levers 1 and 2 and the small-step tail of the element-wise steps.
