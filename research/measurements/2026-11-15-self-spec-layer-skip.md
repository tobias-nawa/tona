# Layer-skip self-speculative drafting, offline acceptance (#155)

Written before any run (AGENTS.md rule 2). Results are appended below the line "Results" and read against this plan.

## Question

Can a draft pass of the Coder with layers skipped (or fewer experts per layer) predict the full model's greedy tokens
well enough, at a cost low enough, to raise decode tokens per second with identical output, (a) alone, (b) as an
extension behind the MTP chain? Background and prior art: [research/56](../56-self-speculative-layer-skip.md).

## What is measured and how

`tona-layer-skip` (CPU reference, `engine/tools/tona_layer_skip.cpp`): the full model generates greedily from a prompt;
every `stride` tokens the draft pass of each mask drafts `depth` tokens from there with the true tokens as prefix, from
a snapshot of the sequence state, which is restored after. Output per start and mask: for each depth i whether the draft's
argmax equals the full model's token (match), the draft's top-1 probability and -log p of the true token. The accepted
length of a chain is the leading run of matches; `S_j` = fraction of starts with j leading matches; the extension after
an MTP chain of k is the fraction with matches on depths k+1..k+i. This is exact for greedy decoding (the true tokens
are the full model's own output). Deterministic counters: match flags, no timing needed. The CPU reference computes the
arithmetic the GPU kernels are checked against; the draft passes run the same code with a layer mask.
`tools/self_spec_model.py` turns survival curves, the layer byte costs (`tools/layer_cost.py`) and the engine's pass
costs (generator.h defaults: pass 23.7 ms, widths 2..8 28.9 to 52.6 ms, MTP step 1.15 ms, fill 1.3 ms) into tokens per
second against the MTP-only baseline.

## Decision (before the numbers)

Reference platform and 32 GB class (both pass costs are bandwidth/expert-bound; the draft step is priced by bytes, the
skipped layers' expert reads included):

- **Propose the engine design** (stop for review) if the model predicts, for some mask, a gain of **at least +3 %** tokens
  per second over MTP-only on code (the focus workload) with the lower end of a bootstrap 95 % interval (by start
  position) above 0, and no loss above 1 % on text. 3 % because the build needs a scratch-state path, a second draft
  graph and verify-side snapshots; below that the complexity does not pay, and AGENTS.md rule 1 (reference never worse)
  would have to be argued for a gain that small.
- **Close as negative** otherwise, with the measured curves and the break-even table (the cost at which a mask would
  pay), so the question is not asked again for this model. A mask with cost fraction above 0.5 is not expected to pass
  (research/56, arithmetic): a draft step must cost clearly less than the ~9.6 ms the engine spends per token on code.
- The Mac and the 16 to 32 GB offloaded classes get the same model with their pass costs (SSD-bound passes: draft
  step cost = kept fraction of cold experts); a mask that fails on the reference and passes there is reported, not built
  (the ticket's decision is "decode rises on the reference or the 32 GB class").

## Scope

0. Timing run: code00, no masks, 40 generated tokens, `--cos`. Purpose: seconds per full step on this machine, layer
   change statistics; sets the size of the later runs (no claim).
1. Layer ranking: `--cos` on the calibration prompts code01, code03, code05, text00, 96 generated tokens each (the
   residual change per layer is a by-product of the full steps). Calibration set only.
2. Masks on the calibration set, `stride 8`, `depth 6` (6 starts x 4 prompts at most 12 per prompt):
   `topk4`, `topk2` (all layers, fewer experts); `u25` (layers with l mod 4 = 1 skipped, 12 layers) and `u50` (l mod 4 in {1, 2} skipped, 24 layers); QSA layers (l mod 4 = 3) kept;
   `cos8`, `cos16`, `cos24` (skip the 8/16/24 layers with the smallest residual change); `qsa-only` (keep only the 12
   QSA layers); `dn-only` (the Component-Aware draft, expected worst); `cos16+topk4`. A greedy backward elimination
   with acceptance as the objective is **not** run on the CPU path (48 candidates per step x seconds per token); the
   ranking by residual change replaces it, and the best two masks may get a one-layer swap round if the numbers ask for it.
3. Held-out evaluation of the best 3 masks plus the MTP-relevant depth curve on code00, code02, code04, code06, text01,
   text02 (never used in 1 and 2). These numbers decide.
4. MTP survival of the baseline: one `TONA_DRAFT_LOG` run of the bench-set on the production build through the queue
   (per-depth conditional acceptance), as input to the model; not repeated.

Size: starts are spread by `stride`; the run is as small as the confidence interval needs: held-out starts per mask
at least 60 (6 prompts x ~10 starts) for a +/- 0.12 interval on a survival of 0.5. If the interval is wider than the
decision margin the run is extended once, not repeated blindly.

## Expectation

Source and tolerance (each a prior to be overturned, not a target):

| Mask | Expected depth-1 match | Expected survival S_4 | Source |
|---|---|---|---|
| `topk4`, `topk2` | 0.6 to 0.85 (tolerance +/- 0.2) | 0.15 to 0.45 | S2-MoE reduced top-k: 33 to 59 % acceptance; SpecMoE 0.6 to 0.9 on 2 to 128-expert models |
| `u25`, `cos8` (about 17 % of layers) | 0.5 to 0.8 | 0.1 to 0.4 | Draft&Verify, SWIFT 40 to 50 % sublayer skip: acceptance per token 0.45 to 0.92 on dense 13B+; llama.cpp #3565: 8 skipped layers +9 ppl |
| `u50`, `cos24` | 0.2 to 0.5 | below 0.1 | Component-Aware: 33 % skip 0.233 on a hybrid; early exit 50 % 0.0 |
| `qsa-only`, `dn-only` | below 0.1 | about 0 | Component-Aware: 0.019 (linear-only); the QSA-only direction is untested |

Expected outcome: negative. Cost fraction by bytes (`layer_cost.py`, head included, computed): `u50` 0.55, `cos24`
~0.55, `u25` 0.77, `topk4` 0.88 and `topk2` 0.85 (routed experts are only 19 % of a layer's bytes, the dense part
dominates), `qsa-only` 0.32, `dn-only` 0.78. At these costs the model
should find no mask whose extension beats MTP-only on the reference. A result outside this table (a mask with S_4 above
0.4 at cost fraction below 0.5) is a finding and goes to the review as such.

## Active tooling and conditions

`TONA_DETERMINISTIC` is not needed (CPU reference, deterministic arithmetic). No diagnostics, profilers or test switches
besides `OMP_NUM_THREADS=6`. The CPU runs go through `scripts/gpu_queue.py` because they use 6 threads and the page cache
of the 29.6 GB model, which would disturb other measurements. `tools/hw_report.py` and `tools/quiet_check.py` before and
after every run are appended to the results. Rates named: any time is seconds per full CPU step (not a GPU rate); all
tokens-per-second claims come from the cost model, not from a run, and are labelled "model".

## Part 2: trained helper (plan, written before any data is collected)

Request: in the style of DraftExpert (2607.24434) train a small dense FFN per layer
that stands in for the layer's routed experts in the draft pass only, so drafting reads no cold experts; the verify pass is
the untouched full model, so the output is identical by construction.

### Question and decision

Does a draft pass in which every MoE block is replaced by `shared expert + helper(x)` (variant A, routed experts replaced) or
by `helper(x)` alone (variant B, whole block replaced, the shared expert's ~4 MB per layer saved too), with all layers or
with a layer-skip mask, reach a survival high enough, at a cost low enough, to beat MTP-only at (a) the reference and (b) the
32 GB class (26 GB limit), the classes where draft cost is set by bytes and cold experts?

Break-even (the table the numbers are read against; costs by bytes from `tools/layer_cost.py`, pass costs from the
measured classes):

| Class | Full pass, 1 position | Time per token today (code) | Cost of a helper step, all layers (fraction f) | Survival a step needs to break even |
|---|---|---|---|---|
| reference (experts pinned) | 23.7 ms | 9.6 ms (4.7 tokens per 45 ms) | f = (5.10 - 0.98 + 0.05) / 5.10 = 0.82, 19 ms | (19 + 4.5) / 9.6 = 2.4, impossible |
| 32 GB class, limit 26 GB | ~23.7 ms plus cold experts (50.3 vs 89.2 decode: ~20 ms per token against 11.2) | ~20 ms | the same 19 ms, because the helper reads no expert, the point of the part | (19 + ~8) / 20 = 1.35, impossible |
| helper plus a skip mask with f = 0.45 | 10.7 ms | 9.6 / 20 ms | | reference 1.6 (impossible); 32 GB class 0.77 (possible only with a very good draft) |

Reading: a helper removes the 19 % of a token that are routed experts (0.98 GB of 5.1 GB, the cold part on the small
classes) but leaves the dense layers (4.1 GB), which dominate the byte count. It can pay only together with layer skipping or
a cut head, on a class where cold experts dominate the verify pass, and only for a survival of about 0.8 per step at the
extension depth. The measurement answers whether such a draft exists; it is expected to say no.

Decision, before the numbers: **propose the engine design for review** if, on the held-out set, the model predicts at least
+3 % tokens per second over MTP-only on code for the 32 GB class (or the reference) with the lower end of the bootstrap 95 %
interval above 0, for some helper size and mask. **Close negative** otherwise, with the table above filled with the measured
survivals. Training at scale (more tokens, on-policy rounds beyond round 2) only after this plan's pilot says a size and
mask come within 15 % of the break-even survival; the maintainer reviews before that.

### Steps

1. **Data.** `tona-layer-skip --moe-dump DIR` (to be added; the oracle trace already carries `moe_x`, `moe_routed`,
   `moe_shared` per layer, `want_moe`): per token and layer the MoE input x (2560), routed output, shared output, as fp16 in
   `/var/tmp/claude/155/data/` (737 KB per token over 48 layers; 8,000 tokens = 5.9 GB, never `/tmp`). Sources, none of them
   in the held-out set: 4,000 tokens of code from `work/corpus/code.txt` slices (teacher forced; layer-wise regression does
   not need the model's own continuation), 2,000 tokens of the model's greedy output on the calibration prompts of stage 1
   (code01, code03, code05, text00), 1,000 tokens of text, 1,000 tokens validation. Size is bounded by the CPU reference's
   seconds per token (stage 0); if it is above 3 s the pilot shrinks to 3,000 training tokens and says so.
2. **Distillation.** PyTorch (2.9.1, ROCm) from the system Python, per layer independently: SwiGLU FFN d=2560 -> h -> 2560,
   target `routed` (A) or `routed + shared` (B), loss = MSE normalized by the layer's target variance plus 0.1 x (1 - cosine),
   AdamW, 40 epochs, batch 256, early stop on validation. Size sweep h in {128, 256, 512, 1024} = 0.6, 1.1, 2.2, 4.4 MB at
   4.5 bit per weight = 0.6 to 4.7 % of a layer (94 MB on average; one routed expert is 2.05 MB, ten are 20 MB), 26 to 211 MB
   for 48 layers. Runs through `gpu_queue` on the GPU (under 2 GB of VRAM; the machine's other VRAM users are not touched,
   no pressure test), CPU fallback is enough for h <= 256 (about an hour). Reported per layer: relative error on
   validation against the baselines "no routed experts" (`routed=off` mask) and "top-2 / top-4 experts" (masks `topk2`,
   `topk4`), so the helper is only counted if it beats the experts it replaces at lower bytes.
3. **Offline acceptance.** The oracle loads the helper weights (`DraftPass::helper`, fp32 reference; quantizing to the
   engine's format is a later, separate check of the quality loss) and drafts as in part 1: stride 8, depth 6, the held-out
   prompts. Arms per size: helper on all layers; helper plus the best skip masks of part 1 (`cos8`, `cos16`); variant A and B.
   Output: survival curves `S_j`, extension after an MTP chain of k (`tools/self_spec_model.py`), tokens per second in the
   model for the reference and the 32 GB class (cost fraction f by bytes, the class pass costs above), always against the
   MTP-only baseline of the same model.
4. **Round 2, on-policy.** The helper is trained on inputs the full model sees; inside a draft pass later layers see the
   helper's perturbed residual. Round 2 collects (x seen by the helper pass, full MoE output for that x) and retrains
   on the union. Done once, only if round 1 survival at depth 4 is above 0.25.
5. **Engine design for review** only if the decision rule above is met.

### What we do differently from DraftExpert, and why

| DraftExpert (2607.24434) | Ours |
|---|---|
| shared + top-1 + one trained draft expert per layer; the top-1 routed expert still loaded | no routed expert in the draft: the 19 % of bytes that are cold experts are removed, not reduced to one; the shared expert stays (A) or goes too (B) |
| self-distillation of residual, logit/token and router agreement on 64 experts or fewer (DeepSeek-V2-Lite, Moonlight) | 256 experts, top-10, 1 to 3 bit weights: the regression target is the whole routed sum (10 experts), which a small FFN fits worse; the sweep measures how much worse, per layer, against the top-2/4 expert baselines |
| their drafter is a fixed footprint; truncation by confidence-expansion | we price the draft by bytes per class (reference, 32 GB, Mac) and choose helper size and skip mask against that break-even, because on the reference the dense layers, not the experts, set the step cost |
| not combined with layer skipping or an MTP chain | the helper is evaluated alone, with a layer-skip mask and as an extension behind the MTP chain, the combination that the byte arithmetic requires |
| training signal only from the full model's own inputs | round 2 trains on the draft pass's own inputs (on-policy) to limit the error that accumulates over 48 layers |
| needs exact verify by the target | same, and identical output is by construction (no change to the verify pass or the committed state) |

Active tooling for part 2: `want_moe` trace (off elsewhere); PyTorch run under `gpu_queue`; `OMP_NUM_THREADS=6` for the
CPU data collection; no engine switches. hw_report and quiet_check before and after each data collection and training
run, as in part 1.

## Results

Run 2026-11-16, commits up to 0395dcb (+ the training script fix). Everything below is read against the plan above.

### Conditions

`tools/hw_report.py` (start of the day): Ryzen 5 9600X, 6 cores / 12 threads; RX 9060 XT 16 GB (gfx1200); RAM 60.5 GiB, 16
GiB available before the first run; Fedora 44, kernel 7.2.15; no `TONA_*` variables. `tools/quiet_check.py` inside each queued
job, before and after: the machine was never quiet (other work held 1 to 2 GiB of VRAM, load average 5.6 to 6.7, other CPU
use 8 to 100 % between jobs, light swapping at times). The counters of the oracle runs are deterministic (CPU reference
arithmetic, same tokens), so the warnings do not touch them; the only timing is the 2.7 s per step of stage 0, which the
load makes an upper bound. Every job ran through `scripts/gpu_queue.py`.

### A defect found on the way, and what it invalidated

The first two stage 1/2 runs (`s12`, `s12b`) were wrong: the tool restored the saved state once per start instead of once
per mask, so the masks ran chained on each other's state, and the snapshot restore cut the caches back by size, which lost
the data of the draft positions. A duplicate-mask probe (`cos16` and `cos16b`, `routedoff` and `routedoffb`) gave different
results and exposed it (a cliff at the last draft step, a draft that was too good at the first mask and worse for each
following one). Both are fixed (snapshots are full copies; `restore` before every mask); the mini model checks that
duplicate masks give the same tokens (in the queue script), and the final runs contain the duplicates as a guard: **`cos16` ==
`cos16b` and `routedoff` == `routedoffb` in every number below**, and the mask `full` (topk 10) matches the full model at
every depth (survival 1.0). All numbers below are from the fixed tool (`s12c`, `s3`); `s12` and `s12b` are discarded.

### Stage 0: timing and layer change (code00, 24 prompt tokens, 40 generated)

2.7 s per full CPU step (173 s for 64 steps, 6 threads, loaded machine). Smallest residual change of a layer
(relative L2 change of the residual stream): layers 10, 11, 13, 9, 3, 8, 14, 23 (0.11 to 0.18); largest 36, 45, 46, 44
(0.65 to 1.07). The ranking of the four calibration prompts (stage 1, `--cos`, 96+ tokens each) agrees with it on 6 of 8, 14
of 16 and 22 of 24 layers; the masks `cos8/16/24` use the stage 0 ranking (the 4-prompt ranking was not yet available when the
run was queued; a documented deviation: the masks are the stage 0 ranking, the check of agreement is above).

### Stage 4: MTP baseline on the production build

`TONA_DRAFT_LOG` over `scripts/bench-set.py` (main build of 2026-11-15, MTP-256 head, port 8091, 54 GB cgroup, 3 prompts
groups as in the set): **decode** code 92.2 tok/s (3,049 tokens in 33.1 s), text 67.0 tok/s (1,038 tokens in 15.5 s); 1,007
passes, **3.98 tokens per pass**, mean verify pass 42.1 ms, mean draft time 4.4 ms. Conditional acceptance per depth of the
chain (adaptive depth, so deeper values carry selection): depth 1 to 7 = 0.851, 0.845, 0.866, 0.935, 0.91, 0.897, 0.892;
cumulative survival `S_j` = 0.851, 0.719, 0.622, 0.582, 0.53, 0.475, 0.424. The cost model with these and the default pass
costs gives MTP-only 84 tok/s (measured: 92 code, 67 text, mixed). It is the baseline every row below is compared to.

### Stage 2 and 3: acceptance of the draft passes (survival `S_j` = fraction of starts whose first j drafts match the full
model's tokens; step cost by bytes, ms; break-even = the largest step cost at which any scheme with a layer-skip
extension beats MTP-only; reference / 32 GB class (cold-expert factor 1.77, 50.3 against 89.2 tok/s decode, 2026-10-27))

Calibration set (code01, code03, code05, text00; 32 starts; depth 8, stride 10):

| Mask (layers skipped; cost fraction) | S_1 | S_2 | S_4 | S_6 | S_8 | step ms ref / 32 GB | break-even ms ref / 32 GB |
|---|---|---|---|---|---|---|---|
| `topk4` (experts 4 of 10; 0.88) | 0.97 | 0.94 | 0.88 | 0.72 | 0.69 | 21.0 / 28.3 | 3.6 / 4.8 |
| `topk2` | 0.91 | 0.84 | 0.69 | 0.53 | 0.41 | 20.0 / 23.7 | 2.1 / 2.3 |
| `routedoff` (shared expert only; 0.81 of a token) | 0.81 | 0.56 | 0.19 | 0.03 | 0.03 | 19.1 / 19.1 | 0.0 / 0.0 |
| `u25` (12 layers; 0.77) | 0.94 | 0.81 | 0.56 | 0.34 | 0.25 | 18.1 / 31.8 | 1.6 / 2.0 |
| `u50` (24 layers; 0.55) | 0.12 | 0.0 | 0 | 0 | 0 | | |
| `qsa-only` (36 layers; 0.32) | 0.0 | 0 | 0 | 0 | 0 | | |
| `dn-only` (12 QSA layers skipped; 0.78) | 0.69 | 0.44 | 0.19 | 0.06 | 0.0 | 18.6 / 32.0 | |
| `cos8` (8 layers) | 0.97 | 0.94 | 0.94 | 0.88 | 0.81 | 20.4 / 36.0 | 3.8 / 5.1 |
| `cos16` (16 layers) | 0.97 | 0.97 | 0.91 | 0.81 | 0.75 | 16.9 / 29.3 | 3.9 / 5.2 |
| `cos24` (24 layers) | 0.69 | 0.47 | 0.22 | 0.0 | 0.0 | 13.3 / 22.5 | 0.1 / 0.0 |
| `cos16topk4` | 0.97 | 0.91 | 0.72 | 0.59 | 0.47 | 15.0 / 20.0 | 2.8 / 3.4 |

Held-out (code00, code02, code04, code06, text01, text02; 36 starts; stage 3, never used for any choice):

| Mask | S_1 | S_2 | S_4 | S_6 | S_8 | break-even ms ref / 32 GB |
|---|---|---|---|---|---|---|
| `topk4` | 0.92 | 0.83 | 0.78 | 0.61 | 0.58 | 2.5 / 2.9 |
| `cos8` | 1.00 | 0.94 | 0.94 | 0.81 | 0.75 | 4.0 / 5.3 |
| `cos16` | 0.97 | 0.92 | 0.81 | 0.61 | 0.53 | 3.1 / 4.0 |

### Reading against the plan

- **The draft is much better than the expectation table said**: `cos8` and `cos16` (a third of the layers skipped, the ones
  whose residual change is smallest) keep 0.81 and 0.75 survival over 8 tokens on the calibration prompts (0.75 and 0.53
  held out), and `topk4` 0.69 / 0.58. The expectation was S_4 below 0.4 for these; Component-Aware's 0.233 on a hybrid
  does not carry over to a 48-layer model with a third of its layers close to the identity. Quality of the drafts is not the
  obstacle.
- **The cost is.** The best masks cost 16.9 ms (`cos16`) and 20.4 ms (`cos8`) per step at the reference and 29 to 36 ms at the
  32 GB class; the break-even is **3.1 to 4.0 ms** at the reference and **4.0 to 5.3 ms** at the 32 GB class. The steps cost
  4 to 6 times too much. The reason is the one in research/56: decode is bandwidth bound, a kept layer costs its 94 MB, and
  the MTP head produces a token for 1.15 ms while the engine already spends 9.6 ms per token. Skipping more layers breaks the
  draft (`cos24`: 0.69 at depth 1, `u50`: 0.12), skipping fewer does not make the step cheap. A step would have to cost about
  a sixth of a full pass.
- **Combined with MTP** (the model's best scheme at every row): `e = 0`, MTP only. The extension after the MTP chain gains
  at most S_k times the survival of the skip draft, and the MTP chain's own survival is 0.42 at depth 7; with a step of 17 ms
  against 5.2 ms break-even there is no k and e for which the cycle is faster.
- **32 GB class**: the cold experts make the baseline slower (50.8 tok/s model) and the draft step too; the break-even
  rises from 3.9 to 5.2 ms, the steps with experts rise to 29 ms (`cos16`), 36 ms (`cos8`). A mask without routed experts
  (`routedoff`, `helper`) costs 19 ms (dense layers only), but its draft is poor (0.19 at depth 4) or, with a helper, see below.
- **Decision (plan thresholds)**: no mask predicts +3 %; the best scheme gains 0.0 %. **Close #155 as negative** for the
  reference and the 32 GB class. The Mac (SSD-bound, ~49 ms single pass, ~150 ms cycles, MTP-164): not modelled with
  measured pass costs here; rough arithmetic with those figures (docs/hardware/mac-32gb.md, BENCHMARKS 2026-11-12) puts a
  `cos16` step at about 27 ms of dense bytes plus its cold experts against about 45 ms per token today, i.e. an extra
  token at about 27 + 15 ms for a survival of 0.75 to 0.9 gives 0.0179 to 0.0214 tokens per ms against 0.0222 now: at best
  break-even, with the cold experts not counted. Not measurable on this machine; left to the Mac owner as a note, not a build.

### Part 2: trained helper pilot

Data: 3,139 training tokens (2,627 teacher-forced code and prose from the repository's corpus, 512 greedy tokens of the model
on code07 to code10) and 974 validation tokens (code and prose slices, disjoint), MoE input, routed and shared output of all
48 layers as F16 (`--moe-dump`, 1.9 + 0.4 + 0.7 GB in `/var/tmp/claude/155/data`, not in the repository). Distillation
(`tools/self_spec_helper_train.py`): SwiGLU d -> h -> d per layer, MSE normalized by the target's variance plus 0.1 x
cosine, AdamW, 150 epochs, early stop on validation, GPU, 3 minutes for 48 layers. Validation relative error
(||pred - target||^2 / ||target||^2; 1.0 = predicting zero = the shared-only draft; mean over layers, best layer, worst):

| Variant | h (MB per layer at 4.5 bit) | mean | best layer | worst | baseline (no routed part) |
|---|---|---|---|---|---|
| A: routed experts | 256 (1.1) | 0.995 | 0.81 | 1.09 | 1.000 |
| A | 1024 (4.4) | 1.065 | 0.76 | 1.33 | 1.000 |
| B: whole block | 256 (1.1) | 0.901 | 0.19 (layer 0) | 1.17 | 0.820 |

(A first run with the default initialisation and 40 epochs gave 1.55 and 1.83: undertrained, discarded.) **The helpers
do not generalize at this data size**: the validation error of the routed sum of ten experts stays at the level of
predicting zero (the largest hidden size is worse: it overfits 3,139 samples), only the first and last layers (0.19 to
0.4 on B, 0.8 on A) learn something. DraftExpert's own helper replaces 1 of 64 experts with a trained one and keeps the
top-1 expert; ours has to stand in for ten of 256.

Why no acceptance run with these helpers and no larger data run: the pilot is not near break-even by the plan's own rule,
and the byte arithmetic closes the question without it: even a helper with **perfect** survival (every draft right) cannot
pay, because the draft step still reads the dense layers: step 19.4 ms (all layers, h=256), 13.9 ms (with `cos16`), 11.1 ms
(with `cos24`), against a break-even of 4.5 ms (reference) and 6.2 ms (32 GB class); the model's best scheme with perfect
survival is MTP-only (+0.0 %) in all four cases at both classes. A better helper would remove the experts' share (19 %,
and 0.98 GB of the 5.1 GB), not the dense layers' 4.1 GB.

### Verdict

Layer skipping and expert reduction give excellent drafts (survival 0.75 to 0.81 over 8 tokens at a third of the layers) but
the steps cost 4 to 6 times what the MTP head's chain makes affordable; a trained helper cannot fix that (perfect survival
still gives +0.0 %) and does not learn the routed sum from a few thousand tokens anyway. #155 closes negative for the
reference and the 32 GB class; no engine design is proposed. What would change it: a draft step at about a sixth of a
full pass (a different, much smaller draft network, not the model's own layers), or a class whose time per token is much
larger than its single pass (the Mac, left as a note above).
