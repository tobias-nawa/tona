# 65 - Speculative verify on the DeltaNet (GDN) layers: what we pay, is MTP a net win (#169)

Marking: **[own]** our code or measurement file, **[pr]** PR or issue text read this session (2026-11-18), **[abs]** arXiv
abstract, **[known]** row of `research/papers.md`. Rates named: **decode** tok/s, **pass** ms of one verify pass.
Measurements: `research/measurements/2026-11-18-mtp-net-win-169.md`. Drafters (#173): `research/66-drafters.md`.

## 0. Verdict

1. **Our chain verify already is ReplaySSM.** No per-draft state snapshot exists. The DeltaNet state is read once per pass
   and committed lazily by the next pass from the saved delta / k / alpha of the accepted tokens (`mega.h` header,
   `op_dn_recur` in `mega_op_dn.h`, cites arXiv 2610.13778) [own]. `save_state` (the real snapshot) runs only at prefill
   boundaries (`server/prefill*.cpp`), never per verify: **snapshot bytes and time per verify = 0**.
2. **The remaining GDN cost per verify is `dn.recur`: 0.91 ms at width 1, 1.20 at 4, 2.07 at 8 of a 24 / 36 / 55 ms pass
   (3.8 / 3.3 / 3.7 %)** [own, 2026-11-12-verify-pass.md, 9600X; 9950X: 0.93 / 1.22 / 1.66 ms, 2026-11-17-cu-scaling.md]. At W=1 it is within ~25 % of the bandwidth floor of one
   state read plus write. The ceiling for any further DeltaNet work is about 1 ms of 42 ms (2 to 3 %), and the lever
   candidates below stay under 1 %.
3. **vLLM #68081 does not transfer.** The reported loss (-10 % single stream, -48 % at 16 streams) is explained in the issue
   itself by vLLM's mamba state residency (one extra state block per draft token, #55533), a scheduler artifact; we hold one
   state per slot plus a ring of (delta, k, alpha). Our own number: see section 4.
4. **SGLang #50601 is about trees.** We verify chains only (width 8, one path) [own, research/37]; the PR's chain case is what
   `--enable-linear-replayssm-spec` and we already do. It becomes relevant only if a tree drafter is built (Goose-class,
   Bole 2608.01651), see 5.

## 1. What the issues and PRs do

| Item | State | What it does | Gain reported | Transfer |
|---|---|---|---|---|
| [vLLM #68081](https://github.com/vllm-project/vllm/issues/68081) [pr] | open issue | Qwen3.8-27B AWQ, 2x RTX 4090D, MTP with 2 drafts: decode 80 -> 72 tok/s (-10 %), aggregate at 16 / 32 / 64 streams -48 / -46 / -20 %. The thread attributes it to #55533: MTP reserves one mamba state block per draft token, 2 -> 4 blocks per request, halving batch capacity | a loss | none: our slot has one state, speculation is single-stream (the -10 % single stream stays unexplained there: "presumably draft-verify overhead") |
| [SGLang #50601](https://github.com/sgl-project/sglang/pull/50601) [pr] | open PR | Tree verify on GDN without `intermediate_ssm` (one K x V state per draft node). Unrolls H_p = A_p H_0 + sum (A_p/A_i) k_i (x) d_i back to H_0: verify needs H_0^T k_t and H_0^T q_t for all nodes (H_0 read once), two TxT Gram matrices of k.k and k.q, and one pass over nodes in index order (ancestor sums of d_i with scalar weights, d in registers). Commit folds the accepted path into H_0 from the rings | GDN layers of a tree verify 1.33-1.94x, verify forward 1.11-1.24x, 4.8-38 GB snapshot memory gone, up to 1.21x throughput (11 Qwen models, NVIDIA) | chain case = ours. Tree case: needs a tree drafter |
| [vLLM #67590](https://github.com/vllm-project/vllm/pull/67590) RecoverSSM [pr] | open PR | Port of the Qwen GDN RecoverSSM stack: records outside the KV cache, commit kernels after verify, flag `use_gdn_recoverssm`. Measured by a contributor (L40S, Qwen3.5-4B, concurrency 32): +10.1 % at K=3, +7.6 % at K=5; none at concurrency 1, -3 % at K=1 | gains only under load and at K >= 3 | same family as ours; no gain at concurrency 1 |
| [SGLang #50253](https://github.com/sgl-project/sglang/pull/50253) [pr] | open PR | Opt-in cross-round deferred commits for DFlash ReplaySSM: keep accepted factors in the ring across verify rounds, materialise the state only when the ring needs room or a checkpoint consumer asks; 26 % of active rows folded in a diagnostic | +1.32 % / +0.40 % throughput, intervals include zero | see 3 (idea B) |
| [SGLang #51053](https://github.com/sgl-project/sglang/pull/51053) [pr] | open PR | Tiled GDN ReplaySSM compact commit (tcgen05 kernel, bit-identical to stock) | kernel speed only, CUDA | n/a (Blackwell kernel) |
| ReplaySSM 2610.13778, Bole 2608.01651, SpecLA 2607.16673, TreeWY 2608.20961 [known] | | lazy commit from inputs; closed-form tree verify; compact factors | at B=1 "a few percent" (ReplaySSM) | already adopted (chain) |

## 2. How our verify handles the DeltaNet state [own]

- Pass of width W (<= 8, `MEGA_BMAX`): op `OP_DN_RECUR` (36 layers). The stored state S lags one pass behind. At the start of a
  pass the op replays the `commit` accepted tokens of the previous pass from their delta, alpha and k (read from the other half,
  `parity ^ 1`, of the intermediate buffers `pd`, `pa`, `pk`), token by token in the same order as in that pass, then runs the
  W new tokens from the committed state and writes delta, alpha, k of the new tokens into the half `parity`. The state itself is
  read once into registers (32 rows per lane) and written once, after the commit and the new tokens are applied only as far as
  needed for outputs.
- Rejected tokens cost nothing in the state: they are never committed (`commit(m)`, `m` = accepted). KV cache, indexer and
  ring buffers are overwritten at the rejected positions. This is exactly ReplaySSM's "rollback = pointer back".
- Bit-exactness across widths: every token takes the same steps wherever it sits in a pass and however many tokens were
  committed before it (comment in `op_dn_recur`).
- Restore points (`save_state` / `load_state`, `stash` / `unstash`): a copy of the state buffers (`state_bytes()`, ~108 MiB
  [own, 2026-11-12-verify-pass.md]) in VRAM or RAM at prefill boundaries and for prompt-cache handoffs. Not in the decode loop.
- Cost: `dn.recur` per pass in the table of 2026-11-12-verify-pass.md; the commit replay is included (m = nb in that profile).

## 3. Levers

All are bounded by the 2.07 ms of `dn.recur` at W=8 (0.91 at W=1) and have to stay bit-exact or pass the quality check.

| Idea | Mechanism | Expected gain | Verdict |
|---|---|---|---|
| A. Tree verify without snapshots (#50601) | Only for a tree drafter | none for chains | not applicable now |
| B. Deferred commit across rounds (#50253) | Skip the state write when the ring still has room, fold every n-th pass | saves part of the write half of dn.recur: <= 0.3 ms of 42 ms (0.7 %), costs a growing replay at the next fold and a larger ring; they measured +1.3 % with intervals through zero | not worth it; change of numerics order risk |
| C. State in FP16 | halves the state traffic | <= 0.4 ms (1 %) | lossy; ReplaySSM: FP16 needs stochastic rounding; no |
| D. Use CUs better in dn.recur | 768 units, latency-bound waves; 16 -> 32 CUs gives only 1.0 / 1.14 / 1.24 [own, 2026-11-17-cu-scaling.md] | the W=8 step (+0.85 ms over W=1) could shrink; <= 0.5 ms (1.2 %) | candidate, low priority |
| E. Spec width / count rule | the decision of how many drafts to verify | the dominant term is not GDN: moe.gate_up + moe.down grow 2.2 ms per extra token, the W=4 -> 5 step is 8 ms [own] | belongs to #126, not to the DeltaNet |

**Proposed next lever: none for the DeltaNet itself.** The ceiling is ~2 to 3 % of a pass and every candidate is < 1.2 %. If
the coordinator wants one candidate: D (more parallelism in `dn.recur` at W >= 4), expected +0.5 to +1 % decode on code,
measured with `ab.py` (noise sd 0.07 to 0.6 %). Better returns are in the expert terms of the pass (research/37, #126).

## 4. Is MTP a net win on our path

**Yes, clearly** [own, 2026-11-18-mtp-net-win-169.md]. `scripts/ab.py`, same build, bench-set, 9950X, **decode**: with
speculation 96.5 (code) / 69.1 (text) tok/s, without (`TONA_SPEC_OFF=1`, 1.00 tokens per pass) 41.4 / 41.3 tok/s: speculation is
**+133 % on code (2.33x) and +67 % on text (1.67x)**, CI of the difference -57.1 % [-58.0, -56.2] and -40.2 % [-41.2, -39.3] for
spec off. Spec off equals the width-1 pass of the pass table (24.2 ms per token), so it is the real plain decode path. The
vLLM -10 % does not occur; its cause there is state-block residency, which we do not have.
