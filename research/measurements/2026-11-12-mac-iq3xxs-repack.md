# Mac: IQ3_XXS experts repacked for the GPU (TONA_IQ3_XXS_R), server A/B (#108)

M1 Max 32 GB, Metal backend, MTP head cut to 164 experts.

## Plan (written before the run)

**Question.** What does the repacked IQ3_XXS layout (docs/use/tona-types.md; maintainer request 2026-11-12: "even 3 % per
token counts") bring the server's decode? Behind it: 2026-11-12-mac-expert-decode-probes.md, variant pk (synthetic,
in cache): IQ3_XXS gate_up −15 % at B1, −7 % at B4, −3 % per token of experts.

**Arms.** Same build (`work/build/mtl`, `20af2e7` plus the new type, dirty tree); A = DM
(`coder-DM-164e-00001-of-00002.gguf`), B = DM-R (`coder-DMR-164e`, made by `coder_requant repack` from DM: an APFS
clone with the 34 IQ3_XXS gate/up expert tensors repacked in place; the second shard a clone of DM's).

**Scope.** `scripts/ab.py --a mtl --b mtl --a-model DM --b-model DM-R --workload set --memory-max 40G` (bench-set,
11 code and 3 text prompts, context 32768), rounds AB, BA, ... until decided (alpha 0.05, margin 0.5 %), at most 6;
one discarded warm-up start of A. 6.2 min per round: 15 to 40 min.

**Expectation.** Identical texts (the weights are bit-identical: checks 1 to 4 on DM-R give DM's numbers). Decode code
and text 0 to +2 %: IQ3_XXS gate/up are ~30 % of the routed experts (14.8 ms of a 44.5 ms single-token pass,
2026-11-12-mac-verify-profile.md), −15 % of that is ~0.7 ms (1.5 %) at W = 1, less at wider passes. A probe with
`tona-bench` (2 × 2 rounds, 11:10) showed no difference beyond its noise: W1 44.6 / 44.6 ms, W4 92.1 / 92.1 ms. A
result inside 0 ± 0.5 % is the expected outcome, not a failure; ab.py may then run all 6 rounds.

**Active tooling.** None in the servers. `caffeinate -dimsu`. Quiet check and `tools/hw_report.py` before and after.
Before (11:16): quiet check ok; load 2.44 / 3.87 / 3.49, other programs 5.1 % of all CPU threads, commit `20af2e7`
(dirty tree: the new type).

## Results

Raw: [2026-11-12-mac-iq3xxs-repack.json](2026-11-12-mac-iq3xxs-repack.json) (ab.py). After (11:58): quiet check
warned (load 3.99, GPU 5 % busy: the run's own tail), other programs 14.0 % of all CPU threads.

| Metric | A per round | B per round | B against A, mean [95 % CI] | Expected |
|---|---|---|---|---|
| decode code | 31.1 / 31.0 / 32.5 / 31.0 / 31.6 / 31.4 | 30.5 / 32.6 / 30.9 / 32.4 / 31.3 / 31.3 | +0.26 % [−4.05, +4.57] | 0 to +2 % |
| decode text | 23.1 / 23.6 / 24.7 / 23.6 / 23.6 / 23.8 | 23.5 / 25.3 / 23.3 / 25.2 / 24.0 / 23.5 | +1.75 % [−3.65, +7.15] | 0 to +2 % |

Rounds (B against A, code / text): AB −1.9 / +1.7, BA +5.2 / +7.2, AB −4.9 / −5.7, BA +4.5 / +6.8, AB −1.0 / +1.7,
BA −0.3 / −1.3 %. 6 rounds, 38.2 min.

- **No effect measurable.** Both means lie inside the expectation, but so does zero, with a spread of ±4 to 5 %. The
  pass probe before (W1 and W4 equal within 0.3 ms) agrees.
- **Finding: order effect.** In rounds 2 to 4 the arm that ran first in the round was 4.5 to 7 % faster (BA favours B,
  AB favours A); rounds 5 and 6 are within ±2 %. Likely cause (unchecked): the second server of a round starts
  with the other model's pages in the cache and pages in its own trunk (both files share only the second shard).
  For later model A/Bs on the Mac: a warm-up start per arm and round, or more rounds.
- ab.py marks text "B faster after round 2" while its CI over 6 rounds includes zero; the decision rule and the
  interval disagree here. Read as: no evidence of a gain.
- Kept anyway (maintainer request, "even 3 % per token counts"): the change is bit-identical, costs no runtime memory,
  and the probe shows −15 % for the IQ3_XXS kernel in cache. Whether the real pass gains is below what this server A/B
  can resolve; a kernel-level A/B on the real model (profiler parts, 2026-11-12-mac-verify-profile.md) is the finer tool.
