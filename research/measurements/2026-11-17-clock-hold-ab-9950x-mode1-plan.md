## Plan (written before the run)

- Scope: workload short (2000:400 2000:write:400 at --ctx 140000); A `/var/tmp/claude/161/build` against B `/var/tmp/claude/161/build`; rounds 1 to 4, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: prefill@2000 only: the effect to exclude is the 7 to 9 % queue effect of 2026-11-12 (A/A spread 0.2 %), so 1 to 2 rounds decide; 4 at most; decode must stay equal (decode tokens per prompt as in the targets, default 200)
- Duration: 1-5 min (1-4 rounds of 1.2 min, estimate before round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@2000 sd 0.09 % (df 4), decode@2000 sd 0.231 % (df 4), prefill@2000:write sd 0.163 % (df 4), decode@2000:write sd 0.605 % (df 4)
- Expectation (B against A, source: A/A of 2026-11-12-prefill-bimodal.md (prefill 0.2 %, decode 0.1 to 0.5 %)):
  - prefill: -0.5 to +0.5 %
  - decode: -0.5 to +0.5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-954-g0602d8be73; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=/var/tmp/claude/161/wt/work/profile/experts-coder.bin
  - B: build commit v0.0.2-954-g0602d8be73; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_CLOCK_HOLD=1, TONA_EXPERT_PROFILE=/var/tmp/claude/161/wt/work/profile/experts-coder.bin
