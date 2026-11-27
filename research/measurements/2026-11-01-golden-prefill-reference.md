# Golden `GOLDEN_MODE=prefill` without drafter: reference re-recorded (2026-11-01, #87)

`GOLDEN_MODE=prefill scripts/golden.sh compare` (13K prompt token by token through the megakernel, 64 tokens greedy)
reported DIFFERS on main (645db0b) against the reference of 2026-10-27.

- Two runs on `work/build/merged` (main 645db0b): byte-identical to each other, so the output is deterministic.
- They differ from the old reference first at byte 36,002 of 36,158, i.e. in the last decoded tokens (the old reference
  has 36,158 bytes, the new output 36,147).
- With `IE_IDX_SHARE=all` the same build reproduces the reference of 2026-10-27 byte for byte. Nothing but the
  IndexCache default changed this output; no other commit since 2026-10-29 is involved, so no bisect was needed.
- Cause: c25f9b2 (#73) made IndexCache the default (`engine/backend/idx_share.h`, every second QSA layer runs its
  indexer). The commit message and README state it is not bit-identical and that output below about 2,000 tokens of
  context is unchanged, which is why the short decode golden stayed identical. It passed the delta-NLL rule and the task
  benchmark (`2026-10-30-threshold-main.md`, `2026-10-30-indexcache-main.md`).
- The check was not run after c25f9b2 because the mandatory runs (`gpu-status.sh`) do not include
  `GOLDEN_MODE=prefill`.

Decision: the change was intended; the reference `work/golden/prefill/long.txt` is re-recorded from main 645db0b
(old file kept as `/var/tmp/claude/w87/long-2026-10-27.txt`). `IE_IDX_SHARE=all` still reproduces the old reference.
