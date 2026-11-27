# Pinned host buffers against barrier aborts

Own measurement, 2026-10-23, 18:50–19:19. 64 GB RAM, desktop (compositor, 2 × 4K) on the same GPU, **vkcube running
alongside** as graphics load.

## Change

The buffers into which every pass copies logits, features and drafter outputs (`logits_host_`, `logits1_`,
`feat_host_`, `dlog_host_`, `dgate_host_`) were `std::vector` in ordinary memory. For copies from/to such
memory the ROCm runtime pins it via userptr every time, and freeing it triggers a KFD eviction of all queues
of the process through the MMU notifier (ROCm/rocm-systems#12528, gfx1200, open; research/09-gpu-reset.md). Now allocated
once, pinned, with `hipHostMalloc`.

## Setup

`ie-gpu-gen` new model (coder-iq1_m-1019), `--teacher engine/gpu/mega.hip --max 3000 --ctx 4096 --mega`,
`IE_PLAN=gpu`, profile experts-coder-1019.bin, 32 blocks, no CU mask. Per run `evicted_ms` from
`/sys/class/kfd/kfd/proc/<pid>/stats_*/evicted_ms` (every second, every change with timestamp) and kernel log for
`ring … timeout` / `GPU reset`. Script: /var/tmp/claude/vramtest/run.sh (not in the repository).

## Result

| | Runs | Aborts | gfx timeouts | evicted per run | Perplexity | tok/s (decode) |
|---|---:|---:|---:|---|---:|---:|
| before: `std::vector`, 96 slots (teacher runs in the afternoon) | ~10 | ~50 % | 2 resets that day | ~700 ms, keeps growing | – | – |
| pinned, reserve 2000 MB (92 slots) | 6 | **0** | 0 | 716–733 ms, once 1893 | 4.928 (all 6 equal) | 25.0 |
| pinned, default reserve, 96 slots (as before) | 6 | **0** | 0 | 706–724 ms | 4.927 (all 6 equal) | 25.5 |

- Eviction happens **exactly once per run**, ~18 s after the start, i.e. at the end of loading (loaded in 17.2 s).
  Nothing is added during the 3000 tokens. The one outlier (+1.2 s) coincided with the end of run 2.
- The before row is not a measurement with the same script but the observation from the teacher runs in the
  afternoon (same settings, without vkcube). With 12 of 12 clean runs under more load the difference is
  still clear.
- The perplexity differs between 92 and 96 slots in the 4th digit: different summation order when more
  experts are in VRAM. Bit-identical within one setting.

## Open

- Where the ~720 ms at loading come from (presumably pinning the 25 GB of experts or the first start). The reset at
  18:45 (`ring gfx_0.0.0 timeout`, a terminal window) coincided with a start. `amdgpu.debug_evictions=1` would show the cause
  (needs root).
- Whether the server runs just as cleanly under Claude Code: `just hang-repro`.
