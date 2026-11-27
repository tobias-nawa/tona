# Installer: interface, hardware detection, models for 32 GB

Two agent research runs of 2026-10-27, not checked by ourselves. [V] page read, [S] search hits only, [W] knowledge
of the agent. Benchmarks are vendor claims unless stated otherwise.

## Interface (TUI)

| Option | State | Forms | Remark |
|---|---|---|---|
| C++ FTXUI | v7.0.3 (2026-08-06) [V] | input, checkbox, radiobox, menu, dropdown, gauge built in [V] | no dependencies, Windows supported, FetchContent with a fixed tag; static linking unchecked |
| Go Bubble Tea | v2.1.0 (2026-10-08) [V], v2 stable since 2026-02-23 [S] | via Huh [S] | static binary and cross-compile easy; second language, detection (HIP, Metal, DXGI) only via processes or CGO |
| Rust ratatui | 0.30.2 (2026-06-19) [V] | missing, extra crates [W] | good base; third language |
| C++ notcurses | v3.0.17 (2025-10-28) [V] | – | no release for a year, weak on Windows |
| own ANSI like ie-top | – | build ourselves | enough for display, not for forms |

The agent's recommendation: C++ with FTXUI, core as a library of its own, `--yes`/configuration file without the TUI.

## Hardware detection without root

No library delivers GPU and VRAM reliably on all platforms; vendor tools plus a fallback are usual. Linux: sysfs `mem_info_vram_total`/`_used` (our source, `hipMemGetInfo` does not see the desktop),
`rocminfo`, `nvidia-smi`. Mac: Metal `recommendedMaxWorkingSetSize` (only via Objective-C/Swift), `system_profiler`
for display. Windows: DXGI `QueryVideoMemoryInfo`; WMI `AdapterRAM` is limited to ~4 GB [S]. CPU: CPUID or
`/proc/cpuinfo`. Measure the SSD read speed ourselves, with `O_DIRECT` (otherwise one measures the page cache).

## How others do it

- **Ollama `install.sh`** [V, 455 lines]: GPU via `lspci`/`lshw`, downloads the matching package (e.g. `-rocm`), sets up
  systemd; **no checksum**, no VRAM estimate.
- **hipfire** v0.4.3 [V, README]: `install.sh` downloads a verified kernel package per architecture, otherwise builds locally;
  `hipfire doctor` (driver/ROCm), `hipfire fit <model> --ctx N` (does it fit, what would be chosen), `hipfire pull`
  with a registry (partly SHA-256 and size). Closest model for subcommands.
- **llmfit** [V]: speed = bandwidth ÷ model size × 0.55 (MoE: active parameters); fit by required ÷
  available (≤ 60 % good … > 98 % too tight).
- **LM Studio** [S]: `lms load --estimate-only` (weights + context); reported misestimates (22.9 instead of
  11.5 GB, lmstudio-ai/lms #499). Lesson: verify the estimate with a trial run.

## Hugging Face

`GET /api/models/{repo}/tree/main` returns per LFS file `lfs.oid` (SHA-256), `size` [V]; pages with `cursor`.
Resume the download via `resolve/main/<file>` with HTTP range [W], write to `.part`, then verify SHA-256.
Multi-part GGUFs (`…-00001-of-0000N.gguf`): all parts by pattern from the list, verify each one individually.

## Models for 32 GB RAM

| Variant | Size | 32 GB RAM + 16 GB VRAM | Code quality | in our engine |
|---|---|---|---|---|
| ISTA-Coder IQ1_M (today) | 29.6 GB + 28.8 GB PLE (SSD) | yes, experts partly from the NVMe; own measurement up to 2026-10-23: 28–33 tok/s decode | SWE-V 79.2, LCB v6 87.4 (ISTA) | runs |
| Gyro-Coder TQ2_0 | 19.9 GiB + 50.7 GiB PLE | yes | only KLD 0.118 against Q8_0, 13/16 own tasks | new trellis kernel |
| Gyro-Coder-L TQ3_0 (new since 2026-10-23) | 27.5 GiB + PLE | tight | KLD 0.077 | new trellis kernel |
| Qwen3.8-27B ISTA IQ3_XXS / IQ3_S | 10.1 / 11.8 GB | yes, entirely in VRAM | SWE-Pro 61.7, LCB 84.6–85.7 | dense, full attention: new path |
| Ornith-1.5-35B-A3B Q4_K_M / Q5_K_M | 21.7 / 25.3 GB | yes | SWE-V 79, TB 2.1 67.8 | different widths: new path |
| REAP-288, Swift 1.5, Signal 3.8 | 66–116 GB | no | – | – |

Sources: HF cards ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF, agentionai/…-Gyro-Coder-GGUF,
ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF, ornith-ai/Ornith-1.5-35B-A3B-GGUF [V]; details in 08-models.md.

SSD streaming, third-party numbers: llama.cpp, RTX 4090 + 32 GB DDR5, UD-IQ1_M with `--n-cpu-moe 39`: 25–27 tok/s (decode)
at 3K, 10–12 at 120K (one user, 09-04) [V]. A published cache hit rate at 32 GB does not exist.

## Conclusion

- PC with 32 GB: the same Coder, experts partly from the NVMe; the limit is set by `IE_RAM_GB`. First measure with
  today's engine at the 32 GB limit (pinning + NVMe), the old number is from 2026-10-23.
- Mac with 32 GB: most likely the Gyro-Coder (trellis kernel needed), or a smaller model; belongs to a Metal
  backend, not to this step.
