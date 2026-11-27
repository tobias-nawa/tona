# Feasibility of an engine of our own

As of 2026-10-22. Analysis by an agent. All calculations of the hardware ceiling are
**rough estimates**, not measurements. Reference values rest on single data points.

> **Correction 2026-10-22 (06-qwen38-architecture.md, recounted from the GGUF):** dense weights per
> token are **4.11 GB** (incl. 0.52 GB LM head, 1.28 GB BF16 hyper-connections), not ~2.5 GB.
> That puts the ceiling at **~62 passes/s**, not ~95. Experts per token at 0.98 GB is correct.

> **Decided on 2026-10-22: we build.** The recommendation below is superseded. Still useful
> are the hardware ceiling (section 1, our target) and the effort table (section 2, our
> component list).

## Summary of the agent

A custom engine is not worth it. The room for improvement over existing engines lies in
kernel fine-tuning for gfx1200, not in the hardware. Exactly this work is being made faster
by upstream projects (llama.cpp and others) than one person could manage.

## 1. Hardware ceiling (estimate)

### Bandwidths

| Path | Value | Source |
|---|---|---|
| VRAM | 128 bit × 20 Gbps = **320 GB/s**; 25.6 TFLOPS FP32, 51.3 TFLOPS FP16 | [Tom's Hardware](https://tomshardware.com/pc-components/gpus/amd-radeon-rx-9060-xt-launches-on-june-5-starting-at-usd299) |
| PCIe | locally in sysfs: **32 GT/s x16 (Gen5)**, about 63 GB/s per direction, about 50 in practice | [ASUS specification](https://rog.asus.com/se/motherboards/rog-strix/rog-strix-b850-i-gaming-wifi/spec/) |
| DDR5-6000 dual | theoretically 96 GB/s, but the 9600X has **one CCD**, whose fabric link reads 32 B/clock: **~64 GB/s** at FCLK 2000. Measured 62.5 GB/s on one CCD | [Chips and Cheese](https://old.chipsandcheese.com/2024/08/14/amds-ryzen-9950x-zen-5-on-desktop/) |

**Consequence: 64 GB brings capacity, not speed.** Even EXPO 6000 instead of 5600 changes
little about RAM bandwidth as long as the fabric link is the limit.

### Model and cache

- Qwen3.8-Flash-Next: 48 layers, 512 experts per layer, 10 active, expert 2560×640,
  6B active parameters ([model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)).
- Coder IQ1_M: 23 GB for 12,288 experts, i.e. ~1.9 MB per expert and **~0.9 GB of
  routed weights per token**.
- Hit rate of the expert cache, measured with exactly this coder on AMD: 33% cache
  → 85% hits, 67% → 94% ([llama.cpp #29887](https://github.com/ggml-org/llama.cpp/pull/29887)).
  Our cache holds an estimated ~40%, so ~90% hits.

### Per forward pass

- VRAM: dense weights ~2.5 GB (estimated) + 0.8 GB experts → ~10 ms
- RAM: ~90 MB of misses → ~1.5 ms, in parallel

**Ceiling: ~95 passes/s, with MTP (×1.6–1.8) ~150 tok/s decode.** VRAM is binding, not
RAM or PCIe.

**Prefill:** 820 tok/s × 12 GFLOP/token ≈ 10 TFLOP/s, i.e. ~20% of the FP16 peak. The
bottleneck is compute or kernel quality, not PCIe (~0.3 s upload per 8K block).

### How close do existing engines get to the ceiling?

Existing engines stay well below the ceiling on this card. One reference point shows that
the bottleneck is latency and kernel overhead (48 layers, DeltaNet, QSA, synchronization),
not bandwidth.

- llama.cpp with the whole model in the VRAM of an R9700 (640 GB/s): only 43 tok/s decode without MTP (#29887).

**Theoretically 2–3× is possible, realistically 1.3–1.8×**, and only with months of work
on kernel fusion and graphs specifically for gfx1200.

Caveat: measure again with 64 GB. The drop to 28.5 tok/s decode after 10K tokens may come
from page faults at 32 GB.

## 2. Effort

One experienced developer, in person-months (PM). Estimate of the agent.

| Component | without reuse | with ggml |
|---|---:|---:|
| GGUF, tokenizer, architecture (DeltaNet, QSA indexer, gated residual, n-gram table from SSD) | 3 | 1 |
| HIP kernels for IQ1/2/3: matvec, WMMA prefill, DeltaNet scan, sparse attention | 6–8 | 1–2 |
| CPU experts with AVX-512 | 1 | 0.5 |
| Routing, expert cache, asynchronous uploads, CPU/GPU split | 2–3 | 2 |
| KV cache (int8) + DeltaNet checkpoints, prefix cache | 2 | 1.5 |
| MTP speculation | 1–2 | 1 |
| Server: OpenAI/Anthropic API, SSE, tool calls, thinking | 2–3 | 0.5 (llama-server) |
| Parity tests, profiling | 2–3 | 2 |
| **Total** | **~20 PM** | **~9 PM**, then ongoing maintenance |

Size of comparable projects (GitHub API, 10-22):

- **llama.cpp:** 11,825 commits. Support for this architecture alone: +2,881 lines ([#27742](https://github.com/ggml-org/llama.cpp/pull/27742)).
- **ik_llama.cpp:** 5,106 commits. Per README only CPU and CUDA fully functional, ROCm issues unwelcome ([#1643](https://github.com/ikawrakow/ik_llama.cpp/issues/1643)).
- **ktransformers:** 1,359 commits, 121 contributors.
- The expert-tiering engine considered as a base had 3,619 commits in 4 weeks, 2,630 of them from
  its maintainer, i.e. a very fast-moving upstream.

## 3. Options

| Rank | Option | Control | Effort | Risk / maintenance |
|---|---|---|---|---|
| 1 | **(e) Mix:** configure an existing expert-tiering engine, llama.cpp as second backend, own slim proxy and measurement layer | orchestration full, engine via flags | days | low. The upstream hangs on one person, the MIT license is a safeguard |
| 2 | (c) Configure the existing engine, contribute gfx1200 measurements and patches | medium | days to weeks | breaking changes weekly |
| 3 | (d) llama.cpp master with `-cmoe --moe-cache-mib` ([#29887](https://github.com/ggml-org/llama.cpp/pull/29887), SLRU [#30637](https://github.com/ggml-org/llama.cpp/pull/30637); computing misses on the CPU still open in [#30966](https://github.com/ggml-org/llama.cpp/pull/30966)) | medium | hours | old values on this card 20/450 tok/s (decode/prefill), measured before the cache. Measure again |
| 4 | (b) Fork the existing engine | high | permanent | at ~90 maintainer commits per day, left behind after days |
| 5 | (a) from scratch | full | 9–20 PM | tied to one model; every new architecture 0.5–2 PM |

### What "control" really brings here

- **Reachable via flags or a proxy:** cache strategy, VRAM budget next to the desktop
  (1.4 GB), determinism, fidelity of the tool-call format
  (`--pcie-frac`, `--kv`, `--spec`, `parallel`).
- **Not reachable:** model quality, the 16 GB limit, the 320 GB/s. For a coding worker,
  model quality matters most.

## 4. Recommendation of the agent

Option (e): take the existing engine as it is. Put own work only into the coordinator
layer: proxy, prompts that use the prefix cache well, a measurement harness. Measure
llama.cpp master on the same machine as reference and fallback path. Fork only if upstream
is abandoned.

**Drawbacks:**

- Depends on a four-week-old project with one maintainer and 647 open issues and PRs.
- Updates can break configurations.
- The possible 1.3–1.8× stays on the table.
- Several workers in parallel share the same 320 GB/s, each one becomes slower.
- Agent loads are prefill-heavy. A cold context of 30K takes ~37 s, whatever the engine.
