# Switch to Rust? (2026-10-25)

Two agent research runs (Sonnet, 2026-10-25) plus one own measurement. Sources at the end; "read" = page fetched,
"snippet" = search hit only.

**Decision, 2026-10-25: stay with C++.** Security of the server layer through fuzzing the C++ server.

## Result

- **Kernels stay HIP C++.** For gfx1200 there is no mature way to write device code in Rust that carries a
  persistent megakernel with mailbox, wave intrinsics and WMMA. The Rust target `amdgcn-amd-amdhsa` is
  tier 3, nightly only with `-Zbuild-std`. CubeCL is alpha by its own account and designed for matmul on RDNA3.
- **The Rust engines that are fast on AMD also write their kernels in HIP C++** and use Rust only as the
  host: hipfire (kernels compiled by hipcc to `.hsaco`, ROCm via dlopen).
- **CPU side:** The AVX-512 intrinsics we use have been stable since Rust 1.89 (2025-08-07)
  (`_mm512_dpbusd_epi32`, `_mm512_permutexvar_epi8`, `_mm512_reduce_add_ps`, `__mmask64`), avx512fp16 since 1.94.
  A gain over the C++ kernels is not to be expected, it would be the same machine code.
- **Server layer** (HTTP, JSON, SSE, tool calls, tokenizer) is the only place with real benefit:
  memory safety when parsing foreign input, mature crates (axum/hyper, serde, `tokenizers`). No influence
  on tok/s.
- "Rust is the AI language of the future" holds for routers, servers and tooling (HF TGI router, `tokenizers`), not
  for high-performance kernels.

## Own measurement: fork-join OpenMP vs. Rayon (Ryzen 9600X, 6 threads)

20,000 regions per run, median of 7 (min–max), overhead = time per region minus work per thread.
Code: `/var/tmp/claude/forkjoin/` (g++ -O2 -march=znver5 -fopenmp; rustc 1.97.1, rayon 1, opt-level 3).

| Work per thread | OpenMP passive | OpenMP active | Rayon |
|---:|---:|---:|---:|
| 0 | 0.50 µs (0.49–0.56) | 0.50 (0.48–0.51) | 0.87 (0.87–0.88) |
| 10 µs | 0.45 (0.44–0.45) | 0.46 (0.45–0.47) | 0.98 (0.98–1.00) |
| 40 µs | 0.47 (0.45–0.48) | 0.46 (0.46–0.48) | 1.01 (0.99–1.10) |

Rayon costs ~0.5 µs more per region. At ~48 regions per pass that is ~25 µs of 39 ms (< 0.1 %): no
argument either way. (fork_union README: on large machines Rayon is 2–10× above a
spin pool; on 6 cores here only 2×.)

## hipfire (read, v0.4.3, 2026-10)

Rust runtime for AMD RDNA, Apache-2.0. Models including `qwen3.8:flash-next` in its own formats MQ2/MQ3/MQ4; **no
IQ GGUF**, no numbers for gfx1200. Flash-Next on R9700 (gfx1201): decode 33.1 tok/s (tg128), prefill 6,431
tok/s (pp16384). Aimed at Strix Halo and multiple GPUs (expert parallelism). Speculation: DFlash, MTP
(+14–26 % on text, −3.1 % on code edits), n-gram. Not a reference for us (different format).
Adoptable regardless of language: shadow/check mode (check a new path against a recorded run, lock it on
deviation), hash-checked loading of the kernel objects, publish prefix snapshots only after commit.

## Options

| Option | Gain | Risk |
|---|---|---|
| stay with C++ | – | – |
| Rust only for the server layer (own process or C ABI per request) | Safety when parsing, fuzzing easier | second toolchain; the boundary must never be crossed per token |
| Rust host + HIP kernels | Ergonomics | Mailbox, pinned memory, OpenMP replacement: a lot of `unsafe`, small benefit |
| everything in Rust | none measurable | Device code does not work |

## PoC plan (server layer only, GPU core untouched)

1. Baseline of the current server: time to first token, overhead per SSE token, 1–8 clients at once.
2. Rust binary (axum, serde, `tokenizers`) with a mock backend (fixed token streams): the same measurements.
3. Connection to the C++ core through a small C ABI (`submit`, `poll`) or a Unix socket.
4. Fuzzing of the JSON/HTTP inputs (cargo-fuzz) and aborted streams; for comparison the same cases against
   cpp-httplib.
5. Success: decode tok/s and time to first token unchanged (spread over several runs), no crashes
   under fuzzing, build without touching `toolchain/env.sh`.

A cheap intermediate step with the same security gain: fuzz the current C++ server (libFuzzer on
`api.cpp`/`chat.cpp`).

## Sources

Read: github.com/Kaden-Schutt/hipfire (README, docs/ARCHITECTURE.md, docs/BENCHMARKS.md, issue #860),
github.com/tracel-ai/cubecl, github.com/EricLBuehler/mistral.rs, github.com/maeddesg/vulkanforge,
github.com/ashvardanian/fork_union, doc.rust-lang.org (std::arch, RELEASES.md), crates.io API (hip-sys 0.1.2,
cubecl-hip-sys 7.14, rocm-rs 0.5.2, rocmrc 0.5.1).
Snippets: doc.rust-lang.org/rustc/platform-support/amdgcn-amd-amdhsa.html, rust-lang/rust PR #138940, issue #111137,
huggingface.co/blog/tgi-multi-backend, ashvardanian.com/posts/beyond-openmp-in-cpp-rust.
