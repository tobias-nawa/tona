# ADR: several model architectures in one engine (Coder, Ornith, Bolt)

Accepted 2026-11-15 and built for the Coder, Ornith and Bolt on the HIP reference backend (status 2026-11-25, code read, not rerun).
The plan and its tickets are in the git history of this page.

| Part | State |
|---|---|
| Model layer: `arch.h`, `model_base.h`, `binder.h`, `blocks.h`, one module per architecture, `load_model` in `model_registry.cpp` | built; the Coder's class is `Qwen4ExpModel` |
| Registry with `ArchSupport{arch, types, tier}`, refusal at load naming the first tensor without a kernel, `tona-server --list-backends` | built (`engine/backend/registry.h`, `builtin.h`) |
| Server: `ChatFamily`, `ChatFormat`, pixtral pre-tokenizer, `fingerprint` | built (`engine/server/chat_qwen3_*.cpp`, `chat_nemotron_*.cpp`, `engine/model/pretok_pixtral.cpp`) |
| HIP builder split, one kernel translation unit per architecture | built: `engine/gpu/arch/{qwen4exp,qwen35moe,nemotron_h}/` |
| CPU reference per architecture | built: `engine/cpu/ref_{qwen4exp,qwen35moe,nemotron_h}.cpp` |
| Isolation gate | built: `tools/gates/arch_isolation.py` |
| Ornith, Bolt (NVFP4, IQ2_XXS, Mamba-2 with replay commit, latent MoE, tokenizer, chat) | built, tier Maintained, catalogued as tested (`toolchain/models.toml`) |
| Bolt DFlash drafter with LiLiCorr | **not built**; Bolt drafts from the history only |
| Metal | tier Experimental, Coder only |
| Rates and quality | [BENCHMARKS.md](../../BENCHMARKS.md) (rows of 2026-11-19 and 2026-11-20); promotion not decided here |

## 1. Context

| | Coder (`qwen4exp`) | Ornith 1.5 35B-A3B (`qwen35moe`) | Nemotron 3.5 Bolt 74B-A9B (`nemotron_h_moe`) |
|---|---|---|---|
| Layers | 48: 36 DeltaNet + 12 QSA | 40 (30 DeltaNet + 10 gated attention) plus 1 MTP layer (`blk.40`) | 88: 40 Mamba-2 + 8 attention + 40 MoE |
| Residual | 4 gated streams (hyper-connection) | one stream | one stream |
| d_model, vocab | 2560, 248,320 | 2048, 248,320 | 4096, 131,072 |
| Attention | QSA (indexer, top-k blocks), 24 Q / 2 KV x 256, gate | full context, 16 Q / 2 KV x 256, gate, QK norm | full context, 32 Q / 2 KV x 128, no gate, no QK norm |
| FFN | MoE + shared expert | MoE 8 of 256 + shared, ff 512 | latent MoE (4096 -> 1024), sigmoid router with bias, 9 of 240..496 experts per layer, ff 1536..2560, scale 5.0, shared expert ff 5376 |
| Draft | MTP head (separate GGUF), DFlash drafter | MTP head in the model file | no MTP; DFlash with LiLiCorr (separate GGUF, arch `dflash`) |
| Native context | 262,144 (1M with YaRN) | 262,144 | 262,144 |
| Tokenizer, template | `gpt2/qwen35`, ChatML + XML tool calls | same | `gpt2/pixtral`, other system block |

Bolt has at most 496 experts per layer (240 in the smallest), pruned unevenly. Source: GGUF headers (`tools/gguf_info.py`) and
`research/08-models.md` section 3.

### 1.1 Quantization types and kernels

| Type (ggml id) | Used by | HIP kernel | CPU dequant oracle | CPU experts (AVX-512) | Metal |
|---|---|---|---|---|---|
| F32 (0), BF16 (30) | all small tensors | yes | yes | n/a | yes |
| Q4_K (12) | Ornith: 15.2 GiB | yes | yes | added by the Ornith line | yes |
| Q6_K (14) | Ornith: 4.9 GiB | yes | yes | added by the Ornith line | yes |
| Q8_0 (8) | Bolt: 1.7 GiB | yes | yes | n/a | yes |
| IQ2_XXS (16) | Bolt: 16.3 GiB (all experts) | added by the Bolt line | yes (`model/quant.cpp`) | added | decoder only |
| NVFP4 (40) | Bolt: 2.3 GiB (`ssm_in`, `ssm_out`, per-tensor F32 `.scale`) | added | added | n/a | no |

## 2. Decision

1. **One module per architecture**, selected by `general.architecture`: `ModelBase` plus a typed module each (`Model` = the Coder;
   `Qwen35MoeModel`; `NemotronHMoeModel`). No superset `Hparams`.
2. **The `Backend` interface stays model-neutral.** The registry knows which architectures and quant types a backend serves; an
   unsupported pair fails at load with the reason.
3. **In the HIP backend the architecture is a compile-time unit:** its own program builder (host, cold) and its own kernel
   translation unit with `k_mega<NB, Profile>`. The Coder's unit includes no op of another architecture, so its machine code stays
   identical (`scripts/isa-diff.py`).
4. **Architectures are compositions of block builders** (DeltaNet, gated attention, Mamba-2, MoE), shared where the maths is shared.
5. **Order:** bit-exact refactor of the Coder, then shared blocks, then Ornith and Bolt. Server, tokenizer, chat and installer follow
   the model's family through the model layer, never `if (arch == ..)` in shared code.
6. **Alternatives rejected:** one generic `Hparams` with flags (cross-model branches in hot kernels, 2 ms per pass for an unused
   case); `std::variant` with visitors; one binary per architecture; one kernel with every op family (grows the Coder's kernel,
   rule 1); a forked HIP backend (the toolbox would exist three times).

## 3. Design

### 3.1 Model layer (`engine/model/`)

| File | Content |
|---|---|
| `arch.h` | `ArchId { Qwen4Exp, Qwen35Moe, NemotronHMoe }`, `detect_arch(const Gguf&)` (throws with the known names), `ArchNeeds`, `weight_types()` |
| `model_base.h` | abstract `ModelBase` (cold path): `arch()`, `gguf()`, `dims()`, `plan()` (per-layer expert geometry), `chat_family()`, `mtp_source()`, `fingerprint(Hasher&)` (prompt-cache key), `describe()`; `model_as<M>` checks `arch()` once |
| `binder.h` | the shape-checked `Binder` ("nothing may be left over"); an unknown tensor or wrong shape stops the load |
| `blocks.h/.cpp` | `DeltaNetWeights`, `GatedAttentionWeights`, `MoeWeights`, `Mamba2Weights` with shared binders |
| `qwen4exp.*` | API unchanged; `Model` derives from `ModelBase` (Metal sees no difference) |
| `qwen35moe.*`, `nemotron_h_moe.*` | typed `Hparams`, a layer kind per layer (read from the tensors present, cross-checked with the per-layer arrays), `plan()`, chat family, MTP or drafter description |
| `model_registry.cpp` | `load_model(path)`: the only `switch` over `ArchId` in the model layer |
| `tokenizer.cpp` | pre-tokenizer chosen by `tokenizer.ggml.pre` (`qwen35`, `pixtral`; other values throw); gate `tests/tokenizer_vs_llamacpp.py` |

### 3.2 Registry

`backend::registry::Entry` has `name`, `probe`, `archs` (`ArchSupport{arch, QuantSet types, Tier}`) and `create(const ModelBase&, ..)`.
`pick(name)` takes the model: "auto" chooses the first backend whose probe passes and lists `model.arch()`, otherwise the error names
every backend and why. Before `create` the registry compares `model.weight_types()` with `ArchSupport::types` and names the first
tensor without a kernel. The downcast lives in each architecture's factory, never in pass code. `add()` still accepts the old Metal
shape (Coder only). The matrix is printed by `tona-server --list-backends`; tiers: [backends/README.md](../backends/README.md#tiers).

### 3.3 HIP backend: program builder and kernel profile

- **Builder.** `ProgramBuilder` keeps the toolbox (`op`, `addw`, `mv_begin/mv_end`, `group`, state, KV and weight allocation, slots,
  `run()`) and calls an `ArchProgram` per architecture (`alloc_buffers`, `build_target`, `build_mtp`, `kernels`). Layer loops are
  data over block builders (`deltanet_block`, `gated_attention_block`, `mamba2_block`, `moe_block`). Coder-only files live in
  `engine/gpu/arch/qwen4exp/`. Gate for the move: `TONA_MEGA_DUMP_PROGRAM` byte-equal to the baseline.
- **Kernel profile.** `k_mega<int NB, class Profile>`; the profile carries op families, the Q6_K layout, head sizes and LDS floats;
  `run_op` composes the dispatch with `if constexpr`. Each architecture has `arch/<name>/kernel.hip` with NB 1, 4, 8 and a
  `KernelSet`; `Mega::start_kernel` calls `KernelSet::pick(nb, program_traits)` (one indirect call per pass).
- **Why a separate translation unit.** The two tile cases of `matvec_phase.h` cost every pass 2.1 ms at 1 token (+9 %) although no
  task ran them (SGPR spills of `k_mega<8>` 147 to 163, code size +8 %,
  `research/measurements/2026-11-15-pc-q6k-tile-spills.md`). A model must not carry op families it does not run.
- **Resource contract per kernel:** at most 192 VGPRs and two blocks of 16 waves per WGP (above that barriers hang, `mega_kernel.h`);
  LDS not above `SM_FLOATS` (64 + 256 + 4096 floats); `scripts/kernel-regs.py` output in the measurement file.
- **Limits made parameters** (Coder values unchanged): `MegaMailbox<MaxE>` (256 for the Coder, Bolt up to 496); `MEGA_UMAX` 128 is
  enough; expert profile file version 2 with per-layer counts (version 1 still loads); expert cache geometry and slot planning per
  layer; `MEGA_MAX_CELLS = 4096`/`SM_FLOATS` keep QSA scores in LDS, full-context attention (262,144 positions) cannot (3.9).

### 3.4 State, commit, snapshots

Both new architectures keep recurrent state and obey the pass contract ([backends/README.md](../backends/README.md#interface-contract)):
`forward(nb)` up to 8 tokens, `commit(m)` keeps m. DeltaNet uses replay (the next pass applies the m accepted tokens from stored
delta, k and alpha; ReplaySSM, arXiv 2610.13778). Ornith reuses it; Mamba-2 stores `dt`, `B`, `x` per token and layer and applies
the accepted prefix at the start of the next pass (a per-token state copy would cost 4 MiB per layer, 1.3 GB for 40 layers x 8
tokens). State regions go through `alloc_state`, `seq_buf` and `alloc_kv`, so stash, prompt cache and `snap_fits` need no knowledge
of Mamba. The prompt-cache key adds `ModelBase::fingerprint`.

### 3.5 Prefill, CPU reference, Metal, server

- **Prefill.** The generic part (chunking, expert streaming, double buffers, events) is shared; the per-layer sequence and `PfLayer`
  variants (`PfDeltaNet`, `PfGatedAttn`, `PfMamba2`, `PfMoe`) belong to the architecture. Coder-only kernels (`prefill_qsa`,
  `prefill_ple`, `prefill_hc`, `prefill_topk`) live in `arch/qwen4exp/`. Mamba-2 prefill is a chunked SSD scan. Whether
  `prefill_attn.hip` covers unindexed full attention is unchecked.
- **CPU reference.** `cpu/ref_ops.h` has the primitives; `ref::Oracle` (`reset`, `step(token) -> logits`, `max_ctx`) one
  implementation per architecture. An oracle we write shares our reading of the architecture, so every new architecture is also
  checked against llama.cpp-hip (N7; `scripts/build-llamacpp.sh`). CPU experts (`cpu/experts.cpp`) are a table of row functions per
  quant type with `expert_type_supported`.
- **Metal.** Unchanged view (`model/qwen4exp.h`, old registry `Entry`). Mac step: register with the new `Entry`, declare
  `ArchSupport{Qwen4Exp, types, tier 3}`, run the checks of `engine/backend/metal/AGENTS.md`. Ornith and Bolt are refused with a
  message.
- **Server.** `ServerContext::model` is `std::unique_ptr<ModelBase>`; tools needing Coder specifics (`tona-check`, `tona`) use
  `model_as<Model>`. `ChatFamily` (`Qwen3Coder`, `Nemotron`) selects a `ChatFormat` (renderer, parser, thinking close text, sampling
  from `general.sampling.*`); gate `tests/chat_vs_jinja.py` (byte-equal to the GGUF's Jinja template). Installer:
  `toolchain/models.toml` entries have `arch`; `tona setup` filters by the backend's `ArchSupport` and hardware class; catalogue
  entries are added after the model runs, with the license check of `research/42-model-licenses.md`.

### 3.9 New building blocks

Each ticket starts with the skill `research-before-building` and measures its block alone before wiring it in.

| Block | For | What it is | Start from |
|---|---|---|---|
| Plain residual ops (`OP_RES_NORM`) | Ornith, Bolt | `R' = R + y`, `xn = rmsnorm(R') * w`, int8; the Coder's ops assume 4 gated streams | the maths; fill `b_.mixed`/`b_.qmix` directly so the DeltaNet builder is reused |
| Full-context attention, decode | Ornith (HD 256, group 8, gate, QK norm), Bolt (HD 128, group 16) | split-K over all cached positions, online softmax, two-level merge, no indexer, partial buffer `max_ctx / chunk` | QSA `OP_ATTN_WMMA`; flash-decoding merges at 262,144 positions; int8 KV |
| Full-context attention, prefill | Ornith, Bolt | causal chunked attention over the prefix | `research/34-prefill-attention-kernel.md` |
| Partial RoPE with sections | Ornith | `rope.dimension_count` 64 of 256, sections [11, 11, 10, 0]; for text one RoPE (unchecked against llama.cpp) | llama.cpp `qwen35moe` |
| Routing policies | Ornith (softmax, top 8, renormalized), Bolt (sigmoid + bias for the choice, normalized, scale 5.0) | one policy definition for CPU and GPU | `OP_HOST_WAIT`, `OP_ROUTE` |
| MTP in the model file | Ornith | `blk.40`, `eh_proj` [2*2048 -> 2048], `enorm`, `hnorm`, `shared_head_norm`, shared LM head | the Coder's MTP code; llama.cpp `nextn` |
| Mamba-2 decode and prefill | Bolt | causal conv (4 taps, SiLU), selective scan with `dt = softplus(dt + bias)`, gated group RMSNorm (8 groups of 1024), replay commit; prefill as chunked SSD scan on WMMA | Mamba-2/SSD (arXiv 2405.21060), llama.cpp `ssm_scan`, `research/13-deltanet-chunked.md` |
| NVFP4 | Bolt | 64 weights in 36 bytes (FP8 E4M3 scales, E2M1 values) times `.scale`; doubled E2M1 values are small integers, so the int8 path is exact with the scale folded per 16 weights; no FP4 hardware on gfx1200 | llama.cpp NVFP4/MXFP4 |
| IQ2_XXS | Bolt | 256 weights in 66 bytes (grid index, sign bits, 4-bit block scale) | the Metal decoder; int8 loops of types 18 and 22 |
| Latent MoE | Bolt | latent down and up are Q8_0 matvecs; experts have up and down only, activation `relu(x)^2` (confirm against llama.cpp) | activation as a policy of the MoE op |
| Pixtral pre-tokenizer | Bolt | other regex split than `qwen35` | llama.cpp's pixtral split |
| DFlash with LiLiCorr | Bolt, later | block drafter, window 1024, conv kernel 2, NVFP4 and BF16, correction head over the top 8; the server's suffix speculation works without it | the Coder's drafter program |

## 5. Prior art

llama.cpp ([src/models/](https://github.com/ggml-org/llama.cpp/tree/master/src/models)) and vLLM
([ModelRegistry](https://docs.vllm.ai/en/latest/contributing/model/registration.html)) key one module per architecture on its name;
we add typed hparams per module, a kernel with only that architecture's ops, and registry entries with quant types. The Hazy
Research megakernel ([blog](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)), MPK
([arXiv 2512.22219](https://arxiv.org/abs/2512.22219)) and AMK ([arXiv 2606.09682](https://arxiv.org/abs/2606.09682), abstract only;
Llama-family only) motivate per-model compile-time specialization. mistral.rs, SGLang and vLLM's hybrid state handling were not
opened.

## 6. Consequences and checks

Build time and fat-binary size grow with each architecture (three kernel instantiations each; unchecked whether it matters, a CMake
option could select architectures). A fourth architecture touches one model module, one arch directory, one registry line per
backend and, if needed, one chat family. Every GPU run goes through `scripts/gpu_queue.py` on a quiet machine.

**Coder guard (rules 1 and 3):** C1 and C6 for every ticket, C2 to C5 by what it touches. A ticket touching only `arch/<new>/` and
op headers only the new translation unit includes cannot change the Coder, so C1 is its whole guard.

| Check | Question | Decision |
|---|---|---|
| C1 `scripts/isa-diff.py` against the baseline build | Did an instruction, register count or LDS size of any kernel change? | Identical: go on. Changed in a Coder kernel: find the cause or revert. Changed in a new architecture's kernel: expected, N1 applies. No GPU needed |
| C2 `TONA_MEGA_DUMP_PROGRAM` byte-equal to baseline (default start with MTP and drafter; CPU-expert mode with two slots) | Same op programs, task tables, buffer layout? | Different and unintended: fix. Intended: re-record goldens, run C4. For tickets touching `engine/model/`, `engine/backend/` or `engine/gpu/` host code |
| C3 Goldens with `TONA_DETERMINISTIC=1` | Same tokens? | Different: a logic bug, bisect. Run the golden of the touched path: `gpu` (model, program, passes), `GOLDEN_PLAN=cpu` (planning, selection), both prefill goldens, `golden-server.sh` (server) |
| C4 Pass curve: batch bench, `TONA_BENCH_HASH=1`, W = 1, 4, 8, 3 interleaved runs | Did a pass get slower? | Only if C1 shows a kernel change or the ticket touches per-pass hot-path host code (`gpu/mega.hip`, `mega_serve.cpp`, `mega_seq.cpp`, `mega_select*.cpp`, `mega_experts.cpp`, `mega_prefetch.cpp`, `mega_state.cpp`, `mega_mtp*.cpp`, `mega_abort.cpp`, `prefill_run.cpp`, `prefill_moe_run.cpp`). Slower beyond `noise.toml`: fix, no merge |
| C5 `scripts/ab.py` production A/B with the real server and MTP | Production decode and prefill unchanged after the wave? | Once per wave; slower: bisect with C4 |
| C6 `tools/gates/arch_isolation.py` | Shared code free of per-architecture branches? | Violation: move the branch into the architecture module |

**New architectures:**

| Check | Question | Decision |
|---|---|---|
| N1 `scripts/kernel-regs.py` | At most 192 VGPRs, two resident blocks of 16 waves per WGP, LDS within `SM_FLOATS`? | No: barriers can hang; split the op or change the profile |
| N2 Block against the CPU primitive | Same result within tolerance, at real layer dimensions? | Attention also at 262,144 positions (partial-buffer sizing breaks only there) |
| N3 Block speed | Within 2x of its bandwidth or compute bound? | No: research the cause first (principle 8). No BENCHMARKS row |
| N4 Commit contract (Mamba-2, MTP) | State after `forward(nb)` + `commit(m)` equals the state after m tokens? | Bit-equal with `TONA_DETERMINISTIC=1`; 20 passes of random width 1 to 8, plus a prefill in one piece against random pieces |
| N5 Quant kernel against the dequant oracle | Same numbers? | Within the int8 activation error; one layer's real tensors per type |
| N6 Tokenizer and chat | Same token stream and rendered prompt as the reference? | `tests/tokenizer_vs_llamacpp.py`, `tests/chat_vs_jinja.py` |
| N7 llama.cpp-hip cross-check, once per model | Same function as an independent implementation? | Top-1 equal except near-ties, logit deviation at the level between llama.cpp's own CPU and HIP runs; one code and one prose prompt, 64 teacher-forced steps, plus one 32K prompt. Larger: localize layer by layer with the oracle |
| N8 Tier-2 measurement, once per model | How fast and good is the model in Tona? | AGENTS.md rule 2; numbers go to BENCHMARKS.md and decide promotion |

## 7. Implementation plan

Done except B8: baseline, then wave 1 (bit-exact Coder refactor: model layer, registry, server, builder split, kernel profile, core
generalization, CPU reference split, isolation gate, Mac step), wave 2 (full-context attention, plain residual ops, routing
policies), wave 3 (Ornith: module, CPU experts, HIP module, MTP, catalogue; Bolt: module, NVFP4, IQ2_XXS, Mamba-2, latent MoE,
HIP module, tokenizer and chat, DFlash, catalogue). The conflict points between lines are `gpu/matvec_phase.h` (one owner for NVFP4
and IQ2_XXS), `toolchain/models.toml`, `BENCHMARKS.md` and the registries.

## 8. Decisions of the maintainers (2026-11-15)

1. Ornith and Bolt start as tier 2 (maintained); promotion after N8.
2. Whether to offer Bolt through the installer is decided after the Bolt line shows it is worth it; its license (OpenMDW-1.1 per the
   research notes) is checked per `research/42-model-licenses.md` before any entry exists.
3. Kernels of all architectures in every binary, with a CMake option to drop some if size or build time matters.
4. `Model` is renamed `Qwen4ExpModel`.
5. The Bolt drafter comes last (open on 2026-11-25): Bolt decodes at 9 B active parameters without one, and the server's suffix
   speculation is model-independent.
