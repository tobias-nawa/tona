# Code provenance audit (#122, item 2)

Date: 2026-11-11. Scope: every tracked file under `engine/` (without `engine/third_party/`), `installer/`, `tools/`,
`scripts/`, `bench/`: 629 files (179 `.cpp`, 163 `.h`, 141 `.py`, 33 `.hip`, 29 `.sh`, 10 `.metal`, 9 `.mm`, 5 `.swift`,
data and build files). Tona's own licence is MIT.

Question per file: was code **copied** or **ported** from another project (licence duty: copyright and licence notice),
or only **inspired** (idea, own implementation; naming is a courtesy)?

## Result in short

- Code that carries foreign copyright: four files, all from llama.cpp/ggml (MIT, `Copyright (c) 2023-2026 The ggml
  authors`), all four already listed in `engine/third_party/README.md`. No hidden copies elsewhere.
- No GPL, LGPL or other copyleft code anywhere. No Apache-2.0 code was copied, so **no NOTICE file is needed**.
- Three findings that need a decision (details below):
  1. **FTXUI** (MIT, statically linked into `tona-install`) has no licence text in the release's `tona-LICENSES.txt`.
  2. **`engine/model/mtp_vocab.inc`** (resolved 2026-11-13): 32,768 of its 41,489 IDs were the `code32` list of
     llama.cpp PR 31031 (no licence stated). Decision: nobody is contacted, the list is gone; the file is now
     regenerated from our own corpora only.
  3. **`engine/gpu/mega_mtp.cpp`** `quantize_iq4_*`: the algorithm of ggml's `quantize_row_iq4_nl_impl`, rewritten.
     Not in the third-party list.
- Everything else with a reference in a comment (PRs, papers, blog posts) is `inspired` or `independent`.

## Method

1. Candidates: grep over `engine installer tools scripts bench` for project names (llama.cpp, ggml, vLLM, SGLang,
   FlashAttention, FlashInfer, MLX, Triton, Marlin, CUTLASS, Composable Kernel, rocWMMA, aiter, ArcticInference),
   arXiv IDs, "ported", "modelled on", "after", "like", "as in", "from", GitHub URLs; plus ggml-specific identifiers
   (`kmask_iq2xs`, `ksigns_iq2xs`, `iq2xxs_grid`, `scale_min_k4`, `kvalues_iq4nl`, ...).
2. Similarity, in two independent ways, for **all 629 files** against the **whole llama.cpp tree** (checkout
   `work/src/llama.cpp`, HEAD `d9ee007`, which is the tag `b11822`, so the version named in the README is the one
   compared):
   - Token shingles (comments and preprocessor lines stripped, 7 / 8 / 12 tokens, identifiers kept).
   - Line match: whitespace-normalised lines of at least 28 characters (20 for chosen file pairs), without
     `#include`, comments and blank lines.
   Scripts: scratchpad only, not committed (`sim.py`, `lines.py`).
3. Targeted comparison against the cited upstream where it is reachable: llama.cpp PR 31233 (patch), vLLM PR 52228
   (`acceptance_estimator.py`), vLLM PR 47808 (added Python lines), aiter PR 6670 (file list), ArcticInference
   `csrc/suffix_decoding/*` (SuffixDecoding C++), `gguf-py` and `ggml/src/gguf.cpp`, the ggml-cuda/Metal/Vulkan
   gated-delta-net and flash-attention kernels.
4. Regeneration check: `tools/gen_tables.py` run against the llama.cpp checkout reproduces `quant_tables.h` and
   `unicode_tables.h` byte for byte (`git status` clean afterwards, nothing kept).

Grades: `copied` (text identical or generated from the source), `ported` (structure and logic taken over, rewritten),
`inspired` (idea or algorithm, own code, no shared text), `independent` (no relation beyond the common file format or
a public specification).

## Table 1: foreign code in the tree

| File | Source (project, file, version, licence) | Grade | Evidence | Measure |
|---|---|---|---|---|
| `engine/model/quant_tables.h` | llama.cpp `ggml/src/ggml-common.h`, b11822, MIT | copied (generated) | `tools/gen_tables.py` reproduces the file byte for byte; 97 % of 12-token shingles occur in `ggml-common.h`; header names licence and copyright | none, listed |
| `engine/model/unicode_tables.h` | llama.cpp `src/unicode-data.cpp`, b11822, MIT | copied (generated) | same regeneration check; header names licence and copyright | none, listed |
| `engine/model/quant.cpp` | llama.cpp `ggml/src/ggml-quants.c` (`dequantize_row_*`), b11822, MIT | ported, partly near-verbatim | 40 of 113 lines with at least 28 characters and 78 of 219 with at least 12 occur verbatim in `ggml-quants.c`; 60 % of 7-token shingles; same loop structure and operand order (documented on purpose: bit-exact with `-ffp-contract=off`); e.g. `deq_iq2_xxs` next to `dequantize_row_iq2_xxs` | none, listed. The first line says "Ported from", good |
| `engine/model/tokenizer.cpp` | llama.cpp `src/unicode.cpp` (`unicode_regex_split_custom_qwen35`, `unicode_cpt_from_utf8`) and `src/llama-vocab.cpp` (BPE merge queue ordering), b11822, MIT | ported (rewritten as the class `Qwen35Splitter`, other control flow and names) | Rule order, flag logic and the contraction tests follow the llama.cpp function case by case; only 1 line is identical (`return l.rank > r.rank \|\| ...`); the comment says "one-to-one from unicode_regex_split_custom_qwen35" | none, listed |

## Table 2: borrowed ideas and algorithms that are not in the list

| File | Source | Grade | Evidence | Measure |
|---|---|---|---|---|
| `engine/gpu/mega_mtp.cpp` (`kIq4`, `nearest_iq4`, `iq4_candidate`, `quantize_iq4_block`) | llama.cpp `ggml-quants.c` `quantize_row_iq4_nl_impl` + `kvalues_iq4nl`, MIT | inspired, borderline ported | Same procedure: 15 candidate scales `(k + values[0]) / max`, nearest of the 16 IQ4_NL values, least-squares refit `sum(x q) / sum(q q)`; comment says "scale like ggml". Different selection (minimum squared error instead of `d * sumqx`), no weights, no shared lines. `kIq4` equals ggml's `kvalues_iq4nl` | Add the file to the README list as "algorithm after ggml, rewritten", or one comment line naming the function and licence; use `tables::kvalues_iq4nl` instead of the second copy of the 16 numbers |
| `engine/gpu/mega_alloc.cpp` `upload_q8_0` | ggml `quantize_row_q8_0` | independent | Absolute maximum divided by 127 and rounding is the definition of Q8_0; 0 shared lines | none |
| `engine/gpu/class_lru.h`, `tools/cache_sim/classlru.py` | llama.cpp PR 31233 (ClassLRU, `src/llama-moe-cache.cpp`), MIT, merged | inspired | Same algorithm (CRF with half-life, class = floor(log2 CRF), victim = max age/2^class). 1 of 70 lines in common (`#include <cmath>`); the upstream uses intrusive index lists over slots, Tona `std::list` per class per expert; different API (`use/tick/insert/promote/victim`) | none; the comment names the PR already |
| `engine/cpu/ref.cpp`, `engine/cpu/dflash.cpp`, `engine/model/qwen4exp.cpp` | llama.cpp `src/models/qwen4exp.cpp`, `src/models/dflash.cpp`, `common/speculative.cpp` | inspired | "Structure follows llama.cpp ..." (layer order). 1 of 347 and 0 of 243 lines match; the model architecture is public (model card, paper); different code (no ggml graph) | none; comment kept as a courtesy |
| `engine/gpu/prefill_dn.hip`, `engine/gpu/mega_op_dn.h`, `engine/backend/metal/kernels/deltanet.metal` | llama.cpp PR 20340 (fused gated delta net: `ggml-cuda/gated_delta_net.cu`, Metal, Vulkan), MIT | inspired | "Modelled on PR 20340" (idea: token-by-token recurrence without chunking). 0 shared lines in 142, 177 and 57 against all gated-delta-net kernels (CUDA, Metal, Vulkan, CPU); other decomposition (one wave per V head and 16 columns, readlane) | none |
| `engine/backend/rope_yarn.h` | YaRN (arXiv 2309.00071), transformers `_compute_yarn_parameters`, llama.cpp `rope_yarn` | inspired | A published formula (corr dims, ramp, mscale); the file builds a per-pair table in double, no shared lines | none |
| `engine/server/draft.cpp` `allocate()` | vLLM PR 47808 (DSpark confidence-scheduled verification), Apache-2.0 | inspired | 0 of 139 lines in common with the Python added in the PR; a greedy allocation by expected rate | none, no NOTICE (no Apache code taken) |
| `engine/spec/draft_calib.h` | vLLM PR 52228 (`acceptance_estimator.py`), Apache-2.0 | inspired | idea only (calibrated acceptance); vLLM's estimator is a Triton kernel over logits, Tona's is a host-side IRLS logistic fit; 0 of 84 lines | none |
| `engine/spec/suffix.*`, `engine/server/suffix.*` | SuffixDecoding (arXiv 2411.04975), ArcticInference `csrc/suffix_decoding`, Apache-2.0 | inspired | 0 lines in common with `suffix_tree.cc/.h`, `int32_map.h`; other data structure (trie of counts) | none |
| `engine/gpu/mega_op_attn.h` | aiter PR 6670 (FlyDSL QSA sparse decode, gfx950), MIT | independent | "After the pattern of": one block per KV group. The reference is a Python DSL kernel; no overlap possible; the HIP code and its LDS layout are Tona's | none |
| `engine/gpu/prefill_attn.hip` (flash-attention form) | FlashAttention (Dao et al.), BSD-3-Clause | independent | Online softmax is the published algorithm; 0 of 181 lines in common with ggml's `fattn*.cu`; the FlashAttention sources themselves were not compared (see "Not verified") | none |
| `engine/backend/idx_share.h`, `engine/gpu/mega_select.cpp`, `engine/gpu/prefill_run.h`, `engine/server/sampling.*`, `engine/server/generator.h`, `tools/coder_requant/select_experts.py` | papers: IndexCache 2603.12201, XShare 2602.07265, OEA 2511.02237, layered prefill 2510.08055, Leviathan et al. 2211.17192, REAP 2510.13999 | inspired | Paper algorithms with citation, implemented from the paper | none |
| `engine/gpu/matvec*.h/.hip`, `gemm_dequant.hip`, `engine/cpu/experts*.{h,cpp}`, `engine/backend/metal/kernels/decode.metal`, `bench/mac/*k.metal` | ggml block formats (`ggml-quants.c`, `ggml-cuda/dequantize.cuh`, `ggml-metal`) | independent (format-determined), tables reused from the listed `quant_tables.h` | 0 lines in common with `ggml-quants.c` in `matvec_dot_sub.h` (98 lines), `gemm_dequant.hip` (166), `matvec_q8_kquants.h` (71), `experts.cpp` (245) and the others. The shingle overlap (up to 16 %) is the bit unpacking that the format dictates. `scale_min_k4` is a 6-bit unpack that the format dictates | none: tables come through `quant_tables.h`, so the notice covers them |
| `engine/model/gguf.cpp` `kTypes`, `tools/gguf_info.py`, `tools/coder_requant/gguf_layout.py` | ggml `type_traits` (type id, name, block size, bytes per block) | independent (facts) | Numbers of the GGUF file format, own formatting, 0 shared lines | none; the comment "values from ggml.c" is enough |
| `engine/server/chat_render.cpp` (`kToolsIntro` and the following literals), `engine/server/generator.cpp` (budget-forcing sentence) | Qwen chat template (inside the GGUF) and Qwen docs | independent (functional text of the model) | The strings must match the template so the model behaves; `tests/chat_vs_jinja.py` checks equality with the template | none; Qwen's licence is in item 1 of #122 |

## Table 3: dependencies and data

| Item | Origin and licence | Finding | Measure |
|---|---|---|---|
| `engine/third_party/httplib.h`, `nlohmann/json.hpp` | cpp-httplib v0.51.0 (MIT), nlohmann/json v3.12.0 (MIT) | SHA-256 equals the README, licence files present, release adds them | none |
| FTXUI v7.0.3 (`installer/CMakeLists.txt`, `FetchContent`, pinned commit `f921fad2`) | Arthur Sonzogni, MIT ("Copyright (c) 2019 Arthur Sonzogni") | Fetched at build time, **linked into `tona-install`** (FetchContent builds FTXUI as static libraries by default; not checked with `ldd` on a built binary), and `tona-install` is a release asset. `scripts/release-build.sh` builds `tona-LICENSES.txt` from `LICENSE` plus `engine/third_party/LICENSE-*` only, so the FTXUI notice is missing in the binary distribution. MIT needs it in every copy | Add `engine/third_party/LICENSE-ftxui` (text of the pinned tag, which the glob picks up) and a README line "fetched by CMake, statically linked into tona-install"; extend the sentence in `README.md` section Licence and the comment in `release-build.sh` |
| OpenSSL (`engine/update`, `installer`) | OpenSSL 3.x, Apache-2.0 | `find_package(OpenSSL)`, linked dynamically against the system library, nothing copied, nothing bundled in the release (not checked on the build host: if a release ever ships `libssl`, NOTICE and licence become due) | none now |
| `bench/corpus/python-stdlib-3.14.txt` | CPython 3.14 (`argparse`, `subprocess`, `logging`, `asyncio`), PSF licence | `LICENSE-python.txt` is the full licence file (277 lines, includes the PSF agreement and the CWI/CNRI notices); `README.md` and `bench/README.md` name the exception; the file contains the original author headers | none (optionally keep the "# ==== file: x.py ====" markers; they are the only change) |
| `bench/threshold/` (pydash MIT, boltons BSD-3-Clause) | cloned at pinned commits by `fetch.py` into `work/`, never committed | Nothing foreign in the repository; the context text is built at run time. If the benchmark texts or its results ever contain function bodies of these projects, the licences (BSD-3: notice in redistributions) apply to that file | none |
| `engine/model/mtp_vocab.inc` (41,500 token IDs) | Tona's own count (`scripts/build-mtp-vocab.py`) over code and text on the author's machine, tokenized with `tona-tokenize`; only counts and IDs are stored. Until 2026-11-13 the file also held the `code32` list of llama.cpp PR 31031 (praneshgo, no licence stated, 79 % of the IDs) | No third-party data left: the script has no download, the list is a function of own corpora and our tokenizer | none; the measured effect of the swap is in `research/measurements/2026-11-13-mtp-vocab-own.md` |
| `scripts/bench-long-code.txt` (36 KB) | Tona's own earlier prefill source (German comments) | Own code, no foreign lines (the two llama.cpp mentions are comments of ours) | none |
| `engine/tests/data/*`, fuzz seeds, golden files | generated by Tona's own tools from fixed seeds, or hand-written; `yarn_ref.txt` holds reference numbers of the YaRN formula (origin of the generator not traced) | no foreign text | none |
| `engine/tests/test_quant.cpp` expected values | outputs of `model/quant.cpp`, confirmed equal to ggml's `to_float` | facts, no ggml code | none |
| `tools/tests/requirements*.txt`, `.pre-commit-config.yaml` | development dependencies | not distributed | none |

## Licence compatibility

- llama.cpp / ggml: MIT, compatible; duty is copyright and licence text in all copies, binaries included. The release
  already ships `LICENSE-ggml` in `tona-LICENSES.txt` (glob `engine/third_party/LICENSE-*`). The generated table
  headers name licence and copyright.
- vLLM, ArcticInference, aiter, SGLang: Apache-2.0 (vLLM, ArcticInference) and MIT (aiter). No code was taken, so the
  Apache-2.0 duties (licence copy, NOTICE, change notices) do not arise; the patent clause matters only for taken
  code. Citing the PR numbers in comments is a courtesy.
- GPL, AGPL, LGPL, CC-BY-SA or similar: none found in the tree, the generated or fetched pieces, or the data.
- FlashAttention (BSD-3), pydash (MIT), boltons (BSD-3), CPython (PSF): see Table 3.

## Not verified (honest limits)

- No local copy of SGLang, FlashAttention, FlashInfer, MLX, Marlin, CUTLASS or Composable Kernel exists, and the code
  has no comment that claims a port from them. Only the comment mentions (PRs, papers) were compared where the
  upstream could be fetched (see Method 3). For the paper-based files (IndexCache, XShare, OEA, layered prefill,
  REAP) I did not compare the authors' own code, if it exists. llama.cpp PR 31286 (CRF admission) was named but not
  compared line by line.
- The line and shingle method finds text copies and light renaming. It cannot rule out that someone retyped an
  upstream algorithm from memory without any shared text; for such cases the answer rests on the comments and on
  the structure comparison above.
- `bench/mac/*`, `engine/backend/metal/**` and the Mac tools were compared against all of llama.cpp (including
  `ggml-metal`) and showed only idioms (`threadgroup_barrier(...)`, `int main(...)`); nothing else.
- Whether the release process ships a bundled OpenSSL: not checked on the build host.

## Actions, in order of weight

1. Add FTXUI's MIT licence to the release texts (`engine/third_party/LICENSE-ftxui`, README row, one sentence in the
   README licence section and in the `release-build.sh` comment). This is the only place where a licence duty is
   currently unmet.
2. `mtp_vocab.inc`: done, code32 dropped and the list regenerated from our own data (2026-11-13).
3. `mega_mtp.cpp`: one README line or comment for the ggml IQ4_NL scale search; reuse `tables::kvalues_iq4nl`.
4. Nothing for the remaining `inspired` and `independent` files; the existing citations in comments are enough.
