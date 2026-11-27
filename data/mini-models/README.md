# Mini-model headers

Headers of the real models, from which `tools/make_mini_model.py --from-header` builds the mini models
([docs/contribute/testing.md](../../docs/contribute/testing.md#the-mini-model)). Each `*.header.gguf.xz` is a GGUF file without tensor data,
xz-compressed: metadata, tokenizer, chat template and tensor directory. No weights.

| File | Source (Hugging Face, pinned revision) | Licence | Text |
|---|---|---|---|
| `coder.header.gguf.xz` | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF` @ `bba88856f8c3c0c9e40e45f82d05e228b814ac43`, `IQ1_M` parts 1 and 2 (tensor directory) | Qwen Community License 1.0 | [licenses/qwen-community-1.0.txt](licenses/qwen-community-1.0.txt) |
| `coder-mtp.header.gguf.xz` | `ggml-org/Qwen3.8-Flash-Next-GGUF` @ `91fb00353e766eb43944ed223665ea4fb87dad47`, `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` | Qwen Community License 1.0 | same |
| `ornith.header.gguf.xz` | `ornith-ai/Ornith-1.5-35B-A3B-GGUF` @ `12393612fd4f730ff5aadc23e9b8f9648aa49ceb`, `Ornith-1.5-35B-Q4_K_M.gguf` | MIT, Copyright (c) 2026 deepreinforce.ai | [licenses/mit-ornith.txt](licenses/mit-ornith.txt) |
| `bolt.header.gguf.xz` | `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-GGUF` @ `e0fecb69cb3e5e4c5e8a056fa2aec98c78e91246`, `NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf` | OpenMDW-1.1 | [licenses/openmdw-1.1.txt](licenses/openmdw-1.1.txt) |

The files match the pins in [toolchain/models.toml](../../toolchain/models.toml) (size and SHA-256 checked before the
export). The keys `general.license*`, `general.repo_url` and `general.quantized_by` of the source stay in each header;
the quantizer's `quantize.*` and `split.*` keys are removed.

These files are not under the MIT licence of Tona. Each carries the licence of the model it comes from. The licence
texts are copied unchanged from the publisher's repository.

- **Qwen Community License 1.0.** Copying and distribution are allowed if the notice and the text go with the copies
  (`licenses/qwen-community-1.0.txt`). Condition 2 (a separate licence for a model-as-a-service or AI work assistant
  business) binds whoever uses the Software commercially, not this repository.
- **MIT.** Distribution is allowed with the copyright and permission notice. The publisher's Hugging Face repositories
  ship no licence file; the text comes from the publisher's code repository `ornith-ai/Ornith-1` and the declaration
  is in the GGUF metadata ([evidence](../../research/2026-11-25-ornith-license.md)). The licences of the model it builds
  on are not stated.
- **OpenMDW-1.1.** Distribution is allowed if a copy of the agreement and the notices of origin stay with it.
