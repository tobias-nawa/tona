# Licences

## Tona

[MIT](../../LICENSE). Exceptions:

- `bench/corpus/python-stdlib-3.14.txt` comes from CPython and is under the PSF licence
  ([bench/corpus/LICENSE-python.txt](../../bench/corpus/LICENSE-python.txt)).
- The vendored and ported code in [engine/third_party/](../../engine/third_party/README.md) (cpp-httplib, nlohmann/json,
  parts of llama.cpp/ggml) is MIT under its own copyright.

Releases ship these texts as `tona-LICENSES.txt`. The shipped ROCm libraries are unmodified AMD binaries; their licence
texts are in `LICENSES.txt` next to them. AMD, Radeon and ROCm are trademarks of Advanced Micro Devices, Inc.

## Models

**Models are not part of Tona.** `tona setup` and `tools/get_model.py` download them from Hugging Face under the
model's own licence. The installer shows it in the model details (key `i`); records with evidence:
`data/modeldb/models/`.

- **Coder.** The weights in [toolchain/models.toml](../../toolchain/models.toml) derive from
  [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) under the **Qwen Community License 1.0**: free
  use, but a company running a model-as-a-service or "AI work assistant" business needs a separate Qwen licence. The
  quantized repositories label themselves differently (ISTA-DASLab: Apache-2.0 tag, text says the base licence is
  inherited; ggml-org: "other"; DFlash2 source: Apache-2.0): read the model cards before commercial use.
- **Ornith.** MIT as declared by the publisher; the licence file is missing on Hugging Face
  ([evidence](../../research/2026-11-25-ornith-license.md)).
- **Bolt.** OpenMDW-1.1.
- **Mini-model headers.** [data/mini-models/](../../data/mini-models/README.md) holds the tokenizer, chat template and tensor
  directory of the four models above (no weights) under their own licences, with the licence texts next to them.

## Expert profiles

The files in `toolchain/profiles/` are counts of how often each expert of a model was used on a mix of code and prose
(`scripts/make_expert_profile.py`); they hold no weights and no text of the models.

## Dependencies of the build, tests, gates and CI

Only FTXUI (linked into `tona`) and the ROCm libraries (above) ship in a release. Every name a requirement
file, `.pre-commit-config.yaml`, a workflow or a CMake file pulls in must stand here with its licence and be pinned by
hash, SHA or exact tag (`tools/gates/dependencies.py`, [docs/contribute/github.md](github.md#dependencies)). Licences are those the
package or repository declares (PyPI metadata, GitHub licence field), read on 2026-11-25.

| Name | Where | Licence |
|---|---|---|
| FTXUI (`ftxui`, `LICENSE-ftxui`) | FetchContent in `installer/setup.cmake` | MIT |
| cpp-httplib (`LICENSE-httplib`), nlohmann/json (`LICENSE-nlohmann-json`), llama.cpp / ggml (`LICENSE-ggml`) | vendored in `engine/third_party/` | MIT |
| rocm, rocm-profiler, rocm-sdk-core, rocm-sdk-devel, rocm-sdk-device-gfx1200, rocm-sdk-libraries | `toolchain/rocm-10.1.lock` (AMD wheels) | many open-source licences per component; texts in `LICENSES.txt` next to the libraries (not read per package) |
| pre-commit, cfgv, identify, nodeenv, platformdirs, python-discovery, virtualenv, filelock, pyyaml | `tools/gates/requirements.txt` | MIT (nodeenv: BSD) |
| distlib | same | PSF-2.0 |
| packaging | same, `tools/tests/requirements.txt` | Apache-2.0 or BSD-2-Clause |
| pytest, pluggy, iniconfig, colorlog, wcwidth | `tools/tests/requirements.txt`, `scripts/screenshot-requirements.txt` | MIT |
| coverage, diff-cover | `tools/tests/requirements.txt` | Apache-2.0 |
| gcovr, jinja2, markupsafe, lxml | same | BSD-3-Clause (jinja2: BSD) |
| pygments | same | BSD-2-Clause |
| lizard | `tools/tests/requirements.txt`, hook `functions` | MIT |
| numpy | `tools/tests/requirements.txt` | BSD-3-Clause and others (0BSD, MIT, Zlib, CC0-1.0) |
| chardet | same | 0BSD |
| pathspec | same | MPL-2.0 |
| pyte | `scripts/screenshot-requirements.txt` | LGPL-3.0 (used by a script, not shipped) |
| clang-tidy | `tools/gates/requirements-tidy-macos.txt` | Apache-2.0 with LLVM exception (PyPI says "Apache 2.0") |
| clang-format | hook `clang-format` | Apache-2.0 with LLVM exception (PyPI says "Apache 2.0") |
| jsonschema | hook `offer` | MIT |
| astral-sh/ruff-pre-commit | `.pre-commit-config.yaml` | Apache-2.0 (the hook repository; ruff itself is MIT) |
| shellcheck-py/shellcheck-py | `.pre-commit-config.yaml` | MIT (the wrapper; ShellCheck itself is GPL-3.0, run as a separate program) |
| rhysd/actionlint | `.pre-commit-config.yaml` | MIT |
| zizmorcore/zizmor-pre-commit | `.pre-commit-config.yaml` | the hook repository declares none; zizmor itself is MIT |
| github.com/zricethezav/gitleaks/v8 | `.pre-commit-config.yaml` (gitleaks, built from this module path) | MIT |
| Homebrew `libomp` (LLVM OpenMP runtime) | `scripts/release-build-macos.sh`, shipped as `libomp.dylib` in the macOS release | Apache-2.0 with LLVM exception (`licenses/llvm-LICENSE.TXT`) |
| Homebrew `openssl@3` | same, linked statically into the macOS `tona` | Apache-2.0 (`licenses/openssl-LICENSE.txt`) |
| actions/checkout, actions/setup-python, actions/cache, actions/attest, actions/upload-artifact, actions/download-artifact | `.github/workflows/` | MIT |
| github/codeql-action | `.github/workflows/codeql.yml`, `scorecard.yml` | MIT |
| ossf/scorecard-action | `.github/workflows/scorecard.yml` | Apache-2.0 |
