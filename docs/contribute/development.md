# Development setup

Building Tona from a checkout. Users install the release with `tona` ([README](../../README.md#install)). Rules:
[AGENTS.md](../../AGENTS.md); sending a change: [CONTRIBUTING.md](../../CONTRIBUTING.md).

## Requirements

To run the engine: Linux with the amdgpu driver, access to `/dev/kfd` (group `render`) and a supported GPU
([README](../../README.md#hardware)). To build, test and pass the gates no GPU is needed:

- Python >= 3.11 with `venv`, git, cmake, ninja, `just` 1.40 or newer (Ubuntu 24.04 packages 1.21, which rejects the
  `justfile`: take the release binary from [casey/just](https://github.com/casey/just/releases)),
- GCC 14 or newer (`-march=znver5`; Ubuntu 24.04 has `g++-14`, Fedora and CI GCC 16) and the OpenSSL development files
  (`libssl-dev` on Debian and Ubuntu, `openssl-devel` on Fedora),
- about 12 GB of free disk space for the ROCm toolchain (`just lint` needs it for clang-tidy).

Without the reference GPU run `scripts/setup.sh --no-gpu`: same toolchain and gate tools, no step that needs the device.
macOS: [macos.md](../use/macos.md).

## First build

```sh
git clone https://github.com/tobias-nawa/tona && cd tona
scripts/setup.sh            # ROCm 10.1 into work/, self-tests, WMMA smoke test on the GPU; no sudo, system unchanged
source toolchain/env.sh     # activate the toolchain
just build                  # the engine into work/build/engine
```

Pins: ROCm in [toolchain/rocm-10.1.lock](../../toolchain/rocm-10.1.lock) (SHA-256), models in
[toolchain/models.toml](../../toolchain/models.toml).

**Model.** `tools/get_model.py coder-iq1_m` fetches the model at the pinned revision and SHA-256 (`--from DIR` verifies
and links an existing download). Or run the guided installer from the checkout; it also builds, runs a trial and
optionally sets up the service and the `tona` command:

```sh
work/build/engine/tona setup                 # guided (TUI), six pages, every page pre-filled (after just build)
```

In a checkout it works on the checkout and writes `work/tona.conf` (read by `just`, the service and `tona`); `--release`
installs the signed release instead ([installer.md](../use/installer.md)).

**Reference build of llama.cpp** (pinned tag, ROCm and Vulkan; oracle for parity tests, yardstick for speed). Vulkan
needs the SPIR-V headers and `glslang` (Fedora: `sudo dnf install spirv-headers-devel glslang`):

```sh
scripts/build-llamacpp.sh        # into work/build/llama.cpp-{hip,vulkan}
scripts/baseline.sh llamacpp-hip # baseline measurement; also llamacpp-vulkan
```

## Recipes

[`just`](https://github.com/casey/just) 1.40 or newer (`justfile`); `just` alone lists the recipes in groups. The installed
server uses port 8662 (8080 clashes with other dev servers); `just run` and `just test-gpu` take a free port and never
8662. A fixed port for `run`: `just run_port=9000 run`. `tona` uses `TONA_PORT` (default 8662).

| Group | Command | What it does |
|---|---|---|
| start | `just setup [--no-gpu]` | fetch the ROCm toolchain and the gate tools, install the git hooks |
| start | `just setup-tona` | build, then the guided setup of the checkout |
| every day | `just build` | build the engine |
| every day | `just fmt` | clang-format and ruff format on changed files |
| every day | `just lint` | all gates on your changes since `origin/main` |
| every day | `just test` | everything without GPU and model, what CI runs: unit tests, mini-model token check, pytest, installer tests |
| every day | `just test-gpu` | all tests with GPU and model, then the API test (agent round through both APIs) against its own server on a free port |
| every day | `just ready` | before review: `lint`, `test`, `test-gpu` when GPU and model are here, and the checks `tools/select_checks.py` names |
| run | `just run [options]` | server from the build in the foreground, 256K context (KV cache then in RAM), free port; options go to `tona-server`; smaller: `just ctx=65536 run` |
| run | `just tona` | the control centre ([client.md](../use/client.md)); starts the server if none runs |
| run | `just ask "question" [n]` | one answer without a server |
| measure | `just bench [runs]`, `just bench-pass` | decode measurement, median and range; verify pass without reloading |
| measure | `just ab ...` | A/B speed comparison ([testing.md](testing.md#speed-comparison-scriptsabpy)) |
| measure | `just baseline [target]` | measure the yardsticks (llama.cpp HIP, Vulkan) |
| deep | `just sanitize` | the unit tests with ASan and UBSan |
| deep | `just fuzz [seconds] [targets]` | fuzz the parsers (`api`, `gguf`, `minisig`; CI: `fuzz.yml`, 60 s per pull request, 30 min per night); one folder per target under `work/fuzz`, crashes in `<target>/findings` |
| deep | `just coverage [base]` | coverage with floor check; with a base ref also 80 % on changed lines |
| deep | `just cross-check` | compare the GPU path with the CPU reference (top-1 and logit deviation) |
| deep | `just lint-all` | the gates on every file and the whole history |
| machine | `just mem` | show RAM, page cache and VRAM use |
| machine | `just check-ram`, `just hang-repro` | test the RAM after installing it; reproduce a GPU hang (both ask first) |

Hidden: `start`, `stop` and `install-service` run the server as systemd service `tona-server`
(`journalctl --user -u tona-server -f`); `tona start`, `tona stop` and `tona setup` of a checkout call them, so you do not.
The old names (`check`, `serve`, `top`, ...) print where they went and exit 1.

Measurements: `scripts/ab.py` ([testing.md](testing.md#speed-comparison-scriptsabpy)); checks a change needs:
`python3 tools/select_checks.py` ([verification.md](verification.md)); settings: [environment.md](../use/environment.md).

## Run the server by hand

```sh
work/build/engine/tona-server work/models/coder-iq1_m/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf \
    --mtp work/models/draft/mtp-Qwen3.8-Flash-Next-Q8_0.gguf --port 8662 \
    --profile work/profile/experts-coder.bin
tools/server_smoke.py http://127.0.0.1:8662     # agent round over both APIs
```

Claude Code against it, without the `tona` script:

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:8662 ANTHROPIC_AUTH_TOKEN=local ANTHROPIC_MODEL=qwen3.8-flash-next-coder \
CLAUDE_CODE_ATTRIBUTION_HEADER=0 claude
```

`CLAUDE_CODE_ATTRIBUTION_HEADER=0` stops Claude Code from putting a per-request billing line before the system prompt,
which would defeat the prefix cache (the server drops that line too). The system prompt has about 20,000 tokens; at
about 1,650 tok/s prefill ([BENCHMARKS.md](../../BENCHMARKS.md)) the first answer takes about 12 s, then the prompt cache
applies ([prompt-cache.md](../use/prompt-cache.md)).

## Layout of the repository

| Path | Content |
|---|---|
| [engine/](../../engine/) | C++23/HIP: GGUF, CPU reference and experts (`cpu/`), megakernel (`gpu/mega`), `server/`, `client/`, `update/`, `tona-*` tools, tests |
| [installer/](../../installer/) | the code of `tona setup`; its tests build as a project of their own |
| [docs/](..) | documentation, index [README.md](../README.md) |
| [research/](../../research/) | notes and measurements, [research/README.md](../../research/README.md) (working archive) |
| [bench/](../../bench/) | harness `ibench`: `bench/run run --target llamacpp-hip` |
| [toolchain/](../../toolchain/), [scripts/](../../scripts/) | pinned ROCm and models, setup, baselines, release scripts |
| [tools/](../../tools/) | `gguf_info.py`, `get_model.py`, `hw_report.py`, `quiet_check.py`, the code gates |
| [data/modeldb/](../../data/modeldb/) | model database and offer data |
