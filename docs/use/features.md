# What Tona can do today

What is on `main`, one line per feature. `partial`: some of it works; `planned`: not built. Rates are **decode** or
**prefill**, measured on the reference PC ([reference-platform.md](../backends/reference-platform.md)); speeds per model and context:
[BENCHMARKS.md](../../BENCHMARKS.md). Other hardware has no backend yet ([backends/README.md](../backends/README.md#tiers)).

## Models and tasks

- Three models: the **Coder** (Qwen3.8-Flash-Next, IQ1_M, 58 GB), **Ornith 1.5 35B-A3B** (Q4_K_M, 22 GB) and
  **Nemotron 3.5 Bolt 74B-A9B** (2-bit, 22 GB) ([how-it-works](../how-it-works.md#the-models)). Other architectures are
  rejected at load.
- Models are downloaded from Hugging Face at a pinned revision and SHA-256 ([toolchain/models.toml](../../toolchain/models.toml))
  and keep their own licence ([licenses](../contribute/licenses.md#models)).
- One offer file drives installer and client: tasks, languages, one recommendation and at most one alternative per task,
  strengths and weaknesses, each with a basis word (measured, provisional, vendor claim)
  ([data/modeldb/offer.json](../../data/modeldb/offer.json)).
- Two tasks work: coding with agents, and knowledge and research. Four are listed as "coming" (`planned`): documents and
  images, audio to text, reading aloud, image generation.
- 299 candidate model records with licence and fit ([data/modeldb/README.md](../../data/modeldb/README.md)).

## Setup (`tona setup`)

- Detects hardware and says plainly when there is no backend (exit code 3). Six pre-filled pages, so Enter alone gives a
  good setup ([installer.md](installer.md#the-pages)).
- Everything also works without a screen: `--task`, `--lang`, `--models`, `--dry-run`, `--json`, `--yes`, fixed exit codes
  ([installer.md](installer.md#commands-without-a-screen)).
- Downloads are verified by SHA-256 and resume.
- Release installation: signed release into `~/.local/share/tona`, systemd user unit `tona-server`, `tona` in
  `~/.local/bin` ([release.md](../maintain/release.md#installation-without-a-checkout)). Checkout installation:
  [development](../contribute/development.md#first-build).
- API key file `~/.config/tona/api-key`; listening for other PCs is off by default and needs the key.
- The priority in a shortage only sets the memory share; the engine does not yet throttle the other side (`partial`,
  [installer.md](installer.md#what-the-priority-does-today)). `tona uninstall` removes the installation.

## Client (`tona`)

- A control centre with four pages (Start, Models, Agents, Settings) and commands `status`, `models`, `use`, `config`,
  `doctor` with `--json` and fixed exit codes ([client.md](client.md)).
- `tona use <model>` restarts the server on another model; it asks first. Switching without cutting running answers is
  `planned` ([client.md](client.md#switching-the-model-restarts-and-updates)).
- Suggests another installed model when the project's languages fit it better; never switches by itself.
- `TONA_USE` and `TONA_PRIORITY` are stored; the engine reads only `TONA_PRIORITY`, in a developer start on the Mac
  (`partial`, [client.md](client.md#where-things-are)).

## Agents

- `tona claude`, `opencode`, `junie` and `pi` start the agent in a project folder against the local server with its
  tools, MCP servers and skills ([client](client.md#coding-agents)). Claude Code's claude.ai connectors do not load
  against a local server.
- `TONA_TOOLS=lean` gives a smaller first prompt; Claude Code compacts from 128K tokens and starts in `acceptEdits` mode.
- Effort per role (main, subagent, auxiliary) and a thinking budget per effort level
  ([environment.md](environment.md#server)). Quality against thinking length is not measured (`partial`).

## Server and APIs

- OpenAI API (`/v1/chat/completions`, plain and streamed) and Anthropic API (`/v1/messages`, `/v1/messages/count_tokens`),
  plus `GET /v1/models`, `/status` and `/health`; tool calls and thinking through both.
- Several conversations at once (`TONA_SEQS`); prompt cache over VRAM, RAM and SSD, a hit gives the cold answer with
  `TONA_DETERMINISTIC=1` ([prompt-cache.md](prompt-cache.md)); warm-up requests with `max_tokens: 0`
  ([environment.md](environment.md#warm-up-requests)).
- Context up to 262,144 tokens at the service default; the server default is 16,384 ([how-it-works](../how-it-works.md#assumptions)).
- Binds to `127.0.0.1`; an optional API key protects `/v1/*`; foreign `Host` or `Origin` headers are refused on a
  loopback bind ([SECURITY.md](../../SECURITY.md)).

## Memory and speed techniques

Details in [how-it-works](../how-it-works.md#how-the-engine-computes).

- Own HIP kernels; a persistent megakernel computes decode, WMMA kernels compute prefill (FP8 for the large dense
  weights; `TONA_PREFILL_FP8=0` returns to f16).
- Expert tiers across VRAM, pinned RAM and SSD; the CPU computes misses with AVX-512 ([expert-tiers.md](../architecture/expert-tiers.md)).
  Per-model expert usage profiles ([toolchain/profiles/](../../toolchain/profiles/)).
- Speculative decoding with the model's MTP head, drafts from the history, and the DFlash2 drafter as an option.
- Hot head of the KV cache in VRAM, the rest in pinned RAM, for Ornith and Bolt at long context.
- `TONA_DETERMINISTIC=1` for bit-exact comparison ([testing.md](../contribute/testing.md)).

## Safety and desktop friendliness

- No blocking GPU call while the megakernel waits on the CPU; waits pause the kernel after 100 ms
  (`TONA_MEGA_YIELD_MS`) ([AGENTS.md](../../AGENTS.md#gpu-safety-reference-backend)).
- VRAM for the desktop stays free (default 1,024 MB reserve) ([environment.md](environment.md#backend)). Giving VRAM back
  while running is a design only (`planned`, [vram-giveback.md](../architecture/vram-giveback.md)).
- `TONA_FREE_CUS` leaves compute units to the desktop at a cost in speed; `TONA_CLOCK_HOLD` keeps the GPU clock up between
  requests; both off by default.
- The unit limits memory (`MemoryMax`, no swap) and restarts a failed server at most three times in ten minutes.
- The measurement queue waits while a monitor is on and stops a GPU job within one second when the user returns
  ([shared-machine.md](../contribute/shared-machine.md)).

## Updates and releases

- `tona` checks for a release, installs it, restarts the server and goes back if it does not start; checks are off until
  switched on ([update.md](update.md)).
- Only releases with a minisign signature of one of two keys compiled into the program are accepted. The public keys are
  not set yet, so a build trusts no release today (`partial`, [engine/update/keys.h](../../engine/update/keys.h)).
- Releases are built reproducibly in a pinned container, attested, and published after the maintainer signs them locally
  ([release.md](../maintain/release.md)). The workflow has not run on GitHub yet (`partial`).

## Contributing and gates

- "Optimize Tona for my hardware" starts an agent playbook ([optimize-for-your-hardware.md](../contribute/optimize-for-your-hardware.md)).
- Code gates in every commit and in CI ([AGENTS.md](../../AGENTS.md#code-gates)); measurement tooling
  ([measurement.md](../contribute/measurement.md), [testing.md](../contribute/testing.md)); unit tests without model and GPU (`just test-cpu`),
  sanitizers, coverage floor and fuzzers.
- Backend blueprint and one page per candidate platform ([blueprint.md](../backends/blueprint.md), [backends/](../backends/)).
- macOS: the shared part builds and its tests pass; the Metal backend is in development and off by default (`partial`,
  [macos.md](macos.md)).
- Weekly paper scan with a coverage ledger ([research/papers.md](../../research/papers.md)).
- [CONTRIBUTING.md](../../CONTRIBUTING.md), [SECURITY.md](../../SECURITY.md).
