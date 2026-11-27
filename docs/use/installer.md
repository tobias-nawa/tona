# Setup (`tona setup`)

`tona` starts it when nothing is installed; `tona setup` runs it again. Detects the machine, proposes one model per task with its estimated speed, downloads and verifies what you choose. Six
pre-filled pages, so Enter alone gives a good setup; nothing large is downloaded unasked. Every page also works from the
command line (`installer/core/flow.h`). Release against checkout: [release.md](../maintain/release.md#installation-without-a-checkout),
[README](../../README.md#install). Models come from `data/modeldb/offer.json`, sizes from `toolchain/models.toml`.

## The pages

| Page | What it asks | Pre-filled |
|---|---|---|
| Machine | nothing: graphics card, processor, memory, free space, class | detected |
| Tasks | what you want to do; unbuilt tasks are "coming" and cannot be switched on | coding and knowledge |
| Languages | only with coding: your languages; strongest model per language | none |
| Models | per task the recommendation (on) and at most one alternative (off), with size, decode range, strengths; total download | the recommendations |
| Room | memory Tona may take, priority in a shortage, who may connect, start at login, parallel agents, context per agent | balanced, this PC only, no autostart, 2 agents, auto |
| Go | steps, progress, next commands | |

**Keys:** Enter main action, arrows or `j` `k` move, Space on/off, Esc or Backspace back, `?` help, `q` quit (not while a
step runs), `i` model details, `e` advanced settings, `p` recheck free space, `l` log. Below 80 x 24 only a "too small"
notice shows. `NO_COLOR` switches colours off.

A language without measurement says "still open". Download time assumes 100 Mbit/s. Speeds are ranges:
[speed estimate](#speed-estimate).

### Classes

| Machine | Class | Offer |
|---|---|---|
| PC, 16 GB VRAM, 64 GB RAM | `pc-16gb-vram-64gb-ram` | Coder, Ornith, Bolt |
| Mac, 32 GB or more (chip sets the speed) | `mac-32gb`, `mac-64gb` from 64 GB | Coder only (the Metal backend runs no other model) |

A Mac below 32 GB shows the nearest Mac class and the memory it has.

### Finish

After the steps, the Finish screen lists what changes outside Tona's folders. Both boxes are ticked, Space switches,
Enter applies:

- Add `~/.local/bin` to `PATH`. Shown only when the folder is missing from `PATH`; it names the files.
- Check for updates once a day. Shown only in a release installation ([update](update.md)).

The PATH row adds one block between `# >>> tona >>>` and `# <<< tona <<<` to the profile of your login shell and of every
other shell that already has one: `~/.zshrc` (or `$ZDOTDIR/.zshrc`), `~/.bashrc` (`~/.bash_profile` on macOS),
`~/.config/fish/conf.d/tona.fish`, `~/.profile` for other shells. A second run adds nothing, a linked file is edited at
its target, a file that cannot be written is skipped. `tona uninstall` removes the blocks.

Without a screen nothing is written to a profile unless you pass `--modify-path`; `--no-modify-path` forbids it. The last
page says "added to PATH in ...", or, when the box was empty, shows the line to add by hand. It also lists the four
agents (Claude Code, OpenCode, Junie, Pi) and marks those not found on `PATH`.

### Speed estimate

The list shows decode at 16K context as a range, `unknown` where the backend cannot estimate. `i` adds prefill, the
inputs and the anchor. No exact number is shown.

Decode of these models is bound by the bytes read per token. The estimate takes a rate measured on an anchor machine and
multiplies it by the ratio of the seconds per token, anchor to this machine: dense weights and the experts that fit in
GPU memory at its bandwidth, the experts in RAM at the RAM bandwidth, the rest at the drive's speed
(`installer/core/speed.cpp`). Prefill scales with the GPU's compute units.

| Input | From |
|---|---|
| GPU memory bandwidth, compute units | table of the backend module by GPU (PCI id) or Apple chip |
| Experts in GPU memory | the VRAM limit (PC) or the GPU working set (Mac) of the settings |
| RAM bandwidth | measured once per run, 512 MiB read by 2 to 8 threads (PC); unified memory on the Mac |
| RAM for experts | the RAM limit less KV cache and engine overhead |
| Drive speed | measured on the model file, else by drive kind (PC); measured on the model file (Mac) |

| Anchor | Measured on |
|---|---|
| PC | RX 9060 XT 16 GB, Ryzen 9 9950X, 64 GB DDR5-6000 (rates of `data/modeldb/offer.json`) |
| Mac | M1 Max 32 GB (`research/measurements/2026-11-26-mac-release-rates.md`) |

**Agents and context.** The rate shown on the Room page, in the plan and in `--json` follows the chosen agents and context
(`installer/core/speed.cpp`, `project_load`): one pass reads the dense weights once, the union of the experts the agents
route to (it grows with the agents) and every agent's KV cache. The rate of each agent is the anchor's rate times the
ratio of the pass times; "together" is that times the agents. KV that takes the experts' room in RAM (PC) or in the GPU
working set (Mac) moves experts to the drive, which shows as a step. The KV size per position of Ornith and Bolt is fitted
to their measured decode at 2K and 64K (`installer/modules/hip_gfx1200.cpp`); a new measured point replaces an estimate
when it is added to the anchors (`data/modeldb/offer.json` rates; the Mac's in `installer/modules/metal_apple.cpp`).
In `--json` the rates under the chosen settings are `models[].speed.settings` (`agents`, `context_tokens`, `per_agent`,
`together`); `models[].speed.decode` stays the rate at 16K with one agent. The drive input of an anchor is what the
setup's own drive probe measures on the anchor machine (Mac: expert-style reads, 6.7 GB/s on the M1 Max), so the
estimate on the anchor's hardware is the anchor's rate.
The band widens by 15 points for more than one agent and for a context more than 4 times away from the anchor's.

**Width:** plus or minus 15 % for the anchor's class (same GPU and VRAM size, as much RAM, RAM bandwidth within 20 %;
M1 Max with 32 GB or more), plus or minus 30 % for another GPU or chip of the backend. Each input the detection did not
find adds 15 points (60 at most) and is named in the details; without the GPU bandwidth there is no estimate. Prefill
adds 10 points. A model without a measurement on the backend has no estimate (the Mac has two). The split of
experts and dense weights of Ornith and Bolt is derived from their parameter counts, not read from the files.

### What the priority does today

The answer (work first, balanced, Tona first) is stored as `TONA_PRIORITY` (`work`, `balanced`, `tona`) and sets the
memory you keep for your programs (on 64 GB: 27, 20 or 8 GB; the system keeps 4 GB). The engine does not throttle the
other side: Tona is hard-limited to its share (`TONA_RAM_GB`, `MemoryMax`). `TONA_USE` (`desktop` or `server`) records
whether the PC is also used for other work.

### Parallel agents and context

| Field | Values | Default |
|---|---|---|
| Parallel agents | 1 to 4 | 2 |
| Context per agent | auto, 64K, 128K, 256K (Mac: 32K, one agent) | auto |

Auto keeps the RAM the experts need to stay pinned (31.6 GB for the Coder) and gives each agent the largest context that
fits in the rest, at least 32K. A fixed size is used as chosen; if the experts then do not fit, the page says so.
The speed line under the memory bars follows both fields. Same settings in `tona config`: `agents`, `context`
([client](client.md)).

### Advanced settings (`e`)

RAM and VRAM limit, drafter, effort, shared experts (OEA), subagent
effort, keeping the GPU warm, port, API key, checkpoints on the SSD, the command `tona`. A digit edits a number row; Enter
keeps, Esc drops. Changing "Room" recomputes the RAM limit.

### States

- **No suitable graphics card:** nothing is installed; a processor-only setup is shown, not selectable.
- **Too little space:** needed against free (plus a 5 GB checkpoint buffer), and what fits: a shorter list, a smaller
  alternative, another drive (release installation). `p` checks again. Nothing is downloaded before it fits.
- **Download failed:** complete files are kept, a partial file resumes; "resume" or "quit". Closing the window or Ctrl+C
  ends the download after the running step; the next `tona setup` continues.

## Commands without a screen

```sh
tona setup --task coding --lang rust --dry-run      # the plan for this machine, nothing changes
tona setup --models coder,bolt --yes                # downloads exactly these models and sets up
tona setup --task coding --json                     # the plan as JSON on stdout
tona setup --port 9100 --yes                         # another server port (the page proposes the next free one)
tona setup --print                                  # detect and recommend only
```

| Option | Meaning |
|---|---|
| `--task LIST` | `coding`, `knowledge`; a coming task is a usage error (exit 6) |
| `--lang LIST` | `python`, `typescript`, `javascript`, `rust`, `csharp`, `go`, `java`, `cpp`, `php`, `dart`, `bash` |
| `--models LIST` | `coder`, `ornith`, `bolt` (card name or id); the tasks follow |
| `--priority P`, `--use U` | the answers of the page Room |
| `--updates on\|off` | the question "Check for updates once a day?" on the last page (release; default on) |
| `--json`, `--dry-run`, `--print` | change nothing; `--json` prints only the JSON |
| `--yes` | no questions; downloads exactly what is named, else the coding recommendation or the existing configuration |
| `--help-dev` | the options for developers: `--repo`, `--release`, `--model-from`, `--dump-pages` (every page as text at 80 x 24) |

Without a terminal the output is plain English text. A named choice without `--yes`, `--dry-run` or `--json` opens the
guided setup pre-filled (terminal) or is a usage error (no terminal).

### Exit codes

| Code | Meaning |
|---|---|
| 0 | done; with a plan option: the plan can run |
| 1 | a step failed, or the plan has errors |
| 2 | usage error; no terminal and nothing to do |
| 3 | no backend for this hardware |
| 4 | not enough free space |
| 5 | a download failed (the next run resumes) |
| 6 | a named task is not available yet |
| 7 | a required program is missing |

### The JSON plan

Schema 1: `hardware` (`class`, `class_exact`, `ram_bytes`, `disk_free_bytes`, `backend`), `tasks`, `languages`, `models`
(`id`, `role` `primary` or `additional`, `size_bytes`, `present`, `speed`: `"unknown"` or `decode` and `prefill` with
`low`, `high` (tok/s), `context_tokens`, `width`, `anchor`, `unknown_inputs`), `drafter`, `download` (`bytes`, `seconds`, `assumed_mbit_per_s`), `space` (`free_bytes`, `buffer_bytes`, `fits`),
`room`, `notes`, `packages_ok`, `steps`, `exit_code` (what the same run would return).

## Configuration

The installer writes `TONA_MODELS` (the first model starts), `TONA_LANGS`, `TONA_USE`, `TONA_PRIORITY`, `TONA_SEQS`, `TONA_CTX` (the size auto resolved to), `TONA_CTX_AUTO` and the advanced
settings to `work/tona.conf` (checkout) or `~/.config/tona/tona.conf` (release). Switching models is the client's job.

## Tests

`installer/tests/test_flow.cpp` (ctest label `unit`, no network, no model):

```sh
cmake -S installer -B work/build/installer -G Ninja && cmake --build work/build/installer && ctest --test-dir work/build/installer
```
