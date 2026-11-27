# The control centre: `tona`

One program (`engine/tools/tona.cpp`): the guided setup, control centre (four pages, 80 x 24), commands without a screen,
coding agents, background server, updater. Everything a page does is also a command.

## Pages

| Page | Key | What it does |
|---|---|---|
| Start | `1` | one row per conversation (`writes` = decode rate, `reads in` = prefill rate), graphics and memory, last requests, hints; load phases; if the server is down, what `tona doctor` found |
| Models | `2` | the offer (installed, to download, coming), filters, model card. `Enter` switches (asks), `d` downloads, `x` removes (asks) |
| Agents | `3` | agent, project folder, languages found, loaded model and a suggestion if another installed model suits the languages better, tools, thinking effort. `Enter` runs the command shown |
| Settings | `4` | rows of [engine/client/settings.cpp](../../engine/client/settings.cpp); `s` writes `tona.conf`, `z` fills in the installer's recommendation |

Keys: `Enter` main action, `↑↓` or `j k` move, `←→` change a value, `Space` on/off, `Esc` or `⌫` back, `Tab` between form
fields, `1`-`4` pages, `?` help, `q` close (asks whether the server stops too, unless saved under Settings), `w` switch
model. Below 80 x 24 only `q` works. `NO_COLOR` is honoured. Tona never
switches the model by itself; the Agents page only suggests.

### Switching the model, restarts and updates

All three ask first and name the running conversations. A model switch writes the settings and **restarts the server**,
which cuts running conversations. `Esc` while the new model loads switches back. Updates (release installation only): `u`
shows size and notes, then installs and restarts when no conversation runs; `c` turns checks on or off (press twice); `R`
goes back (press twice). See [update.md](update.md).

## Coding agents

In a project directory:

```sh
tona claude [dir]       # Claude Code against the local server in dir (default: here); options go to claude, e.g. tona claude -r
tona opencode           # OpenCode (provider "tona" as inline config, your opencode.json stays untouched)
tona junie              # Junie (custom model profile in work/tona/junie-models)
tona pi                 # pi (provider "tona" in ~/.pi/agent/models.json; other entries stay)
tona start, tona stop   # server in the background (also tona restart)
```

`tona` is in `~/.local/bin` after a release installation; in a checkout it is `work/build/engine/tona` (`just build`,
linked by `tona setup`, or `ln -sfn "$PWD/work/build/engine/tona" ~/.local/bin/tona`; `just top` runs it).

If no server runs, the agent commands start it and wait for the load (about 20 s), and on exit ask whether to stop it (it
pins about 25 GB of RAM and almost all VRAM); `tona stop` frees both. Each agent keeps its tools, MCP servers and skills.
Claude Code compacts from 128K (`TONA_COMPACT`) and starts with `acceptEdits` (`TONA_MODE`); `TONA_TOOLS=lean` gives a
faster first answer ([environment.md](environment.md#the-tona-command)). Claude Code's claude.ai connectors do not load
against a local server; local MCP servers do. Port: `TONA_PORT=9000 tona`. The key comes from `TONA_API_KEY_FILE`
(default `~/.config/tona/api-key`). A missing agent is an error (`tona: claude is not installed`, exit 1).

## Commands without a screen

Plain text without a terminal, `--json` for data. `tona` without a terminal prints `tona status`.

| Command | What it does |
|---|---|
| `tona status [--json]` | server state, model, conversations, decode rate, memory |
| `tona models [--task ID] [--json]` | the offer with its state: `loaded`, `installed`, `to_download`, `coming` (a task without a model is one `coming` row, exit 0; `--json` has `status` and `coming_note`) |
| `tona use <model> [--yes] [--no-restart]` | write the model into the settings and restart; with running conversations it asks (or needs `--yes`) |
| `tona config get <key>`, `set <key> <value>`, `list` | the settings table; values are checked |
| `tona config set agents <1-4>`, `set context <auto\|64K\|128K\|256K>` | restart the server like `use` (`--yes`, `--no-restart`) |
| `tona doctor [--json]` | one line per check, `ok` or `warn` with the fix: programs, toolchain, model files, expert profile, key, port, service, RAM and VRAM limits |
| `tona claude\|opencode\|junie\|pi [dir] [options]` | an agent in `dir`; the exit code is the agent's |
| `tona start`, `tona stop`, `tona restart` | the server in the background (checkout: `just`; release: unit `tona-server`, on macOS a background process) |
| `tona autostart on\|off\|status` | release: start the server at every login or not (Linux: unit enabled; macOS: LaunchAgent); off by default, `on` starts nothing before the next login |
| `tona setup [options]` | the guided setup again; every option of [installer.md](installer.md#commands-without-a-screen) works here (`--yes`, `--models`, `--print`, `--json`, `--dry-run`) |
| `tona update [--yes] [--rollback]` | the update of key `u` without the screen: check, show version and size, ask, install, restart; goes back by itself when the new version does not start ([update.md](update.md)) |
| `tona uninstall [--yes] [--models]` | stop the server, turn autostart off, remove the unit or LaunchAgent, the `tona` link, the settings and the install directory after one question; the models directory after a second one |
| `tona --version` | version, platform, backend |

| Exit code | Meaning |
|---|---|
| 0 | ok |
| 1 | the action failed (message on stderr) |
| 2 | unknown command, option, key or value |
| 3 | `status`: the server does not answer |
| 4 | `use`, `config set agents\|context`, `update`, `uninstall`: a question and neither `--yes` nor a terminal or "yes" to answer it |
| 5 | `doctor`: at least one check is `warn` |

## Where things are

- Settings: `work/tona.conf` (checkout) or `~/.config/tona/tona.conf` (release); the release file holds all settings;
  `tona start` reads it and starts the server with them, so the unit and the LaunchAgent hold none. `TONA_USE`, `TONA_PRIORITY`, `TONA_CLOSE` belong to the
  client. **The engine does not read `TONA_USE` and `TONA_PRIORITY` in an installed setup**; only a developer start on the
  Mac without `TONA_RAM_GB` reads `TONA_PRIORITY` ([environment.md](environment.md#tools)).
- Recently started agents: `launches.json` in `work/tona/` (checkout) or `~/.local/state/tona/` (release).
- The offer comes from `data/modeldb/offer.json` (generated into `engine/offer/` by `tools/gen_offer_data.py`).

## Code

`engine/client/`: `feed`/`poller.cpp` (data), `server_view`/`ui` (state), `keys*.cpp`, `view*.cpp`, `commands*.cpp`,
`agent*.cpp` (one per agent, shared `agent_launch.cpp`), `service_cmd.cpp`, `foreground.cpp` (`start --foreground`), `update_cmd.cpp`, `manage_cmd.cpp` (`uninstall`, `--version`), `control`, `app*.cpp` (terminal loop). All but
`app*.cpp` and the poller are tested without a terminal (`client_*`, `agent_commands`, `agent_launch`). No UI library.
