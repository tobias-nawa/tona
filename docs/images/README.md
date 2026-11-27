# Pictures

Terminal pictures come from [`scripts/screenshot-tui.py`](../../scripts/screenshot-tui.py): the program runs in an 80 x 24
pseudo-terminal with a neutral environment (fresh home, UTC, no user or host name) and the screen is written as SVG. The
script refuses a screen that shows the real home folder, user name or host name. The commands are in
[`scripts/make-screenshots.sh`](../../scripts/make-screenshots.sh); the first run builds the venv with the pinned terminal
emulator (`scripts/screenshot-requirements.txt`). Binaries come from `TONA_INSTALL_BIN` and `TONA_CLIENT_BIN`.

Every picture is written as SVG and as PNG (rasterized at 192 dpi with ImageMagick's RSVG delegate).

| Picture | Command |
|---|---|
| `installer-<page>.svg`: machine, tasks, languages, models, details, help, room, go | `scripts/make-screenshots.sh installer` |
| `tona-<page>.svg`: start, models, agents, settings, help (stub, port 8674) | `scripts/make-screenshots.sh client-stub` |

Each picture also exists as PNG. Not photographed: "server cannot be reached" (shows machine paths) and "update"
(needs a release installation). The installer runs in a throwaway home and stops before the download; hardware, free
memory and free disk are those of the machine that makes the picture. The stub serves
`engine/tests/data/client/status-decode.json` through `scripts/status-stub.py`; the number after "last" in the write-rate
line varies by one token per second.

## The control centre against the real server

```sh
PORT=8662 scripts/make-screenshots.sh client-live       # writes docs/images/tona-start.svg; OUT=file for another name
```

Needed for a good picture:

1. The server runs with the model loaded (the Start page says "Server running").
2. Two agents work at the same time, one writing and one reading, for example `tona opencode run '<a task that takes a
   minute>'` in a folder `shop` and `tona claude` in a folder `docs` (the folder name is shown as the project).
3. Start the command while both work; it waits 20 s and reads the screen.

Memory is shown as sums only (graphics memory, Tona, others, free); close what you do not want counted.
