#!/usr/bin/env python3
"""A picture of a terminal program as SVG: the program runs in a pseudo-terminal, keys are sent, the screen is read
with its colours and written as a small, crisp vector file.

    scripts/screenshot-tui.py --out docs/images/x.svg [--size 100x30] [--keys "down enter wait:0.5"] -- PROGRAM [ARGS]

The same input gives the same file: the screen is read after the output has been quiet for --settle seconds, the SVG
has no timestamps and no random ids, and the program runs with a neutral environment (a fresh HOME, no user or host
name) so that nothing of this machine reaches the picture. The picture is refused if the screen still shows the real
home folder, the user name or the host name.

In the program's arguments {home} is the neutral home folder (for a --install-dir that must not be a real one).

Keys (separated by spaces or commas): enter tab btab esc space backspace up down left right home end pgup pgdn,
ctrl-<letter>, wait:<seconds> (a pause, also after every key: --pause), text:<characters> (typed as they are), or one
character. The terminal emulator is pyte (pinned in scripts/screenshot-requirements.txt); the venv is built by
`python3 -m venv work/venv-screenshot && work/venv-screenshot/bin/pip install --require-hashes -r
scripts/screenshot-requirements.txt`. Everything but the capture (SVG, keys, privacy check) works without pyte.
"""

from __future__ import annotations

import argparse
import fcntl
import getpass
import os
import re
import select
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import termios
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

# One theme, dark: the 16 ANSI colours of the terminals people use (close to VS Code's dark theme), so the picture
# looks like the program does in a terminal.
PALETTE = {
    "black": "#1e1e1e", "red": "#f14c4c", "green": "#23d18b", "brown": "#e5c07b",
    "blue": "#3b8eea", "magenta": "#d670d6", "cyan": "#29b8db", "white": "#e5e5e5",
    "brightblack": "#808080", "brightred": "#f78b8b", "brightgreen": "#5af0aa", "brightbrown": "#f5e08a",
    "brightblue": "#6fb0ff", "brightmagenta": "#e69be6", "brightcyan": "#5fd4ec", "brightwhite": "#ffffff",
}  # fmt: skip
DEFAULT_FG = "#d4d4d4"
DEFAULT_BG = "#1e1e1e"
FRAME_BG = "#2b2b2b"
FRAME_LINE = "#3c3c3c"
DOTS = ("#ff5f56", "#ffbd2e", "#27c93f")
FONT = "'JetBrains Mono','Cascadia Mono','DejaVu Sans Mono','Menlo','Consolas','Liberation Mono',monospace"
SETTLE_LIMIT = 5.0  # a screen that keeps redrawing (a clock, a graph) is read after this long, quiet or not
FONT_SIZE = 14
CELL_W = 8.4  # 0.6 em, the advance of the fonts in the stack; textLength pins it for any other font
CELL_H = 18
PAD = 16
BAR = 34
# xterm's colours of 0-15 in the 256 palette (pyte's table) back to the names of the theme
XTERM16 = [
    "000000", "cd0000", "00cd00", "cdcd00", "0000ee", "cd00cd", "00cdcd", "e5e5e5",
    "7f7f7f", "ff0000", "00ff00", "ffff00", "5c5cff", "ff00ff", "00ffff", "ffffff",
]  # fmt: skip
ANSI_ORDER = ["black", "red", "green", "brown", "blue", "magenta", "cyan", "white"]
FAINT = "faint:"  # marks a colour drawn while SGR 2 (dim) was on: pyte keeps no such attribute of its own
ALIASES = {"bfightmagenta": "brightmagenta"}  # pyte 0.8.2 misspells background 105

KEYS = {
    "enter": "\r", "tab": "\t", "btab": "\x1b[Z", "esc": "\x1b", "space": " ", "backspace": "\x7f",
    "up": "\x1b[A", "down": "\x1b[B", "right": "\x1b[C", "left": "\x1b[D",
    "home": "\x1b[H", "end": "\x1b[F", "pgup": "\x1b[5~", "pgdn": "\x1b[6~",
}  # fmt: skip


@dataclass(frozen=True)
class Cell:
    """One screen cell with its colours resolved (None: the terminal default); data "" is the tail of a wide char."""

    data: str = " "
    fg: str | None = None
    bg: str | None = None
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strike: bool = False
    faint: bool = False

    def style(self) -> tuple:
        return (self.fg, self.bg, self.bold, self.italic, self.underline, self.strike, self.faint)


def resolve_color(name: str) -> str | None:
    """A pyte colour (name, 6 hex digits or "default") as #rrggbb, None for the default."""
    name = ALIASES.get(name, name)
    if name == "default":
        return None
    if name in PALETTE:
        return PALETTE[name]
    if re.fullmatch(r"[0-9a-f]{6}", name):
        if name in XTERM16:
            i = XTERM16.index(name)
            return PALETTE[("bright" if i >= 8 else "") + ANSI_ORDER[i % 8]]
        return "#" + name
    raise ValueError(f"unknown colour {name!r}")


def parse_keys(spec: str, pause: float) -> list[tuple[str, float]]:
    """The key list as (bytes to send, seconds to wait afterwards)."""
    out: list[tuple[str, float]] = []
    for tok in re.split(r"[\s,]+", spec.strip()):
        if not tok:
            continue
        if tok.startswith("wait:"):
            out.append(("", float(tok[5:])))
        elif tok.startswith("text:"):
            out.extend((c, 0.02) for c in tok[5:])
            out.append(("", pause))
        elif tok in KEYS:
            out.append((KEYS[tok], pause))
        elif re.fullmatch(r"ctrl-[a-z]", tok):
            out.append((chr(ord(tok[5]) - 96), pause))
        elif len(tok) == 1:
            out.append((tok, pause))
        else:
            raise ValueError(f"unknown key {tok!r}")
    return out


def cell_width(ch: str) -> int:
    return 2 if unicodedata.east_asian_width(ch) in "WF" else 1


def runs(row: list[Cell]) -> list[tuple[int, int, str, tuple]]:
    """Neighbouring cells of one style as (first column, columns, text, style); blanks without a background stay out."""
    out: list[tuple[int, int, list[str], tuple]] = []
    for x, c in enumerate(row):
        if c.data == "":
            if out:  # the tail of a wide character belongs to the run of its head
                out[-1] = (out[-1][0], out[-1][1] + 1, out[-1][2], out[-1][3])
            continue
        st = c.style()
        if out and out[-1][3] == st and out[-1][0] + out[-1][1] == x:
            out[-1] = (out[-1][0], out[-1][1] + 1, out[-1][2] + [c.data], st)
        else:
            out.append((x, 1, [c.data], st))
    return [(x, n, "".join(t), st) for x, n, t, st in out]


def mix(color: str, into: str, share: float) -> str:
    """The colour `share` of the way to `into` (faint text is its colour at half strength)."""
    a, b = (tuple(int(c[i : i + 2], 16) for i in (1, 3, 5)) for c in (color, into))
    return "#" + "".join(f"{round(x + (y - x) * share):02x}" for x, y in zip(a, b, strict=True))


def svg_text(text: str) -> str:
    """Escaped text with every space as &#160;: browsers that show the SVG as an <img> (GitHub) collapse runs of
    spaces in <text> despite xml:space, and textLength then spreads the letters. A character reference, not a raw
    U+00A0, so that no invisible character lands in the file."""
    return escape(text).replace(" ", "&#160;")


def row_svg(y: int, row: list[Cell], x0: float, y0: float) -> tuple[list[str], list[str]]:
    """(background rects, text elements) of one row, as strings."""
    rects: list[str] = []
    texts: list[str] = []
    for x, n, text, (fg, bg, bold, italic, underline, strike, faint) in runs(row):
        left, top = x0 + x * CELL_W, y0 + y * CELL_H
        if bg:
            rects.append(f'<rect x="{left:.1f}" y="{top:.1f}" width="{n * CELL_W:.1f}" height="{CELL_H}" fill="{bg}"/>')
        if not text.strip():
            continue
        color = fg or DEFAULT_FG
        if faint:
            color = mix(color, bg or DEFAULT_BG, 0.45)
        attrs = [f'x="{left:.1f}"', f'y="{top + 13.5:.1f}"', f'textLength="{n * CELL_W:.1f}"', f'fill="{color}"']
        if bold:
            attrs.append('font-weight="bold"')
        if italic:
            attrs.append('font-style="italic"')
        deco = " ".join(d for d, on in (("underline", underline), ("line-through", strike)) if on)
        if deco:
            attrs.append(f'text-decoration="{deco}"')
        texts.append(f"<text {' '.join(attrs)}>{svg_text(text)}</text>")
    return rects, texts


def render_svg(rows: list[list[Cell]], title: str = "") -> str:
    """The screen as an SVG document: a dark window frame with the grid inside. No clock, no random ids."""
    cols, nrows = max(len(r) for r in rows), len(rows)
    width, height = cols * CELL_W + 2 * PAD, nrows * CELL_H + 2 * PAD + BAR
    x0, y0 = PAD, BAR + PAD
    rects: list[str] = []
    texts: list[str] = []
    for y, row in enumerate(rows):
        r, t = row_svg(y, row, x0, y0)
        rects += r
        texts += t
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="{escape(title or "terminal")}">'
    )
    frame = [
        f'<rect width="{width:.0f}" height="{height:.0f}" rx="10" fill="{FRAME_BG}" stroke="{FRAME_LINE}"/>',
        f'<rect x="{PAD / 2}" y="{BAR}" width="{width - PAD:.0f}" height="{height - BAR - PAD / 2:.0f}" rx="4" '
        f'fill="{DEFAULT_BG}"/>',
    ]
    frame += [f'<circle cx="{18 + 18 * i}" cy="{BAR / 2 + 1}" r="5.5" fill="{c}"/>' for i, c in enumerate(DOTS)]
    if title:
        frame.append(
            f'<text x="{width / 2:.0f}" y="{BAR / 2 + 5}" text-anchor="middle" fill="#9d9d9d" '
            f'font-size="12">{svg_text(title)}</text>'
        )
    style = f'<g font-family="{FONT}" font-size="{FONT_SIZE}" xml:space="preserve">'
    return "\n".join([head, *frame, *rects, style, *texts, "</g>", "</svg>", ""])


def forbidden_words(home: str) -> list[str]:
    """What must not be on a screen: the real home folder, the user name, the host name, the temporary home."""
    host = socket.gethostname()
    words = [str(Path.home()), os.environ.get("HOME", ""), getpass.getuser(), host, host.split(".")[0], home]
    return sorted({w for w in words if len(w) >= 3})


def check_privacy(rows: list[list[Cell]], words: list[str]) -> list[str]:
    """The forbidden words found on the screen (text only, case sensitive)."""
    text = "\n".join("".join(c.data for c in r) for r in rows)
    return [w for w in words if w in text]


def squeeze(rows: list[list[Cell]], keep: int) -> list[list[Cell]]:
    """Runs of blank rows (no text, no background) cut down to `keep` rows: the gap above a key line."""
    out: list[list[Cell]] = []
    blank = 0
    for row in rows:
        if any(c.data.strip() or c.bg for c in row):
            blank = 0
        else:
            blank += 1
            if blank > keep:
                continue
        out.append(row)
    return out


def neutral_env(home: str, lang: str, extra: dict[str, str]) -> dict[str, str]:
    """The environment of the program: nothing of this machine but the PATH to find it."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": home, "TERM": "xterm-256color",
        "LANG": lang, "LC_ALL": lang, "USER": "dev", "LOGNAME": "dev", "TZ": "UTC",
    }  # fmt: skip
    env.update(extra)
    return env


def load_pyte():
    try:
        import pyte
    except ImportError:
        sys.exit(
            "pyte is missing: python3 -m venv work/venv-screenshot && work/venv-screenshot/bin/pip install "
            "--require-hashes -r scripts/screenshot-requirements.txt, then run this script with that python"
        )
    return pyte


def faint_screen(pyte):
    """pyte's Screen plus SGR 2 (dim), which FTXUI uses for every secondary line and pyte drops."""

    class FaintScreen(pyte.Screen):
        faint = False

        def select_graphic_rendition(self, *attrs, **kwargs):
            rest, i = [], 0
            attrs = attrs or (0,)
            while i < len(attrs):
                a = attrs[i]
                span = 1
                if a in (38, 48) and i + 1 < len(attrs):  # 38;5;n and 38;2;r;g;b: the 2 in there is no SGR 2
                    span = 3 if attrs[i + 1] == 5 else 5
                if span == 1 and a == 2:
                    self.faint = True
                else:
                    self.faint = self.faint and a not in (0, 22)
                    rest.extend(attrs[i : i + span])
                i += span
            super().select_graphic_rendition(*rest, **kwargs)

        def draw(self, data):
            real = self.cursor.attrs
            if self.faint:
                self.cursor.attrs = real._replace(fg=FAINT + real.fg)
            super().draw(data)
            self.cursor.attrs = real

    return FaintScreen


# Sequences FTXUI emits that pyte does not understand and would print as literal text:
# - DCS/APC/PM/SOS strings (ESC P/ _/ ^/ X ... ESC \), e.g. DECRQSS (ESC P $ q SP q ESC \)
# - DECRQM (CSI ? n $ p), DECSCUSR (CSI n SP q), Kitty keyboard / XTVERSION queries (CSI [><=] ... [uq])
UNKNOWN_CSI = re.compile(rb"\x1b\[(?:\?[0-9;]*\$p|[0-9]* q|[><=][0-9;]*[uq])")
STRING_INTRO = frozenset(b"P_^X")


class EscapeFilter:
    """Strips DCS/APC/PM/SOS strings and unknown CSI queries across read chunk boundaries."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, chunk: bytes) -> bytes:
        self._buf.extend(chunk)
        out, buf = bytearray(), self._buf
        i, n = 0, len(buf)
        while i < n:
            esc = buf.find(0x1B, i)
            if esc < 0:
                out.extend(buf[i:])
                i = n
                break
            out.extend(buf[i:esc])
            if esc + 1 == n:
                i = esc
                break
            nxt = buf[esc + 1]
            if nxt in STRING_INTRO:
                st = buf.find(b"\x1b\\", esc + 2)
                if st < 0:
                    i = esc
                    break
                i = st + 2
            elif nxt == ord("["):
                j = esc + 2
                while j < n and not (0x40 <= buf[j] <= 0x7E):
                    j += 1
                if j == n:
                    i = esc
                    break
                seq = bytes(buf[esc : j + 1])
                if not UNKNOWN_CSI.fullmatch(seq):
                    out.extend(seq)
                i = j + 1
            else:
                out.extend(buf[esc : esc + 2])
                i = esc + 2
        del buf[:i]
        return bytes(out)


# Runs in the child after start_new_session: takes the pty on stdin as controlling terminal, then becomes the program.
# preexec_fn would do the same in the forked child, but it is not safe while other threads run.
CONTROLLING_TTY_EXEC = (
    "import fcntl, os, sys, termios; fcntl.ioctl(0, termios.TIOCSCTTY, 0); os.execvp(sys.argv[1], sys.argv[1:])"
)


class Session:
    """The program in a pseudo-terminal of the given size, its output fed into a pyte screen."""

    def __init__(self, argv: list[str], size: tuple[int, int], env: dict[str, str], cwd: str):
        pyte = load_pyte()
        self.master, slave = os.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", size[1], size[0], 0, 0))
        self.screen = faint_screen(pyte)(*size)
        master = self.master
        self.screen.write_process_input = lambda data: os.write(master, data.encode())
        self.stream = pyte.ByteStream(self.screen)
        self.escape_filter = EscapeFilter()
        self.proc = subprocess.Popen(
            [sys.executable, "-c", CONTROLLING_TTY_EXEC, *argv],
            stdin=slave, stdout=slave, stderr=slave, env=env, cwd=cwd, start_new_session=True,
        )  # fmt: skip
        os.close(slave)

    def _read_once(self, timeout: float) -> bool | None:
        ready, _, _ = select.select([self.master], [], [], timeout)
        if not ready:
            return False
        try:
            data = os.read(self.master, 65536)
        except OSError:
            return None
        if not data:
            return None
        self.stream.feed(self.escape_filter.feed(data))
        return True

    def drain(self, quiet: float, limit: float = 10.0) -> None:
        """Reads until the output has been quiet for `quiet` seconds (at most `limit`)."""
        end, last = time.monotonic() + limit, time.monotonic()
        while time.monotonic() < end and time.monotonic() - last < quiet:
            got = self._read_once(0.05)
            if got is None:
                return
            if got:
                last = time.monotonic()

    def pump(self, seconds: float) -> None:
        """Reads for exactly `seconds`, whatever the program draws meanwhile (a pause between two keys)."""
        end = time.monotonic() + seconds
        while (left := end - time.monotonic()) > 0:
            if self._read_once(min(left, 0.05)) is None:
                return

    def send(self, data: str) -> None:
        os.write(self.master, data.encode())

    def cells(self) -> list[list[Cell]]:
        scr = self.screen
        return [[to_cell(scr.buffer[y][x]) for x in range(scr.columns)] for y in range(scr.lines)]

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)  # only our own child
            try:
                self.proc.wait(3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        os.close(self.master)


def to_cell(ch) -> Cell:
    faint = ch.fg.startswith(FAINT)
    fg, bg = resolve_color(ch.fg.removeprefix(FAINT)), resolve_color(ch.bg)
    if ch.reverse:
        fg, bg = bg or DEFAULT_BG, fg or DEFAULT_FG
    return Cell(ch.data, fg, bg, ch.bold, ch.italics, ch.underscore, ch.strikethrough, faint)


def parse_size(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)x(\d+)", text)
    if not m:
        raise argparse.ArgumentTypeError("size is COLSxROWS, e.g. 100x30")
    return int(m.group(1)), int(m.group(2))


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="the SVG file to write")
    ap.add_argument("--size", type=parse_size, default=(100, 30), help="COLSxROWS of the terminal (default 100x30)")
    ap.add_argument("--keys", default="", help="keys to send, see the header of this script")
    ap.add_argument("--pause", type=float, default=0.4, help="seconds to wait after every key")
    ap.add_argument("--settle", type=float, default=1.0, help="seconds of quiet output before the screen is read")
    ap.add_argument("--lang", default="en_US.UTF-8", help="LANG of the program")
    ap.add_argument("--title", default="", help="title in the window bar")
    ap.add_argument("--cwd", default="", help="working directory (default: a fresh folder named projects/shop)")
    ap.add_argument("--home-parent", default="/var/tmp", help="parent for the neutral home (disk, not RAM)")
    ap.add_argument("--squeeze", type=int, default=-1, metavar="N", help="cut blank-row runs to N rows")
    ap.add_argument("--env", action="append", default=[], metavar="NAME=VALUE", help="environment variable to add")
    ap.add_argument("--show", action="store_true", help="also print the screen as plain text (to find the keys)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- PROGRAM [ARGS]")
    a = ap.parse_args(argv)
    a.cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not a.cmd:
        ap.error("no program: put it after --")
    return a


def main(argv: list[str]) -> int:
    a = parse_args(argv)
    extra = dict(e.split("=", 1) for e in a.env)
    keys = parse_keys(a.keys, a.pause)
    with tempfile.TemporaryDirectory(prefix="shot-home-", dir=a.home_parent) as home:
        # the folder a screen may show as the current project: ~/projects/shop in the neutral home
        cwd = a.cwd or str(Path(home) / "projects" / "shop")
        Path(cwd).mkdir(parents=True, exist_ok=True)
        cmd = [arg.replace("{home}", home) for arg in a.cmd]
        if "/" in cmd[0]:
            cmd[0] = str(Path(cmd[0]).resolve())  # the program starts in another folder
        sess = Session(cmd, a.size, neutral_env(home, a.lang, extra), cwd)
        try:
            sess.drain(a.settle, limit=SETTLE_LIMIT)
            for data, wait in keys:
                if data:
                    sess.send(data)
                sess.pump(wait)
            sess.drain(a.settle, limit=SETTLE_LIMIT)
            rows = sess.cells()
        finally:
            sess.close()
        found = check_privacy(rows, forbidden_words(home))
    if a.show:
        print("\n".join("".join(c.data or " " for c in r).rstrip() for r in rows))
    if found:
        print(f"refused: the screen shows {found}", file=sys.stderr)
        return 2
    if a.squeeze >= 0:
        rows = squeeze(rows, a.squeeze)
    Path(a.out).write_text(render_svg(rows, a.title), encoding="utf-8")
    print(f"{a.out}: {Path(a.out).stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
