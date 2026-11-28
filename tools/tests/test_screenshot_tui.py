"""scripts/screenshot-tui.py and scripts/status-stub.py: what needs no terminal (colours, keys, the SVG), plus
one run in a pseudo-terminal when pyte is installed."""

import json
import re
import subprocess
import sys
import threading
import urllib.request
from http.server import HTTPServer
from xml.dom import minidom

import pytest
from conftest import load_script

S = load_script("scripts/screenshot-tui.py")
status_stub = load_script("scripts/status-stub.py")


def row(text, **style):
    return [S.Cell(c, **style) for c in text]


def test_resolve_color_names_hex_and_default():
    assert S.resolve_color("default") is None
    assert S.resolve_color("red") == S.PALETTE["red"]
    assert S.resolve_color("brown") == S.PALETTE["brown"]  # pyte calls yellow brown
    assert S.resolve_color("bfightmagenta") == S.PALETTE["brightmagenta"]  # pyte's typo for background 105
    assert S.resolve_color("123456") == "#123456"
    # xterm's 0-15 of the 256 palette are the theme's colours, so 38;5;2 and 32 look the same
    assert S.resolve_color("00cd00") == S.PALETTE["green"]
    assert S.resolve_color("ff0000") == S.PALETTE["brightred"]
    with pytest.raises(ValueError):
        S.resolve_color("chartreuse")


def test_parse_keys():
    keys = S.parse_keys("enter, down  wait:1.5 text:ab x ctrl-c", 0.4)
    assert keys == [
        ("\r", 0.4),
        ("\x1b[B", 0.4),
        ("", 1.5),
        ("a", 0.02),
        ("b", 0.02),
        ("", 0.4),
        ("x", 0.4),
        ("\x03", 0.4),
    ]
    assert S.parse_keys("", 0.4) == []
    with pytest.raises(ValueError):
        S.parse_keys("enterr", 0.4)


def test_runs_merge_equal_styles_and_keep_wide_tails():
    cells = row("ab") + row("cd", fg="#ff0000") + [S.Cell("中"), S.Cell("")] + row("e")
    got = S.runs(cells)
    assert [(x, n, t) for x, n, t, _ in got] == [
        (0, 2, "ab"),
        (2, 2, "cd"),
        (4, 3, "中e"),
    ]  # the tail cell widens its head run


def test_squeeze_cuts_blank_runs_but_not_rows_with_a_background():
    text, blank, shaded = row("x"), row(" "), row(" ", bg="#222222")
    rows = [text, blank, blank, blank, shaded, blank, blank, text]
    assert S.squeeze(rows, 1) == [text, blank, shaded, blank, text]
    assert S.squeeze(rows, 0) == [text, shaded, text]


def test_mix_goes_part_of_the_way():
    assert S.mix("#000000", "#ffffff", 0.5) == "#808080"
    assert S.mix("#102030", "#102030", 0.9) == "#102030"


def test_render_svg_is_valid_deterministic_and_escaped():
    rows = [row("a<b & c", fg="#00ff00", bold=True), row(" ", bg="#112233"), row("done", faint=True, underline=True)]
    svg = S.render_svg(rows, "t & <t>")
    assert svg == S.render_svg(rows, "t & <t>")
    doc = minidom.parseString(svg)  # well-formed XML
    assert doc.documentElement.tagName == "svg"
    assert "a&lt;b&#160;&amp;&#160;c" in svg and 'font-weight="bold"' in svg
    assert 'fill="#112233"' in svg  # the shaded cell is a rectangle
    assert 'text-decoration="underline"' in svg
    assert not re.search(r"\d{4}-\d\d-\d\d|id=", svg)  # no timestamps, no random ids
    # blank cells without a background draw nothing
    assert svg.count("<text ") == 3  # the title and the two rows with text


def test_text_has_no_raw_spaces_and_no_raw_nbsp():
    # Browsers that load the SVG as an <img> collapse spaces in <text>: every one is a character reference
    svg = S.render_svg([row("Set up   Tona", fg="#ffffff")], "a title")
    contents = re.findall(r"<text [^>]*>([^<]*)</text>", svg)
    assert contents and all(" " not in c for c in contents)
    assert "Set&#160;up&#160;&#160;&#160;Tona" in svg
    assert "\u00a0" not in svg  # no invisible character in the file


def test_privacy_check_finds_names_and_homes():
    rows = [row("project in /home/zed/shop"), row("host box7")]
    assert S.check_privacy(rows, ["/home/zed", "box7", "nobody"]) == ["/home/zed", "box7"]
    assert S.check_privacy(rows, ["zeD"]) == []


def test_neutral_env_has_no_machine_in_it(monkeypatch):
    monkeypatch.setenv("SECRET_TOKEN", "x")
    env = S.neutral_env("/var/tmp/h", "de_DE.UTF-8", {"A": "b"})
    assert env["HOME"] == "/var/tmp/h" and env["TZ"] == "UTC" and env["LANG"] == "de_DE.UTF-8" and env["A"] == "b"
    assert "SECRET_TOKEN" not in env


def test_parse_args_needs_a_program_and_a_size():
    a = S.parse_args(["--out", "x.svg", "--size", "80x24", "--", "prog", "--flag"])
    assert a.size == (80, 24) and a.cmd == ["prog", "--flag"]
    with pytest.raises(SystemExit):
        S.parse_args(["--out", "x.svg"])
    with pytest.raises(SystemExit):
        S.parse_args(["--out", "x.svg", "--size", "big", "--", "p"])


def test_stub_advances_the_first_request_by_a_fixed_step():
    doc = {"engine": {"active": [{"out_tokens": 480}, {"out_tokens": 0}]}}
    assert status_stub.advance(doc, 0, 96)["engine"]["active"][0]["out_tokens"] == 480
    assert status_stub.advance(doc, 3, 96)["engine"]["active"][0]["out_tokens"] == 480 + 72
    assert doc["engine"]["active"][0]["out_tokens"] == 480  # the recorded document stays as it is


def test_stub_serves_status_and_refuses_the_real_port():
    doc = {"engine": {"active": [{"out_tokens": 1}]}}
    srv = HTTPServer(("127.0.0.1", 0), status_stub.make_handler(doc, 96))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        url = f"http://127.0.0.1:{srv.server_port}/status"
        first = json.load(urllib.request.urlopen(url))
        second = json.load(urllib.request.urlopen(url))
        assert second["engine"]["active"][0]["out_tokens"] - first["engine"]["active"][0]["out_tokens"] == 24
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f"http://127.0.0.1:{srv.server_port}/other")
    finally:
        srv.shutdown()
    assert status_stub.main(["--port", "8662"]) == 2


def test_capture_in_a_pseudo_terminal(tmp_path):
    pytest.importorskip("pyte")
    out = tmp_path / "p.svg"
    # green bold text, a dim line, a key read back: the whole path from pty to SVG
    prog = r"printf '\033[1;32mHello\033[0m \033[2mfaint\033[0m\n'; read -r k; echo \"got:$k\""
    r = subprocess.run(
        [sys.executable, S.__file__, "--out", str(out), "--size", "40x6", "--keys", "x enter", "--settle", "0.3",
         "--home-parent", str(tmp_path), "--", "bash", "-c", prog],
        capture_output=True, text=True, timeout=60,
    check=False,
    )  # fmt: skip
    assert r.returncode == 0, r.stderr
    svg = out.read_text()
    minidom.parseString(svg)
    assert ">Hello<" in svg and S.PALETTE["green"] in svg and 'font-weight="bold"' in svg
    assert "faint" in svg and "got:x" in svg
    assert str(tmp_path) not in svg


def test_escape_filter_strips_dcs_and_unknown_csi_across_chunks():
    # FTXUI sends DECRQSS (ESC P $ q SP q ESC \) and DECRQM / DECSCUSR on start; none may leak as literal text
    payload = b"\x1bP$q q\x1b\\\x1b[?2026$p\x1b[2 q\x1b[1;32mSet up Tona\x1b[0m\x1b[>0q\x1b_apc\x1b\\!"
    f = S.EscapeFilter()
    assert f.feed(payload) == b"\x1b[1;32mSet up Tona\x1b[0m!"
    # Feeding byte-by-byte across every possible split boundary yields the exact same output
    f2 = S.EscapeFilter()
    rebuilt = b"".join(f2.feed(bytes([b])) for b in payload)
    assert rebuilt == b"\x1b[1;32mSet up Tona\x1b[0m!"


def test_capture_is_refused_when_the_screen_shows_the_home(tmp_path):
    pytest.importorskip("pyte")
    out = tmp_path / "p.svg"
    r = subprocess.run(
        [sys.executable, S.__file__, "--out", str(out), "--size", "60x4", "--settle", "0.3",
         "--home-parent", str(tmp_path), "--", "bash", "-c", "echo $HOME"],
        capture_output=True, text=True, timeout=60,
    check=False,
    )  # fmt: skip
    assert r.returncode == 2 and "refused" in r.stderr and not out.exists()
