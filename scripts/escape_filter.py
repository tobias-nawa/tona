"""Stateful filter stripping terminal escape sequences pyte does not understand.

FTXUI queries the terminal with DCS strings such as DECRQSS (ESC P $ q SP q ESC \\) and private CSI
queries (DECRQM, DECSCUSR, Kitty keyboard / XTVERSION). pyte prints unknown DCS/CSI sequences as literal
text unless stripped across arbitrary read chunk boundaries.
"""

from __future__ import annotations

import re

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
