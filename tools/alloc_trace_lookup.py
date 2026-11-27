#!/usr/bin/env python3
"""Assign a fault address to the buffer of an TONA_ALLOC_TRACE file.

Usage: alloc_trace_lookup.py TRACE ADDRESS [SLACK_MB]

Prints every allocation and borrowed range that contains ADDRESS (live or freed, with the time of its free) and the
nearest allocation before and after it, so an out-of-bounds read shows as "past the end of X by N bytes".
"""

import sys
from dataclasses import dataclass


@dataclass
class Block:
    t_alloc: float
    kind: str
    start: int
    size: int
    where: str
    t_free: float | None = None

    @property
    def end(self) -> int:
        return self.start + self.size


def parse(path: str) -> list[Block]:
    blocks: list[Block] = []
    live: dict[int, Block] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split(None, 4)
            if len(parts) < 5:
                continue
            t, kind, addr, size, where = float(parts[0]), parts[1], int(parts[2], 16), int(parts[3]), parts[4].strip()
            if kind in ("hipMalloc", "hipHostMalloc", "range"):
                b = Block(t, kind, addr, size, where)
                blocks.append(b)
                if kind != "range":
                    live[addr] = b
            elif kind in ("hipFree", "hipHostFree") and addr in live:
                live.pop(addr).t_free = t
    return blocks


def describe(b: Block) -> str:
    free = "live" if b.t_free is None else f"freed at {b.t_free:.0f} ms"
    return f"{b.kind} {b.start:#x}..{b.end:#x} ({b.size / 1e6:.1f} MB) at {b.t_alloc:.0f} ms, {free}: {b.where}"


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    blocks = parse(sys.argv[1])
    addr = int(sys.argv[2], 16)
    inside = [b for b in blocks if b.start <= addr < b.end]
    for b in inside:
        print(f"inside   {describe(b)}  (offset {addr - b.start:#x})")
    if not inside:
        print("inside   no allocation or range contains the address")
        below = [b for b in blocks if b.kind != "range" and b.end <= addr]
        above = [b for b in blocks if b.kind != "range" and b.start > addr]
        if below:
            b = max(below, key=lambda x: x.end)
            print(f"before   {describe(b)}  (address is {addr - b.end} bytes past its end)")
        if above:
            b = min(above, key=lambda x: x.start)
            print(f"after    {describe(b)}  (address is {b.start - addr} bytes before its start)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
