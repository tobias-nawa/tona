"""Carry an expert profile from one Coder GGUF to a cut of it (DM, DM-T) by the router rows.

A cut keeps a subset of the Coder's experts with their router rows unchanged, so a cut expert's row equals one row of
the source byte for byte. The cut's profile takes each matched expert's count; an expert with no partner (DM was cut
from an older Coder shard whose experts differ in part) or with several gets 0, and the share of those is reported.
Profile format: engine/backend/expert_profile.h, version 1.

Usage: python3 tools/coder_requant/profile_map.py SOURCE.gguf SOURCE_PROFILE.bin CUT.gguf OUT.bin
"""

import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from coder_requant import sources

PROFILE_MAGIC = 0x50584549  # "IEXP", version 1


def match_rows(cut: bytes, source: bytes, row_bytes: int) -> list[int]:
    """For each row of `cut` the index of the one equal row of `source`; -1 for none or several."""
    where: dict[bytes, list[int]] = {}
    for i in range(len(source) // row_bytes):
        where.setdefault(source[i * row_bytes : (i + 1) * row_bytes], []).append(i)
    out = []
    for j in range(len(cut) // row_bytes):
        hits = where.get(cut[j * row_bytes : (j + 1) * row_bytes], [])
        out.append(hits[0] if len(hits) == 1 else -1)
    return out


def read_profile(path: Path) -> tuple[int, int, list[int]]:
    data = path.read_bytes()
    magic, layers, experts = struct.unpack_from("<3I", data)
    if magic != PROFILE_MAGIC:
        raise ValueError(f"{path}: not a version 1 profile")
    return layers, experts, list(struct.unpack_from(f"<{layers * experts}I", data, 12))


def write_profile(path: Path, layers: int, experts: int, counts: list[int]) -> None:
    path.write_bytes(struct.pack("<3I", PROFILE_MAGIC, layers, experts) + struct.pack(f"<{len(counts)}I", *counts))


def carry(source_gguf: Path, profile: Path, cut_gguf: Path) -> tuple[int, list[int], float, float]:
    """(experts of the cut, its counts, share of cut experts without partner, share of the uses they carry)."""
    layers, n_src, counts = read_profile(profile)
    sr, cr = sources.Reader(source_gguf, None), sources.Reader(cut_gguf, None)
    sl, cl = sr.layout(), cr.layout()
    out, missing, total, carried = [], 0, 0, 0
    n_cut = 0
    for layer in range(layers):
        name = f"blk.{layer}.ffn_gate_inp.weight"
        ts, tc = sl.by_name[name], cl.by_name[name]
        if ts.type != tc.type or ts.dims[0] != tc.dims[0] or ts.dims[1] != n_src:
            raise ValueError(f"{name}: router types or shapes differ ({ts.type} {ts.dims}, {tc.type} {tc.dims})")
        n_cut = tc.dims[1]
        idx = match_rows(cr.tensor(cl, name), sr.tensor(sl, name), tc.nbytes // n_cut)
        row = counts[layer * n_src : (layer + 1) * n_src]
        out += [row[i] if i >= 0 else 0 for i in idx]
        missing += idx.count(-1)
        total += sum(row)
        carried += sum(row[i] for i in idx if i >= 0)
    return n_cut, out, missing / (layers * n_cut), carried / max(total, 1)


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    n_cut, counts, missing, carried = carry(Path(argv[0]), Path(argv[1]), Path(argv[2]))
    layers = len(counts) // n_cut
    write_profile(Path(argv[3]), layers, n_cut, counts)
    print(
        f"{layers} layers x {n_cut} experts; without partner {100 * missing:.1f} % of the cut's experts; "
        f"the matched ones carry {100 * carried:.1f} % of the source profile's uses"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
