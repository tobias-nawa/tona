"""IQ3_XXS tensors to TONA_IQ3_XXS_R (docs/use/tona-types.md), in place on a copy of a GGUF file.

The new type keeps the bytes of IQ3_XXS and only moves them: two 98-byte blocks become one 196-byte super-block with
the index bytes first, then the sign and scale words, then the two block scales, so a GPU lane reads its fields with
aligned loads (research/measurements/2026-11-12-mac-expert-decode-probes.md). Tensor sizes and offsets stay the same;
only the data of the chosen tensors and their type fields change. The copy is an APFS clone on macOS (`cp -c`), so
the file costs disk space only for the changed blocks.
"""

import re
import shutil
import subprocess
import sys

from coder_requant.gguf_layout import Layout, Tensor, parse

IQ3_XXS, IQ3_XXS_R = 18, 1018
BLOCK, SUPER = 98, 196
HEADER_READ = 64 << 20  # enough for the Coder's header (~10 MB)
EXPERTS = r"\.ffn_(gate|up)_exps\.weight$"


def repack_blocks(data: bytes) -> bytes:
    """Pairs of IQ3_XXS blocks (d, 64 index bytes, 32 sign and scale bytes) as super-blocks."""
    if len(data) % SUPER:
        raise ValueError(f"{len(data)} bytes are no whole number of block pairs")
    out = bytearray(len(data))
    for s in range(0, len(data), SUPER):
        a, b = data[s : s + BLOCK], data[s + BLOCK : s + SUPER]
        out[s : s + SUPER] = a[2:66] + b[2:66] + a[66:98] + b[66:98] + a[0:2] + b[0:2]
    return bytes(out)


def chosen(layout: Layout, pattern: str = EXPERTS) -> list[Tensor]:
    """IQ3_XXS tensors whose name matches and whose rows hold whole super-blocks."""
    out = []
    for t in layout.tensors:
        if t.type != IQ3_XXS or not re.search(pattern, t.name):
            continue
        if t.dims[0] % 512:
            raise ValueError(f"{t.name}: row of {t.dims[0]} weights is no whole number of 512-weight super-blocks")
        out.append(t)
    return out


def copy(src: str, dst: str) -> None:
    if sys.platform == "darwin":
        subprocess.run(["cp", "-c", src, dst], check=True)
    else:
        shutil.copyfile(src, dst)


def repack_file(path: str, pattern: str = EXPERTS, chunk: int = 98 << 20) -> list[str]:
    """Repacks the chosen tensors of path in place; returns their names. Data first, then the type fields."""
    with open(path, "r+b") as f:
        layout = parse(f.read(HEADER_READ))
        tensors = chosen(layout, pattern)
        for t in tensors:
            for at in range(0, t.nbytes, chunk):
                pos = layout.data_start + t.offset + at
                f.seek(pos)
                data = f.read(min(chunk, t.nbytes - at))
                f.seek(pos)
                f.write(repack_blocks(data))
        for t in tensors:
            f.seek(t.type_pos)
            f.write(IQ3_XXS_R.to_bytes(4, "little"))
    return [t.name for t in tensors]
