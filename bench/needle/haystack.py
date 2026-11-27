"""The documents of the needle test: a pool of source files, shuffled per text, with coded needles at fixed depths.

Needle line: `# NOTE: the access code for vault <name> is <6 digits>.` Per document NEEDLES_PER_BIN needles in each of
the five depth bins; half of them are asked one by one ("single"), the other half in one answer ("multi"). The two sets
are disjoint, so the answers of the single questions cannot be copied into the multi answer.
"""

import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

HEADER = re.compile(r"^// ===== .* =====$", re.M)
# Depth bins as fractions of the document: start, 25 %, middle, 75 %, end (the last 1 % is left out: the needle
# should not be glued to the question).
BINS = [(0.01, 0.10), (0.22, 0.28), (0.47, 0.53), (0.72, 0.78), (0.90, 0.99)]
NEEDLES_PER_BIN = 8
NAMES = [
    "amber",
    "basalt",
    "cobalt",
    "dune",
    "ember",
    "fjord",
    "garnet",
    "harbor",
    "indigo",
    "jasper",
    "kestrel",
    "lagoon",
    "meadow",
    "nectar",
    "onyx",
    "prairie",
    "quartz",
    "raven",
    "summit",
    "tundra",
    "umber",
    "velvet",
    "willow",
    "xenon",
    "yarrow",
    "zephyr",
    "acorn",
    "birch",
    "cedar",
    "delta",
    "elm",
    "flint",
    "glacier",
    "heron",
    "iris",
    "juniper",
    "kelp",
    "lotus",
    "moss",
    "nova",
    "opal",
    "pine",
    "reed",
]


@dataclass
class Needle:
    name: str
    code: str
    frac: float  # position in the document, 0 = start
    bin: int
    mode: str  # "single" or "multi"
    hard: bool = False
    decoys: tuple = ()  # (position, line) of the stale revisions and the lookalike vault (hard only)

    def line(self) -> str:
        if self.hard:
            return f"# NOTE: vault {self.name} access code, revision 3: {self.code}."
        return f"# NOTE: the access code for vault {self.name} is {self.code}."


def split_pool(text: str) -> list[str]:
    """The pool as units: one per file, each starting with its `// ===== path =====` header line."""
    starts = [m.start() for m in HEADER.finditer(text)]
    return [text[a:b] for a, b in zip(starts, [*starts[1:], len(text)], strict=True)]


def make_needles(seed: int, hard: bool = False) -> list[Needle]:
    """Needles of one text, sorted by depth. Names, codes and the split single/multi follow from the seed only.

    hard: the MRCR-like variant. Each needle is revision 3 of its vault; revisions 1 and 2 (other codes) and a
    lookalike vault `<name>-b` (revision 3, another code) stand at random places of the document. The question asks for
    the current code, so the old revisions and the lookalike are decoys for the selection and for the reading."""
    rnd = random.Random(f"needles-{seed}")
    names = rnd.sample(NAMES, len(BINS) * NEEDLES_PER_BIN)
    codes = rnd.sample(range(100000, 1000000), len(names))
    out = []
    for b, (lo, hi) in enumerate(BINS):
        step = (hi - lo) / NEEDLES_PER_BIN
        modes = ["single", "multi"] * (NEEDLES_PER_BIN // 2)
        rnd.shuffle(modes)
        for k in range(NEEDLES_PER_BIN):
            frac = lo + step * (k + rnd.uniform(0.15, 0.85))
            i = len(out)
            out.append(Needle(names[i], str(codes[i]), frac, b, modes[k], hard))
    if hard:
        for n in out:
            n.decoys = decoys_of(n, rnd)
    return out


def decoys_of(n: Needle, rnd: random.Random) -> tuple:
    """Stale revisions 1 and 2 and the lookalike vault of a hard needle, each with its own 6-digit code."""
    code = lambda: str(rnd.randrange(100000, 1000000))  # noqa: E731
    return (
        (rnd.uniform(0.01, 0.99), f"# NOTE: vault {n.name} access code, revision 1: {code()}."),
        (rnd.uniform(0.01, 0.99), f"# NOTE: vault {n.name} access code, revision 2: {code()}."),
        (rnd.uniform(0.01, 0.99), f"# NOTE: vault {n.name}-b access code, revision 3: {code()}."),
    )


def build_document(units: list[str], seed: int, chars: int, needles: list[Needle]) -> str:
    """Files of the pool in the order of the seed up to `chars` characters, the needle lines inserted at their depth
    (at the next line break)."""
    order = units[:]
    random.Random(f"order-{seed}").shuffle(order)
    text, used = [], 0
    for u in order:
        if used >= chars:
            break
        text.append(u)
        used += len(u)
    lines = "".join(text)[:chars].split("\n")
    offs, pos = [], 0
    for ln in lines:
        offs.append(pos)
        pos += len(ln) + 1
    items = sorted([(n.frac, n.line()) for n in needles] + [d for n in needles for d in n.decoys])
    out, k = [], 0
    for i, ln in enumerate(lines):
        while k < len(items) and offs[i] >= items[k][0] * pos:
            out.append(items[k][1])
            k += 1
        out.append(ln)
    out.extend(line for _, line in items[k:])
    return "\n".join(out)


def single_question(n: Needle) -> str:
    if n.hard:
        return (
            f"What is the current access code of vault {n.name}, the one of its highest revision? "
            "Answer with the code only."
        )
    return f"What is the access code for vault {n.name}? Answer with the code only."


def multi_question(needles: list[Needle], seed: int) -> str:
    names = [n.name for n in needles]
    random.Random(f"multi-{seed}").shuffle(names)
    hard = needles[0].hard if needles else False
    what = "current access codes (highest revision)" if hard else "access codes"
    return (
        f"What are the {what} of these vaults: {', '.join(names)}? "
        "Answer with one line per vault in the form `name: code` and nothing else."
    )


def parse_multi(answer: str) -> dict[str, str]:
    """`name: 123456` lines of the answer (name in lower case, markup around it ignored)."""
    found = {}
    for m in re.finditer(r"([a-z]+)\W{0,4}\s*[:=\-]+\s*\W{0,3}(\d{6})(?!\d)", answer.lower()):
        found.setdefault(m.group(1), m.group(2))
    return found


def pool_from_roots(roots: list[str], exts: set[str], cap_bytes: int) -> str:
    """A pool file from source trees (skips build output and vendored code), for those who have no pool yet."""
    skip = {"node_modules", ".git", "work", "third_party", "build", "bin", "obj", "vendor", "dist", "__pycache__"}
    out, size = [], 0
    for root in roots:
        for dp, dn, fn in sorted(os.walk(root)):
            dn[:] = sorted(d for d in dn if d not in skip)
            for f in sorted(fn):
                if os.path.splitext(f)[1] not in exts or size >= cap_bytes:
                    continue
                try:
                    data = Path(dp, f).read_text(encoding="utf-8")
                except (UnicodeDecodeError, OSError):
                    continue
                if 400 < len(data) < 120_000 and max(map(len, data.split("\n"))) <= 600:
                    out.append(f"// ===== {os.path.relpath(os.path.join(dp, f), root)} =====\n{data}\n")
                    size += len(data)
    return "".join(out)
