#!/usr/bin/env python3
"""Writes the prompts of scripts/bench-set.py as the model sees them (chat markers, no thinking), one file per prompt,
for tona-layer-skip. The lists are read from the script's source, so the set stays the one the drafter is
tuned on.

Usage: tools/self_spec_prompts.py OUT_DIR   ->  OUT_DIR/code00.txt ... code10.txt, text00.txt ... text02.txt
"""

import ast
import sys
from pathlib import Path

BENCH_SET = Path(__file__).resolve().parent.parent / "scripts" / "bench-set.py"


def bench_prompts() -> dict[str, list[str]]:
    tree = ast.parse(BENCH_SET.read_text())
    out = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ("CODE", "TEXT")
        ):
            out[node.targets[0].id.lower()] = list(ast.literal_eval(node.value))
    return out


def render(prompt: str) -> str:
    """The Qwen chat format of the server for one user turn with thinking off (chat_qwen3_render.cpp)."""
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for kind, prompts in bench_prompts().items():
        for i, p in enumerate(prompts):
            (out / f"{kind}{i:02d}.txt").write_text(render(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
