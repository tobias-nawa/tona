#!/usr/bin/env python3
"""Compares the tokenizer against llama-tokenize (llama.cpp b11822): corpus, special cases, random texts.

Usage: engine/tests/tokenizer_vs_llamacpp.py TONA_TOKENIZE LLAMA_TOKENIZE MODEL.gguf CORPUS [--fuzz N]
Compares the ID sequences exactly and checks decode(encode(x)) == x for valid UTF-8.
"""

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

CASES = [
    "",
    " ",
    "  ",
    "\n",
    "\r\n",
    "\n\n\n",
    " \n ",
    "\t\t",
    "a",
    "Hello world",
    " Hello",
    "  Hello  ",
    "it's I'M we'Ve they'LL she'd 'S 'tis ''",
    "don't\nDON'T",
    "x'",
    "'",
    "1234567890",
    "3.14159",
    "1,000,000",
    "v2.0-rc1",
    "0x1F",
    "１２３",  # full-width digits
    "def f(x):\n    return x**2\n\n\n# comment\n",
    "if a<=b and c!=d: pass",
    "a  =  b",
    "foo()\n\t\tbar()",
    "    \n    \n",
    "x\r\ny\rz",
    "  \t\n  x",
    "a \u00a0b\u3000c\u2028d",
    "trailing   ",
    "\n   leading",
    "Ärger über Öl, ß und Ä̈",
    "e\u0301 a\u0300\u0301",
    "\u0301start",
    "日本語のテキスト、中文文本。",
    "한국어 텍스트",
    "مرحبا بالعالم",
    "שלום עולם",
    "Привет, мир!",
    "ελληνικά",
    "हिन्दी पाठ",
    "ภาษาไทย",
    "🚀🔥👍🏽 👨‍👩‍👧‍👦 🇩🇪",
    "math: ∀x∈ℝ, x²≥0 → √x",
    "«quotes» „german“ “english”",
    "—–‐",
    "…!!!???",
    "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
    "<think>\nthink\n</think>\n\nanswer",
    '<tool_call>\n{"name": "f"}\n</tool_call>',
    "<|im_start|><|im_end|>",
    "<|endoftext|>x<|endoftext|>",
    "a<|im_start|>b",
    "<|im_start",
    "<|fake|>",
    "<<|im_end|>>",
    "\x00\x01\x7f",
    "tab\tsep\tvalues",
    "C:\\Users\\x\\file.txt",
    "https://example.com/a?b=c&d=e#f",
    '{"key": [1, 2, {"n": null}]}',
    "SELECT * FROM t WHERE x = 'y';",
    "$ ls -la | grep foo > out 2>&1",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "                                        x",
    "\ufeffBOM",
    "\ufffd",
    "\U0001f600" * 20,
    "zero\u200bwidth\u200cjoin\u200d",
]

# Building blocks for random texts: every category the regex distinguishes
ATOMS = [
    *list("abcXYZ019 '\t\n\r.,;:!?()[]{}<>=+-*/\\\"#$%&@^_`|~"),
    "\xa0",
    "\u3000",
    "\u2028",
    "́",
    "̈",
    "ä",
    "ß",
    "日",
    "本",
    "한",
    "م",
    "א",
    "П",
    "λ",
    "ह",
    "्",
    "ภ",
    "🚀",
    "👍🏽",
    "\u200d",
    "\u200b",
    "\ufeff",
    "²",
    "½",
    "Ⅷ",
    "١",
    "∀",
    "→",
    "€",
    "©",
    "\x00",
    "\x7f",
    "\x85",
    "'s",
    "'LL",
    "'re",
    "  ",
    "\n\n",
    " \n",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "<|endoftext|>",
    "<|im_",
    "|>",
    "def ",
    "return",
    "    ",
    "self.",
    "0x",
    "1e-6",
]


def ids_tona(tool, model, path, special=True):
    args = [tool, model, str(path)] + ([] if special else ["--no-special"])
    out = subprocess.run(args, capture_output=True, check=True).stdout.decode()
    return [int(x) for x in out.split()]


def ids_llama(tool, model, path, special=True):
    # --no-escape: otherwise llama-tokenize turns "\\n" in the text into a newline
    args = [tool, "-m", model, "-f", str(path), "--ids", "--log-disable", "--no-escape"]
    if not special:
        args.append("--no-parse-special")
    out = subprocess.run(args, capture_output=True, check=True).stdout.decode()
    line = [row for row in out.splitlines() if row.startswith("[")]
    return json.loads(line[-1]) if line else []


def decode_tona(tool, model, path):
    return subprocess.run([tool, model, str(path), "--no-special", "--decode"], capture_output=True, check=True).stdout


def build_texts(rng, fuzz, corpus):
    texts = [c.encode() for c in CASES]
    for _ in range(fuzz):
        texts.append("".join(rng.choice(ATOMS) for _ in range(rng.randint(1, 60))).encode())
    for _ in range(20):  # arbitrary bytes, including invalid UTF-8
        texts.append(bytes(rng.randrange(256) for _ in range(rng.randint(1, 40))))
    for _ in range(10):  # excerpts of the corpus with random boundaries
        i = rng.randrange(len(corpus) - 5000)
        texts.append(corpus[i : i + rng.randint(1, 5000)])
    return texts


def check_ids(a, k, t, p):
    """Compares the ID sequences of both tokenizers in both modes; returns the number of mismatches."""
    fails = 0
    for special in (True, False):
        a_ids, b_ids = ids_tona(a.tona, a.model, p, special), ids_llama(a.llama, a.model, p, special)
        if a_ids != b_ids:
            fails += 1
            i = next(
                (j for j, (x, y) in enumerate(zip(a_ids, b_ids, strict=False)) if x != y), min(len(a_ids), len(b_ids))
            )
            print(
                f"MISMATCH case {k} special={special}: {t[:80]!r}\n  from index {i}: ours {a_ids[i : i + 8]}, "
                f"llama.cpp {b_ids[i : i + 8]}"
            )
    return fails


def check_roundtrip(a, k, t, p):
    try:
        t.decode("utf-8")
    except UnicodeDecodeError:
        return 0
    if decode_tona(a.tona, a.model, p) != t:
        print(f"ROUNDTRIP broken case {k}: {t[:80]!r}")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tona"), ap.add_argument("llama"), ap.add_argument("model"), ap.add_argument("corpus")
    ap.add_argument("--fuzz", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    corpus = Path(a.corpus).read_bytes()
    texts = build_texts(random.Random(a.seed), a.fuzz, corpus)

    fails = 0
    with tempfile.TemporaryDirectory(prefix="tona-tok-") as d:
        p = Path(d) / "t.txt"
        for k, t in enumerate(texts):
            p.write_bytes(t)
            fails += check_ids(a, k, t, p) + check_roundtrip(a, k, t, p)
        a_ids = ids_tona(a.tona, a.model, a.corpus)
        b_ids = ids_llama(a.llama, a.model, a.corpus)
        same = a_ids == b_ids
        fails += not same
        print(f"Corpus: {len(corpus)} bytes, {len(a_ids)} tokens, {'identical' if same else 'MISMATCH'}")
    print(f"{len(texts)} texts x 2 modes, {fails} mismatches")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
