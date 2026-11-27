"""parse_reply of bench/polyglot: FILE blocks, and the fallback for a reply without a FILE line."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench/polyglot"))
import tasks

FENCE = "```"


def test_file_block_is_parsed_as_before():
    text = f"FILE: src/lib.rs\n{FENCE}rust\nfn a() {{}}\n{FENCE}"
    assert tasks.parse_reply(text, ["src/lib.rs"]) == {"src/lib.rs": "fn a() {}\n"}


def test_nameless_fence_is_taken_for_a_single_file_exercise():
    text = f"Use this:\n{FENCE}rust\nfn a() {{}}\n{FENCE}\nDone."
    assert tasks.parse_reply(text, ["src/lib.rs"]) == {"src/lib.rs": "fn a() {}\n"}


def test_longest_fence_wins_over_a_usage_snippet():
    full = "fn a() {}\nfn b() {}\nfn c() {}"
    text = f"{FENCE}rust\n{full}\n{FENCE}\nUsage:\n{FENCE}\na();\n{FENCE}"
    assert tasks.parse_reply(text, ["src/lib.rs"]) == {"src/lib.rs": full + "\n"}


def test_no_fallback_for_several_files_or_without_a_fence():
    text = f"{FENCE}rust\nfn a() {{}}\n{FENCE}"
    assert tasks.parse_reply(text, ["a.rs", "b.rs"]) == {}
    assert tasks.parse_reply("no code here", ["src/lib.rs"]) == {}


def test_file_block_beats_the_fallback():
    text = f"{FENCE}\nx\n{FENCE}\nFILE: lib.rs\n{FENCE}\nfn real() {{}}\n{FENCE}"
    assert tasks.parse_reply(text, ["lib.rs"]) == {"lib.rs": "fn real() {}\n"}
