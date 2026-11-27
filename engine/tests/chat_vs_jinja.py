"""Compare tona-chat (C++) with the GGUF chat template, run with jinja2 as in transformers.
Usage: chat_vs_jinja.py TONA_CHAT MODEL.gguf CASES.json (tona-chat takes the chat family from the model)"""

import json
import struct
import subprocess
import sys

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment


def chat_template(path):
    with open(path, "rb") as f:
        f.read(8)
        _, n_kv = struct.unpack("<QQ", f.read(16))
        size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

        def rs():
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8")

        def rv(t):
            if t == 8:
                return rs()
            if t == 9:
                at, n = struct.unpack("<IQ", f.read(12))
                if at in (8, 9):
                    return [rv(at) for _ in range(n)]
                f.seek(size[at] * n, 1)
                return None
            return f.read(size[t])

        for _ in range(n_kv):
            k = rs()
            (t,) = struct.unpack("<I", f.read(4))
            v = rv(t)
            if k == "tokenizer.chat_template":
                return v
    raise SystemExit("no chat template")


def main():
    tona_chat, model, cases_path = sys.argv[1:4]
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
    env.filters["tojson"] = lambda x, indent=None: json.dumps(x, ensure_ascii=False, indent=indent)

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env.globals["raise_exception"] = raise_exception
    tpl = env.from_string(chat_template(model))
    with open(cases_path) as f:
        cases = json.load(f)
    ours = json.loads(
        subprocess.run([tona_chat, "--model", model, cases_path], check=True, capture_output=True, text=True).stdout
    )
    fails = 0
    for i, (c, got) in enumerate(zip(cases, ours, strict=True)):
        kw = {
            "messages": c["messages"],
            "add_generation_prompt": True,
            "enable_thinking": c.get("thinking", True),
        }
        if c.get("tools"):
            kw["tools"] = c["tools"]
        if "effort" in c:
            kw["reasoning_effort"] = c["effort"]
            kw["medium_effort"] = c["effort"] in ("medium", "low")  # the Nemotron template's one reduced level
        want = tpl.render(**kw)
        if got != want:
            fails += 1
            k = next(
                j for j in range(min(len(got), len(want)) + 1) if j == min(len(got), len(want)) or got[j] != want[j]
            )
            lo, hi = max(0, k - 60), k + 60
            print(f"case {i}: differs from character {k}")
            print(f"  C++:   {got[lo:hi]!r}")
            print(f"  Jinja: {want[lo:hi]!r}")
    print(f"{len(cases) - fails} of {len(cases)} cases equal")
    sys.exit(1 if fails else 0)


main()
