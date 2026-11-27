"""curve.py with the answer text stored (#104): wraps ab_server.parse_completion."""
import sys
from pathlib import Path

sys.path.insert(0, "/var/tmp/claude/fp8huge")
import curve  # noqa: E402

OUT = Path("/var/tmp/claude/fp8text")
curve.OUT = OUT
orig = curve.ab_server.parse_completion
label = sys.argv[1]
n = [0]


def keep(resp):
    n[0] += 1
    text = resp["choices"][0]["message"]["content"]
    (OUT / f"{label}-answer-{n[0]}.txt").write_text(text)
    return orig(resp)


curve.ab_server.parse_completion = keep
curve.main()
