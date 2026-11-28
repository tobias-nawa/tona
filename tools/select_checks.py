#!/usr/bin/env python3
"""Names the checks a change needs, from the build graph, and its change class of docs/contribute/measurement.md.

Usage: tools/select_checks.py [--base REF] [--head REF] [--build DIR] [--json]

Compares HEAD with the merge base of REF (default origin/main, else main). For every changed file it asks the build
in DIR (default work/build/engine; ninja knows the inputs of every binary, headers included) which binaries the file
goes into, and from the binaries it derives the golden checks and the unit tests. Files the graph does not see
(toolchain, golden scripts and prompts, generators) trigger everything; a change of engine/CMakeLists.txt is
compared through select_cmake.py (build commands of the golden binaries before and after); so does every case the tool
cannot decide, a missing build, and an error. The tool never silently checks less.
It does not replace the run on the real model: it names the long-context run and the pressure series where a change
reaches them. Exit code 2: git failed (callers then check everything)."""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import select_cmake
import select_graph
import select_rules as rules
from select_graph import Graph, GraphError
from select_rules import Verdict

SOURCE_SUFFIXES = (".h", ".hpp", ".inc", ".cpp", ".hip")
RANK = {"C0": 0, "C1": 1, "C2": 2, "C3": 3, "C5": 5, "C7": 7, "C8": 8}
# Run times of the golden checks with one model load per check (research/measurements/2026-11-02-golden-loads.md)
GOLDEN_SECONDS = {"gpu": 29, "cpu plan": 28, "prefill": 157, "prefill dflash": 27, "server": 105}


def git(*args: str) -> str:
    done = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    if done.returncode:
        raise SystemExit(f"select_checks: git {' '.join(args)}: {done.stderr.strip()}")
    return done.stdout


def default_base() -> str:
    ok = (
        subprocess.run(
            ["git", "rev-parse", "--verify", "-q", "origin/main"], capture_output=True, check=False
        ).returncode
        == 0
    )
    return "origin/main" if ok else "main"


def changed_files(base: str, head: str = "HEAD") -> list[tuple[str, str]]:
    """(status, path) per file against the merge base; a rename is a deletion and an addition."""
    point = git("merge-base", base, head).strip()
    lines = git("diff", "--name-status", "--no-renames", point, head).splitlines()
    return [(line.split("\t")[0][0], line.split("\t", 1)[1]) for line in lines if line]


def binaries_text(reached: list[str]) -> str:
    """The binaries that decide a golden check by name, the rest as a count (a header is in dozens of binaries)."""
    named = [b for b in reached if b in rules.GOLDEN_BINARIES]
    others = len(reached) - len(named)
    return ", ".join(named + ([f"{others} more" if named else f"{others} test or tool binaries"] if others else []))


def is_cmake(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    # Only what select_cmake unpacks (engine/) can be compared; any other CMake file triggers everything
    return (name == "CMakeLists.txt" or name.endswith(".cmake")) and path.startswith("engine/")


def cmake_verdict(path: str, differing: list[str] | None) -> Verdict:
    """differing: the golden binaries whose build commands changed; None if that could not be found out."""
    if differing is None:
        return rules.everything(path, "build definition, and the old and the new configuration could not be compared")
    goldens = [g for binary in differing for g in rules.GOLDEN_BINARIES[binary]]
    why = (
        "build commands of " + ", ".join(differing) if differing else "build commands of the golden binaries unchanged"
    )
    cls = "C5" if "tona-gpu-gen" in differing else ("C3" if differing else "C1")
    # unit_all: a changed CMake file may add or change any test
    return Verdict(path, why, cls=cls, goldens=goldens, unit_all=True, compute="tona-gpu-gen" in differing)


def engine_verdict(path: str, status: str, graph: Graph) -> Verdict:
    reached = graph.reaching(path)
    if not reached:
        if path.startswith("engine/tests/") and not path.endswith(SOURCE_SUFFIXES):
            return Verdict(path, "test data: all unit tests", cls="C1", unit_all=True)
        if status == "A" and path.endswith(SOURCE_SUFFIXES):
            return Verdict(path, "new file the build does not know yet: its includers are in this change", cls="C1")
        return rules.everything(path, "in no binary of the build graph")
    device = path in graph.device or path.endswith(".hip")
    verdict = Verdict(path, "in " + binaries_text(reached), cls=rules.engine_class(path, device), via=reached)
    verdict.compute = device
    verdict.installer = path.startswith("engine/third_party/")
    for binary in reached:
        verdict.goldens += [g for g in rules.GOLDEN_BINARIES.get(binary, []) if g not in verdict.goldens]
    verdict.unit = sorted(n for n, (exe, unit) in graph.tests.items() if unit and exe in reached)
    return verdict if path.startswith("engine/tests/") else rules.extras(verdict, path)


@dataclass
class Plan:
    verdicts: list[Verdict] = field(default_factory=list)
    error: str = ""

    def any(self, attr: str) -> bool:
        return any(getattr(v, attr) for v in self.verdicts)

    def classes(self) -> list[str]:
        found = {v.cls for v in self.verdicts}
        found |= {"C7" for v in self.verdicts if v.long_context}
        found |= {"C8" for v in self.verdicts if v.pressure}
        return sorted(found or {"C0"}, key=RANK.get)

    def goldens(self) -> list[str]:
        wanted = {g for v in self.verdicts for g in v.goldens}
        return [g for g in rules.ALL_GOLDENS if g in wanted]

    def recipes(self) -> list[str]:
        found = set()
        for v in self.verdicts:
            found |= set(rules.RECIPES[v.cls])
            found |= {"R7"} if v.long_context else set()
            found |= {"R8", *rules.RECIPES["C3"]} if v.pressure else set()
            found |= {"R4"} if v.stash and v.cls == "C3" else set()
        if not self.goldens():
            found.discard("R3")
        return sorted(found)

    def unit_tests(self) -> list[str]:
        return sorted({n for v in self.verdicts for n in v.unit})

    def reasons(self, attr: str) -> list[str]:
        return [v.path for v in self.verdicts if getattr(v, attr)]

    def to_dict(self) -> dict:
        goldens = self.goldens()
        skipped = [g for g in rules.ALL_GOLDENS if g not in goldens]
        classes = self.classes()
        needs = ["long-context run"] * self.any("long_context") + ["pressure series"] * self.any("pressure")
        return {
            "class": classes[-1],
            "classes": classes,
            "everything": self.any("everything"),
            "error": self.error,
            "goldens": goldens,
            "skipped": skipped,
            "skipped_count": len(skipped),
            "golden_seconds": sum(GOLDEN_SECONDS[g] for g in goldens),
            "skipped_seconds": sum(GOLDEN_SECONDS[g] for g in skipped),
            "unit_all": self.any("unit_all") or self.any("everything"),
            "unit_tests": self.unit_tests(),
            "python_tests": self.any("python") or self.any("everything"),
            "installer_tests": self.any("installer") or self.any("everything"),
            "bit_exact": self.any("compute") or self.any("everything"),
            "speed": "R5" in self.recipes(),
            "kernel_identity": any(v.cls == "C5" and v.compute and not v.everything for v in self.verdicts),
            "recipes": self.recipes(),
            "needs": needs,
            "long_context_files": self.reasons("long_context"),
            "pressure_files": self.reasons("pressure"),
            "files": [{"path": v.path, "class": v.cls, "why": v.why, "goldens": v.goldens} for v in self.verdicts],
        }


def plan_for(changes: list[tuple[str, str]], build: Path, refs: tuple[str, str] | None = None) -> Plan:
    """refs: (merge base, head) as commits; they let a CMake change be compared instead of triggering everything."""
    plan = Plan()
    pending = []
    differing: list[str] | None = None
    for status, path in changes:
        verdict = None
        if is_cmake(path):
            differing = differing if differing is not None else compare_cmake(refs)
            verdict = cmake_verdict(path, differing)
        else:
            verdict = rules.outside_engine(path)
        if verdict:
            plan.verdicts.append(verdict)
        else:
            pending.append((status, path))
    if not pending:
        return plan
    try:
        graph = select_graph.load(build)
    except GraphError as error:
        plan.error = str(error)
        plan.verdicts += [rules.everything(p, f"no build graph ({error})") for _, p in pending]
        return plan
    plan.verdicts += [engine_verdict(path, status, graph) for status, path in pending]
    return plan


def compare_cmake(refs: tuple[str, str] | None) -> list[str] | None:
    if refs is None:
        return None
    try:
        return select_cmake.changed_binaries(Path(git("rev-parse", "--show-toplevel").strip()), *refs)
    except (GraphError, OSError):
        return None


def render(data: dict, base: str) -> str:
    out = [f"Change class {data['class']} against {base}: {rules.CLASS_NAMES[data['class']]}"]
    if data["classes"][:-1]:
        out[0] += f" (also {', '.join(data['classes'][:-1])})"
    if data["error"]:
        out.append(f"  graph error, so everything runs: {data['error']}")
    out.append("Changed files:")
    for f in data["files"]:
        out.append(
            f"  {f['path']}: {f['class']}, {f['why']}"
            + (f" -> golden {', '.join(f['goldens'])}" if f["goldens"] else "")
        )
    if not data["files"]:
        out.append("  none")
    out.append("Required:")
    out += [f"  {r}  {rules.RECIPE_NAMES[r]}" for r in data["recipes"] if r in rules.RECIPE_NAMES]
    if data["kernel_identity"]:
        out.append(f"  R2  {rules.RECIPE_NAMES['R2']}: identical means class C4, R3 to R5 fall away")
    out.append("  golden: " + (", ".join(data["goldens"]) or "none") + f" ({data['golden_seconds']} s)")
    out.append("  skipped: " + (", ".join(data["skipped"]) or "none") + f" ({data['skipped_seconds']} s saved)")
    out.append("  unit tests: " + unit_text(data))
    out.append(
        f"  bit-exact (TONA_DETERMINISTIC=1): {'yes' if data['bit_exact'] else 'no'}; speed before and after: "
        f"{'yes' if data['speed'] else 'no'}"
    )
    for key, text in (("long_context_files", "long-context run needed"), ("pressure_files", "pressure series needed")):
        if data[key]:
            out.append(f"  {text}, because of {', '.join(data[key])}")
    return "\n".join(out)


def unit_text(data: dict) -> str:
    if data["unit_all"]:
        return "all (just test)"
    parts = [f"ctest -R '^({'|'.join(data['unit_tests'])})$'"] if data["unit_tests"] else []
    parts += ["just test"] * data["python_tests"] + ["installer build and ctest"] * data["installer_tests"]
    return "; ".join(parts) or "none"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--base", default=None, help="what the change is compared with (default origin/main, else main)"
    )
    parser.add_argument("--build", default="work/build/engine", help="engine build whose graph decides")
    parser.add_argument("--head", default="HEAD", help="the commit to check (default HEAD)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    base = args.base or default_base()
    refs = (git("merge-base", base, args.head).strip(), git("rev-parse", args.head).strip())
    data = plan_for(changed_files(base, args.head), Path(args.build), refs).to_dict()
    print(json.dumps(data, indent=2) if args.json else render(data, base))
    return 0


if __name__ == "__main__":
    sys.exit(main())
