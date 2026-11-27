"""Runs the tests of one exercise in a rootless Podman container: no network, read-only root file system, tmpfs for
caches, all capabilities dropped, memory, CPU, process and time limits (same hardening as bench/threshold/sandbox.py).

The staged copy of the exercise (without .meta and .docs, so no reference solution is inside) is mounted at /work
(C++: /work/<slug>, because the exercises' CMakeLists.txt names the target after the directory).
"""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from tasks import Exercise, reference_files

STAGE_DIR = "/var/tmp/claude/139"  # staged copies of the exercises; not /tmp, which is RAM here
TIMEOUT_S = 240  # a cold cargo build of an exercise is the slowest case, measured at about 60 s on the reference PC
OUTPUT_TAIL = 6000
JS_PREP = (
    r"sed -i 's/\bxtest(/test(/;s/\bxit(/it(/;s/\bxdescribe(/describe(/' *.spec.js; "
    "ln -s /opt/node_modules node_modules"
)
# csharp: tests run with the host's .NET SDK and an offline NuGet cache, both mounted read-only (EXTRA_VOLUMES); the
# exercises target net10.0, the SDK on this machine is 11, so the target framework is overridden.
CS_PREP = 'sed -i -E \'s/Skip = "[^"]*"//g\' *Tests.cs'
CS_FLAGS = "-p:TargetFramework=net11.0 -p:NuGetAudit=false -p:NoWarn=NETSDK1188"
# restore and test are two calls: `dotnet test` alone looked for the targeting packs on the network
CS_TEST = (
    f"dotnet restore {CS_FLAGS} -p:RestoreSources=/nuget -v q && "
    f"dotnet test --no-restore {CS_FLAGS} -nodeReuse:false -p:UseSharedCompilation=false --nologo -v q "
    "--logger 'console;verbosity=normal'"
)
# java: gradle 8.7 as the exercises' wrappers ask for it, offline, dependencies from the image's read-only cache
JAVA_TEST = "gradle --offline --no-daemon test"
# cpp: Aider's benchmark/cpp-test.sh
CPP_TEST = 'mkdir -p build && cd build && cmake -DEXERCISM_RUN_ALL_TESTS=1 -G "Unix Makefiles" .. && make'
TS_TEST = "ln -s /opt/node_modules node_modules; npx jest --ci --no-cache"
COMMANDS = {
    "python": "python -m pytest -q -x -p no:cacheprovider {tests}",
    "go": "go test ./...",
    "rust": "cargo test --offline -- --include-ignored",
    "javascript": JS_PREP + "; npx jest --ci ./*",
    "csharp": CS_PREP + "; " + CS_TEST,
    "java": JAVA_TEST,
    "cpp": CPP_TEST,
    "php": "phpunit {tests}",
    "typescript": TS_TEST,
    "bash": "bats {tests}",
    "dart": "cp -r /opt/pub-cache /tmp/pub-cache && dart pub get --offline && dart test",
}
LANG_ENV = {
    "java": {"GRADLE_USER_HOME": "/tmp/gradle-home", "GRADLE_RO_DEP_CACHE": "/opt/gradle-home/caches"},
    "bash": {
        "BATS_RUN_SKIPPED": "true",
        "RUN_HARD_TEST": "true",
    },  # the track's own switch: every `skip` line checks it
    "typescript": {"NO_COLOR": "1", "FORCE_COLOR": "0"},
    "dart": {"PUB_CACHE": "/tmp/pub-cache", "DART_SUPPRESS_ANALYTICS": "true"},
    "csharp": {
        "PATH": "/dotnet:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "DOTNET_ROOT": "/dotnet",
        "NUGET_PACKAGES": "/nuget",
        "DOTNET_NOLOGO": "1",
        "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
        "DOTNET_SYSTEM_GLOBALIZATION_INVARIANT": "1",
        "DOTNET_CLI_HOME": "/tmp",
        "DOTNET_SKIP_FIRST_TIME_EXPERIENCE": "1",
    },
}
EXTRA_VOLUMES = {
    "csharp": [
        f"{Path.home()}/.dotnet:/dotnet:ro",
        "/var/tmp/claude/139/nuget:/nuget:ro",
    ]
}
ENV = {
    "HOME": "/tmp",
    "PYTHONDONTWRITEBYTECODE": "1",
    "GOCACHE": "/tmp/gocache",
    "GOPATH": "/tmp/go",
    "GOFLAGS": "-mod=mod",
    "GOTOOLCHAIN": "local",
    "CGO_ENABLED": "0",
    "CARGO_TARGET_DIR": "/tmp/target",
    "CI": "1",
}


# Skipped tests are switched on in the staged copy of the test files, as Aider does (Java: its regex; JavaScript's
# xtest is handled in the container, see JS_PREP). Test files are always the pristine ones from the exercise.
SKIP_PATTERNS = {
    "java": [(re.compile(r"@Disabled\([^)]*\)\s*\n"), "")],
    # every skipped test runs: the x-prefixed ones (the track's "enable one by one" marks) and `it.skip`
    "typescript": [
        (re.compile(r"\bx(it|test|describe)\("), r"\1("),
        (re.compile(r"\b(it|test|describe)\.skip\("), r"\1("),
    ],
    "dart": [(re.compile(r"\bskip:\s*true\b"), "skip: false")],
    "php": [(re.compile(r"^[ \t]*\$this->markTestSkipped\([^;]*\);[ \t]*\n", re.M), "")],
}


def enable_skipped_tests(ex: Exercise, dest: Path) -> None:
    for pattern, repl in SKIP_PATTERNS.get(ex.lang, []):
        for name in ex.tests:
            f = dest / name
            if f.exists():
                f.write_text(pattern.sub(repl, f.read_text()))


def stage(ex: Exercise, replacements: dict[str, str], dest: Path) -> None:
    shutil.copytree(ex.root, dest, ignore=shutil.ignore_patterns(".meta", ".docs"), dirs_exist_ok=True)
    for name, text in replacements.items():
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    enable_skipped_tests(ex, dest)


def run_tests(ex: Exercise, replacements: dict[str, str], cpus: int = 2) -> tuple[bool, str]:
    """(passed, tail of the output). A missing container or a timeout counts as failed with the reason in the text."""
    with tempfile.TemporaryDirectory(dir=STAGE_DIR) as tmp:
        work = Path(tmp) / "w"
        stage(ex, replacements, work)
        mount = f"/work/{ex.name}" if ex.lang == "cpp" else "/work"
        cmd = [
            "podman",
            "run",
            "--rm",
            "--network=none",
            "--read-only",
            "--tmpfs=/tmp:rw,size=2g,exec",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--memory=6g",
            "--memory-swap=6g",
            f"--cpus={cpus}",
            "--pids-limit=1024",
            f"--timeout={TIMEOUT_S}",
            f"--volume={work}:{mount}:rw,z",
            f"--workdir={mount}",
        ]
        cmd += [f"--volume={v}" for v in EXTRA_VOLUMES.get(ex.lang, [])]
        if ex.lang in EXTRA_VOLUMES:
            cmd.append("--security-opt=label=disable")  # the host SDK is mounted as it is, not relabelled
        for k, v in {**ENV, **LANG_ENV.get(ex.lang, {})}.items():
            cmd += ["--env", f"{k}={v}"]
        cmd += [f"localhost/tona-poly-{ex.lang}", "sh", "-c", COMMANDS[ex.lang].format(tests=" ".join(ex.tests))]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S + 60, check=False)
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        return p.returncode == 0, (p.stdout + p.stderr)[-OUTPUT_TAIL:]


def reference(ex: Exercise) -> tuple[bool, str]:
    """The reference solution of the exercise must pass in this sandbox, else the exercise is dropped for all models."""
    return run_tests(ex, reference_files(ex))
