# Developer commands, grouped by intent. Overview: just   (the names of the hidden recipes at the end: issue #20)
# Every recipe activates the toolchain (toolchain/env.sh) itself.

set shell := ["bash", "-euo", "pipefail", "-c"]
# Settings of the installation (tona setup); without the file the defaults here apply. The environment wins
set dotenv-path := "work/tona.conf"

model := env("TONA_MODEL", "work/models/coder-iq1_m/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf")
# Drafts from the model's MTP head; DFlash2 instead:
# just drafter="--dflash work/models/draft/dflash-Qwen3.8-Flash-Next-Q8_0.gguf" run
drafter := env("TONA_DRAFTER", "--mtp work/models/draft/mtp-Qwen3.8-Flash-Next-Q8_0.gguf")
profile := env("TONA_PROFILE", justfile_directory() / "work/profile/experts-coder.bin")
build_dir := "work/build/engine"
installer_dir := "work/build/installer"
# Branch the gates measure your changes against (origin/main, else main)
base := env("TONA_BASE", `git rev-parse --verify -q origin/main >/dev/null && echo origin/main || echo main`)
# pre-commit from the tools venv of scripts/setup.sh, else the one on PATH
pre_commit := if path_exists(justfile_directory() / "work/venv-tools/bin/pre-commit") == "true" { justfile_directory() / "work/venv-tools/bin/pre-commit" } else { "pre-commit" }
# Server port for start, stop and tona (the installed server's port); run and test-gpu take a free port instead, see run_port
port := env("TONA_PORT", "8662")
# Port of `just run`; empty: a free port, never 8662. A fixed one: just run_port=9000 run
run_port := env("TONA_RUN_PORT", "")
# Server context (256K, the most the model supports; KV cache from 1 GiB in RAM). Otherwise: just ctx=65536 run
ctx := env("TONA_CTX", "262144")
# The ROCm toolchain on Linux; macOS builds the Metal backend with Xcode's toolchain (docs/use/macos.md)
env := if os() == "macos" { "" } else { "source toolchain/env.sh &&" }
engine_flags := if os() == "macos" { "-DTONA_BACKEND_METAL=ON" } else { "" }
# Memory limit for runs with the model: TONA_RAM_GB (page cache included), else RAM minus 6 GB for the rest
limit := `if [ -n "${TONA_RAM_GB:-}" ]; then awk "BEGIN {printf \"%d\", $TONA_RAM_GB * 1e9}"; elif [ "$(uname -s)" = Darwin ]; then echo $(( $(sysctl -n hw.memsize) / 1073741824 - 6 ))G; else echo $(( $(awk '/^MemTotal:/ {print $2}' /proc/meminfo) / 1048576 - 6 ))G; fi`
# macOS has no systemd: no memory limit there (the Metal backend maps the model, the kernel pages it)
# macOS keeps 6 threads: the engine sets the thread count only on Linux (backend/runtime_env.h), where it picks the physical cores
limited := if os() == "macos" { "env OMP_NUM_THREADS=6" } else { "systemd-run --user --scope -q -p MemoryMax=" + limit + " -p MemorySwapMax=0 env" }


# List recipes, in the order of the groups
[private]
default:
    @just --list --unsorted

# Fetch and verify ROCm 10.1 into work/ (once); `just setup --no-gpu` on a machine without the GPU
[group('start')]
setup *args:
    scripts/setup.sh {{args}}

# Build, then run the guided setup of the checkout (options as `tona setup --help-dev`)
[group('start')]
setup-tona *args: build
    {{build_dir}}/tona setup {{args}}

# Configure and build the engine
[group('every day')]
build:
    {{env}} cmake -S engine -B {{build_dir}} -G Ninja {{engine_flags}} >/dev/null && cmake --build {{build_dir}}

# Format the files changed since the base branch (clang-format, ruff format; same pinned versions as the gates)
[group('every day')]
fmt:
    #!/usr/bin/env bash
    set -euo pipefail
    files=$({ git diff --name-only --diff-filter=ACMR "$(git merge-base {{base}} HEAD)"; git ls-files --others --exclude-standard; } | sort -u)
    [ -n "$files" ] || { echo "fmt: no changed files"; exit 0; }
    # pre-commit exits 1 when a hook rewrote files; that is the expected outcome here
    for hook in clang-format-fix ruff-format-fix; do
        echo "$files" | xargs {{pre_commit}} run "$hook" --hook-stage manual --files || true
    done

# All code gates on your commits since the base branch: pre-commit hooks and clang-tidy (AGENTS.md, "Code gates")
[group('every day')]
lint: _configure _tidy_venv
    {{pre_commit}} run --from-ref "$(git merge-base {{base}} HEAD)" --to-ref HEAD
    python3 tools/gates/openmp.py {{build_dir}}
    python3 tools/gates/tidy.py --base {{base}} --engine-db {{build_dir}} --installer-db {{installer_dir}}

# Everything without GPU and model, on any x86-64 CPU: what CI runs (unit tests, mini-model token check, pytest, installer tests)
[group('every day')]
test: _test-cpu _mini-check _test-py _test-installer

# Tests with GPU and model (ctest incl. the model tests), then the API test against a server of its own on a free port
[group('every day')]
test-gpu: build
    {{env}} ctest --test-dir {{build_dir}} --output-on-failure
    just --justfile {{quote(justfile())}} _api-test

# Before review: lint, test, test-gpu where a GPU and the model are, and the checks your change needs (tools/select_checks.py)
[group('every day')]
ready: lint test
    #!/usr/bin/env bash
    set -euo pipefail
    gpu=no
    if [ "$(uname -s)" = Darwin ]; then gpu=yes; elif [ -r /dev/kfd ] && [ -w /dev/kfd ]; then gpu=yes; fi
    if [ "$gpu" = yes ] && [ -e {{quote(model)}} ]; then
        just --justfile {{quote(justfile())}} test-gpu
    else
        echo "ready: test-gpu skipped (no GPU or no model here)"
    fi
    python3 tools/select_checks.py

# Server from the build in the foreground (Anthropic /v1/messages, OpenAI /v1/chat/completions) on a free port, never 8662; options go to tona-server, e.g. --ctx 32768 --thinking on
[group('run')]
run *args: build
    #!/usr/bin/env bash
    set -euo pipefail
    p={{quote(run_port)}}
    [ -n "$p" ] || p=$(just --justfile {{quote(justfile())}} _free-port)
    echo "tona-server on http://127.0.0.1:$p"
    {{env}} {{limited}} TONA_EXPERT_PROFILE={{profile}} TONA_PROMPT_CACHE=work/prompt-cache {{build_dir}}/tona-server {{model}} {{drafter}} --port "$p" --ctx {{ctx}} {{args}}

# The control centre from the build: server state, running request, rates, VRAM/RAM; starts the server if none runs
[group('run')]
tona *args: build
    TONA_START_CMD={{quote("just -f " + justfile() + " port=" + port + " ctx=" + ctx + " drafter=" + quote(drafter) + " start")}} \
    {{build_dir}}/tona --url http://127.0.0.1:{{port}} --start {{args}}

# One answer without a server (prompt in chat format, thinking off)
[group('run')]
ask prompt n="200": build
    {{env}} {{limited}} TONA_EXPERT_PROFILE={{profile}} {{build_dir}}/tona-gpu-gen {{model}} \
        "$(printf '<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n' {{quote(prompt)}})" \
        -n {{n}} --mega --dflash {{drafter}}

# Decode measurement: three prompts, runs runs each, median and range
[group('measure')]
bench runs="3" *env_vars: build
    {{env}} scripts/bench-decode.sh {{runs}} {{env_vars}}

# Verify pass without reloading (time per pass with 1 to 4 tokens, profile of the steps)
[group('measure')]
bench-pass: build
    {{env}} for nb in 1 2 3 4; do {{limited}} TONA_FAKE_EXPERTS=1 TONA_BENCH_NB=$nb {{build_dir}}/tona-gpu-gen {{model}} "def fib(n):" --mega --batch-bench 200 | grep '^B ='; done

# A/B speed comparison of two server variants, stops when the result is clear (options: scripts/ab.py --help, docs/contribute/testing.md)
[group('measure')]
ab *args: build
    {{env}} scripts/ab.py {{args}}

# Measure the yardsticks: llamacpp-hip, llamacpp-vulkan
[group('measure')]
baseline target="llamacpp-hip":
    scripts/baseline.sh {{target}}

# The unit tests with AddressSanitizer and UndefinedBehaviorSanitizer (CI job `engine-sanitizers`)
[group('deep')]
sanitize:
    cmake -S engine -B work/build/sanitize -G Ninja -DCMAKE_BUILD_TYPE=Debug -DTONA_SANITIZE=ON -DTONA_MARCH=x86-64-v3 -DTONA_MODEL=/nonexistent/model.gguf -DTONA_LLAMACPP_BUILD=/nonexistent >/dev/null
    cmake --build work/build/sanitize
    ctest --test-dir work/build/sanitize -L unit --output-on-failure

# Fuzz the parsers (api, gguf, minisig; libFuzzer, ASan, UBSan): seconds per target, optionally the targets; one folder per target in work/fuzz, crashes in <target>/findings
[group('deep')]
fuzz secs="300" *targets:
    scripts/fuzz.sh {{secs}} {{targets}}

# Coverage of the host code (C++ gcov, Python coverage.py) and its gates, report in work/build/coverage; with a base ref also the 80 % on changed lines
[group('deep')]
coverage base="":
    #!/usr/bin/env bash
    set -euo pipefail
    # work/ is a link to the shared folder in worktrees: resolve it, or the scripts in the venv would point into a worktree
    # that is deleted later. A venv whose scripts no longer start is built again.
    venv=$(realpath -m work/venv-test)
    if ! "$venv/bin/gcovr" --version >/dev/null 2>&1; then
        python3 -m venv --clear "$venv"
        "$venv/bin/pip" install -q --require-hashes -r tools/tests/requirements.txt
    fi
    PATH="$venv/bin:$PATH" "$venv/bin/python" tools/gates/coverage_gate.py run {{ if base != "" { "--base " + base } else { "" } }}
    echo "HTML reports: work/build/coverage/cpp/index.html and work/build/coverage/python/index.html"

# Cross-check against the CPU reference (top-1 and logit deviation)
[group('deep')]
cross-check steps="6": build
    {{env}} {{limited}} {{build_dir}}/tona-gpu-gen {{model}} "The capital of France is" -n 4 --check {{steps}} --mega 2>&1 | grep -E '^(Check|Error):'

# The same gates on every file and the whole git history; may be red until the clean-up is done
[group('deep')]
lint-all: _configure _tidy_venv
    {{pre_commit}} run --all-files
    {{pre_commit}} run gitleaks-history --hook-stage manual
    python3 tools/gates/openmp.py {{build_dir}}
    python3 tools/gates/tidy.py --all --engine-db {{build_dir}} --installer-db {{installer_dir}}

# Usage of RAM, the model's page cache and VRAM
[group('machine')]
mem:
    @free -h | head -2
    @fincore -b {{model}} | tail -1 | awk '{printf "Model part 1 in page cache: %.1f GB\n", $1/1e9}'
    @echo "VRAM in use: $(( $(cat /sys/class/drm/card1/device/mem_info_vram_used) / 1048576 )) MiB of $(( $(cat /sys/class/drm/card1/device/mem_info_vram_total) / 1048576 )) MiB"

# Check RAM after installing modules (memtester through sudo)
[group('machine')]
[confirm("check-ram runs memtester through sudo and loads the RAM for a long time. Continue?")]
check-ram *args:
    scripts/check-ram.sh {{args}}

# Reproduce a GPU hang under the watchdog (once beforehand: sudo scripts/debug-setup.sh)
[group('machine')]
[confirm("hang-repro provokes a GPU hang, which can take the desktop down. Continue?")]
hang-repro rounds="3" *env_vars: build
    scripts/hang-repro.sh {{rounds}} {{env_vars}}

# Hidden recipes: used by tona in a checkout, by the installer and by the recipes above

svc := "TONA_MODEL=" + model + " TONA_DRAFTER='" + drafter + "' TONA_PORT=" + port + " TONA_CTX=" + ctx + " TONA_PROFILE=" + profile + " scripts/tona-service.sh"

# Start the server in the background as systemd service tona-server (until stop); `tona start` in a checkout calls it. Hidden so that nobody touches the unit of an installation by mistake
[private]
start *args: build
    TONA_ARGS={{quote(args)}} {{svc}} start

# Stop that service (or a server of `just run`, by its port)
[private]
stop:
    {{svc}} stop

# Writes the permanent unit (starts at login); `tona setup` of a checkout calls it
[private]
install-service *args: build
    TONA_ARGS={{quote(args)}} {{svc}} install

# A port that is free now, never 8662
[private]
_free-port:
    #!/usr/bin/env bash
    set -euo pipefail
    while :; do
        p=$(python3 -I -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
        [ "$p" != 8662 ] && { echo "$p"; break; }
    done

# Unit tests without model and GPU (no toolchain): the tests CI runs in `engine-cpu-tests`
[private]
_test-cpu:
    cmake -S engine -B work/build/cpu -G Ninja -DTONA_MARCH=x86-64-v3 -DTONA_MODEL=/nonexistent/model.gguf -DTONA_LLAMACPP_BUILD=/nonexistent >/dev/null
    cmake --build work/build/cpu
    ctest --test-dir work/build/cpu -L unit --output-on-failure

# CPU token check on the mini models, built from the committed headers (exact logits run in CI job `mini-tokens`; local `just test` checks tokens only so other GCC/libm versions pass). 4 threads as on the CI runner: with every thread of a busy 32-thread machine the spinning OpenMP team takes minutes instead of 26 s
[private]
_mini-check:
    cmake -S engine -B work/build/cpu -G Ninja -DTONA_MARCH=x86-64-v3 -DTONA_MODEL=/nonexistent/model.gguf -DTONA_LLAMACPP_BUILD=/nonexistent >/dev/null
    cmake --build work/build/cpu --target tona-ref
    OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} python3 tools/mini_token_check.py compare --tokens-only --build work/build/cpu

# Installer build and tests (CI job `installer`; no toolchain)
[private]
_test-installer:
    cmake -S installer -B work/build/installer-test -G Ninja >/dev/null
    cmake --build work/build/installer-test
    ctest --test-dir work/build/installer-test --output-on-failure

# Agent round through both APIs against a server of its own on a free port (never 8662, never the tona-server unit); stops it at the end
[private]
_api-test:
    #!/usr/bin/env bash
    set -euo pipefail
    p=$(just --justfile {{quote(justfile())}} _free-port)
    log=work/api-test-$p.log
    # the trap also covers a failed or interrupted run: no server is left on the GPU
    trap 'kill "${pid:-}" 2>/dev/null || true; wait "${pid:-}" 2>/dev/null || true; rm -f "$log"' EXIT
    {{env}} {{limited}} TONA_EXPERT_PROFILE={{profile}} {{build_dir}}/tona-server {{model}} {{drafter}} --port "$p" --ctx 32768 >"$log" 2>&1 &
    pid=$!
    # 10 minutes: the limit of every wait in the project
    end=$(( $(date +%s) + 600 ))
    until curl -sf -m 1 "http://127.0.0.1:$p/status" | grep -q '"ready":true'; do
        kill -0 "$pid" 2>/dev/null || { cat "$log"; echo "api-test: server ended before it was ready"; exit 1; }
        [ "$(date +%s)" -lt "$end" ] || { cat "$log"; echo "api-test: server not ready after 10 min"; exit 1; }
        sleep 1
    done
    /usr/bin/python3 -I tools/server_smoke.py "http://127.0.0.1:$p"

# Compile databases for clang-tidy (configure only, no build)
[private]
_configure:
    {{env}} cmake -S engine -B {{build_dir}} -G Ninja {{engine_flags}} >/dev/null
    {{env}} cmake -S installer -B {{installer_dir}} -G Ninja >/dev/null

# macOS: clang-tidy wheel for tools/gates/tidy.py in work/venv-tidy (Linux uses the ROCm toolchain's)
[private]
_tidy_venv:
    #!/usr/bin/env bash
    set -euo pipefail
    [ "$(uname -s)" = Darwin ] || exit 0
    venv=$(python3 -I -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' work/venv-tidy)
    req=tools/gates/requirements-tidy-macos.txt
    want=$(python3 -I -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$req")
    if [ "$(cat "$venv/requirements.sha256" 2>/dev/null)" != "$want" ]; then
        python3 -m venv --clear "$venv"
        "$venv/bin/pip" install -q --require-hashes -r "$req"
        echo "$want" > "$venv/requirements.sha256"
    fi

# Python tests of the gates and tools (no build needed)
[private]
_test-py:
    #!/usr/bin/env bash
    set -euo pipefail
    # work/ is a link to the shared folder in worktrees: resolve it, or the scripts in the venv would point into a worktree
    # that is deleted later. A venv whose scripts no longer start is built again.
    if [ "$(uname -s)" = Darwin ]; then  # BSD realpath has no -m; Python's resolves missing paths the same way
        venv=$(python3 -I -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' work/venv-test)
    else
        venv=$(realpath -m work/venv-test)
    fi
    # The stamp holds the hash of the requirements it was built from: a new pin rebuilds it too
    want=$(python3 -I -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' tools/tests/requirements.txt)
    if ! "$venv/bin/pytest" --version >/dev/null 2>&1 || [ "$(cat "$venv/requirements.sha256" 2>/dev/null)" != "$want" ]; then
        python3 -m venv --clear "$venv"
        "$venv/bin/pip" install -q --require-hashes -r tools/tests/requirements.txt
        echo "$want" > "$venv/requirements.sha256"
    fi
    PATH="$venv/bin:$PATH" "$venv/bin/python" -m pytest tools/tests

# Hints for the old names (issue #20); delete them after the next release

[private]
[no-exit-message]
check *args:
    @echo "renamed: use just lint" >&2; exit 1

[private]
[no-exit-message]
check-all *args:
    @echo "renamed: use just lint-all" >&2; exit 1

[private]
[no-exit-message]
test-cpu *args:
    @echo "renamed: just test runs the unit tests and everything else CI runs without GPU" >&2; exit 1

[private]
[no-exit-message]
test-py *args:
    @echo "renamed: just test runs the Python tests too" >&2; exit 1

[private]
[no-exit-message]
mini-check *args:
    @echo "renamed: just test runs the mini-model token check too" >&2; exit 1

[private]
[no-exit-message]
test-sanitize *args:
    @echo "renamed: use just sanitize" >&2; exit 1

[private]
[no-exit-message]
serve *args:
    @echo "renamed: use just run" >&2; exit 1

[private]
[no-exit-message]
top *args:
    @echo "renamed: use just tona" >&2; exit 1

[private]
[no-exit-message]
gen *args:
    @echo "renamed: use just ask" >&2; exit 1

[private]
[no-exit-message]
smoke *args:
    @echo "merged: just test-gpu runs the API test" >&2; exit 1

[private]
[no-exit-message]
uninstall-service *args:
    @echo "removed: use tona autostart off" >&2; exit 1
