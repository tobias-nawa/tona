# Common engine commands. Overview: just
# Every recipe activates the toolchain (toolchain/env.sh) itself.

set shell := ["bash", "-euo", "pipefail", "-c"]
# Settings of the installation (tona setup); without the file the defaults here apply. The environment wins
set dotenv-path := "work/tona.conf"

model := env("TONA_MODEL", "work/models/coder-iq1_m/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf")
# Drafts from the model's MTP head; DFlash2 instead:
# just drafter="--dflash work/models/draft/dflash-Qwen3.8-Flash-Next-Q8_0.gguf" serve
drafter := env("TONA_DRAFTER", "--mtp work/models/draft/mtp-Qwen3.8-Flash-Next-Q8_0.gguf")
profile := env("TONA_PROFILE", justfile_directory() / "work/profile/experts-coder.bin")
build_dir := "work/build/engine"
installer_dir := "work/build/installer"
# Branch the gates measure your changes against (origin/main, else main)
base := env("TONA_BASE", `git rev-parse --verify -q origin/main >/dev/null && echo origin/main || echo main`)
# pre-commit from the tools venv of scripts/setup.sh, else the one on PATH
pre_commit := if path_exists(justfile_directory() / "work/venv-tools/bin/pre-commit") == "true" { justfile_directory() / "work/venv-tools/bin/pre-commit" } else { "pre-commit" }
# Server port for serve, start, stop, top and smoke; another port: just port=9000 serve
port := env("TONA_PORT", "8662")
# Server context (256K, the most the model supports; KV cache from 1 GiB in RAM). Otherwise: just ctx=65536 serve
ctx := env("TONA_CTX", "262144")
# The ROCm toolchain on Linux; macOS builds the Metal backend with Xcode's toolchain (docs/use/macos.md)
env := if os() == "macos" { "" } else { "source toolchain/env.sh &&" }
engine_flags := if os() == "macos" { "-DTONA_BACKEND_METAL=ON" } else { "" }
# Memory limit for runs with the model: TONA_RAM_GB (page cache included), else RAM minus 6 GB for the rest
limit := `if [ -n "${TONA_RAM_GB:-}" ]; then awk "BEGIN {printf \"%d\", $TONA_RAM_GB * 1e9}"; elif [ "$(uname -s)" = Darwin ]; then echo $(( $(sysctl -n hw.memsize) / 1073741824 - 6 ))G; else echo $(( $(awk '/^MemTotal:/ {print $2}' /proc/meminfo) / 1048576 - 6 ))G; fi`
# macOS has no systemd: no memory limit there (the Metal backend maps the model, the kernel pages it)
# macOS keeps 6 threads: the engine sets the thread count only on Linux (backend/runtime_env.h), where it picks the physical cores
run := if os() == "macos" { "env OMP_NUM_THREADS=6" } else { "systemd-run --user --scope -q -p MemoryMax=" + limit + " -p MemorySwapMax=0 env" }

# List recipes
default:
    @just --list --unsorted

# Fetch and verify ROCm 10.1 into work/ (once)
setup:
    scripts/setup.sh

# Build, then run the guided setup of the checkout (options as `tona setup --help-dev`)
setup-tona *args: build
    {{build_dir}}/tona setup {{args}}

# Configure and build the engine
build:
    {{env}} cmake -S engine -B {{build_dir}} -G Ninja {{engine_flags}} >/dev/null && cmake --build {{build_dir}}

# All tests (some need the model)
test: build
    {{env}} ctest --test-dir {{build_dir}} --output-on-failure

# Unit tests without model and GPU, on any x86-64 CPU (no toolchain): the tests CI runs in `engine-cpu-tests`
test-cpu:
    cmake -S engine -B work/build/cpu -G Ninja -DTONA_MARCH=x86-64-v3 -DTONA_MODEL=/nonexistent/model.gguf -DTONA_LLAMACPP_BUILD=/nonexistent >/dev/null
    cmake --build work/build/cpu
    ctest --test-dir work/build/cpu -L unit --output-on-failure

# CPU token check on the mini models, built from the committed headers (CI job `mini-tokens`; no model, no GPU)
mini-check:
    cmake -S engine -B work/build/cpu -G Ninja -DTONA_MARCH=x86-64-v3 -DTONA_MODEL=/nonexistent/model.gguf -DTONA_LLAMACPP_BUILD=/nonexistent >/dev/null
    cmake --build work/build/cpu --target tona-ref
    python3 tools/mini_token_check.py compare --build work/build/cpu

# The same unit tests with AddressSanitizer and UndefinedBehaviorSanitizer (CI job `engine-sanitizers`)
test-sanitize:
    cmake -S engine -B work/build/sanitize -G Ninja -DCMAKE_BUILD_TYPE=Debug -DTONA_SANITIZE=ON -DTONA_MARCH=x86-64-v3 -DTONA_MODEL=/nonexistent/model.gguf -DTONA_LLAMACPP_BUILD=/nonexistent >/dev/null
    cmake --build work/build/sanitize
    ctest --test-dir work/build/sanitize -L unit --output-on-failure

# Coverage of the host code (C++ gcov, Python coverage.py) and its gates, report in work/build/coverage; with a base ref also the 80 % on changed lines
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

# Python tests of the gates and tools (no build needed)
test-py:
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

# Start the server (Anthropic /v1/messages, OpenAI /v1/chat/completions); more options e.g. --ctx 32768 --thinking on
serve *args: build
    {{env}} {{run}} TONA_EXPERT_PROFILE={{profile}} TONA_PROMPT_CACHE=work/prompt-cache {{build_dir}}/tona-server {{model}} {{drafter}} --port {{port}} --ctx {{ctx}} {{args}}

svc := "TONA_MODEL=" + model + " TONA_DRAFTER='" + drafter + "' TONA_PORT=" + port + " TONA_CTX=" + ctx + " TONA_PROFILE=" + profile + " scripts/tona-service.sh"

# Start the server in the background (systemd service tona-server; keeps running until just stop). Options go to tona-server
start *args: build
    TONA_ARGS={{quote(args)}} {{svc}} start

# Stop the server (service, or a just serve in another terminal)
stop:
    {{svc}} stop

# Server state in the terminal: start progress, running request, rates, VRAM/RAM; starts it if none is running
top *args: build
    TONA_START_CMD={{quote("just -f " + justfile() + " port=" + port + " ctx=" + ctx + " drafter=" + quote(drafter) + " start")}} \
    {{build_dir}}/tona --url http://127.0.0.1:{{port}} --start {{args}}

# Install the server as a permanent service: starts at every login (then pins ~25 GB RAM)
install-service *args: build
    TONA_ARGS={{quote(args)}} {{svc}} install

# Remove the permanent service
uninstall-service:
    {{svc}} uninstall

# Agent round against a running server, both APIs
smoke:
    /usr/bin/python3 -I tools/server_smoke.py http://127.0.0.1:{{port}}

# Generate text without a server (prompt in chat format, thinking off)
gen prompt n="200": build
    {{env}} {{run}} TONA_EXPERT_PROFILE={{profile}} {{build_dir}}/tona-gpu-gen {{model}} \
        "$(printf '<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n' {{quote(prompt)}})" \
        -n {{n}} --mega --dflash {{drafter}}

# Decode measurement: three prompts, runs runs each, median and range
bench runs="3" *env_vars: build
    {{env}} scripts/bench-decode.sh {{runs}} {{env_vars}}

# Verify pass without reloading (time per pass with 1 to 4 tokens, profile of the steps)
bench-pass: build
    {{env}} for nb in 1 2 3 4; do {{run}} TONA_FAKE_EXPERTS=1 TONA_BENCH_NB=$nb {{build_dir}}/tona-gpu-gen {{model}} "def fib(n):" --mega --batch-bench 200 | grep '^B ='; done

# Cross-check against the CPU reference (top-1 and logit deviation)
cross-check steps="6": build
    {{env}} {{run}} {{build_dir}}/tona-gpu-gen {{model}} "The capital of France is" -n 4 --check {{steps}} --mega 2>&1 | grep -E '^(Check|Error):'

# All code gates on your commits since the base branch: pre-commit hooks and clang-tidy (AGENTS.md, "Code gates")
check: _configure _tidy_venv
    {{pre_commit}} run --from-ref "$(git merge-base {{base}} HEAD)" --to-ref HEAD
    python3 tools/gates/openmp.py {{build_dir}}
    python3 tools/gates/tidy.py --base {{base}} --engine-db {{build_dir}} --installer-db {{installer_dir}}

# The same gates on every file and the whole git history; may be red until the clean-up is done
check-all: _configure _tidy_venv
    {{pre_commit}} run --all-files
    {{pre_commit}} run gitleaks-history --hook-stage manual
    python3 tools/gates/openmp.py {{build_dir}}
    python3 tools/gates/tidy.py --all --engine-db {{build_dir}} --installer-db {{installer_dir}}

# Format the files changed since the base branch (clang-format, ruff format; same pinned versions as the gates)
fmt:
    #!/usr/bin/env bash
    set -euo pipefail
    files=$({ git diff --name-only --diff-filter=ACMR "$(git merge-base {{base}} HEAD)"; git ls-files --others --exclude-standard; } | sort -u)
    [ -n "$files" ] || { echo "fmt: no changed files"; exit 0; }
    # pre-commit exits 1 when a hook rewrote files; that is the expected outcome here
    for hook in clang-format-fix ruff-format-fix; do
        echo "$files" | xargs {{pre_commit}} run "$hook" --hook-stage manual --files || true
    done

# Compile databases for clang-tidy (configure only, no build)
_configure:
    {{env}} cmake -S engine -B {{build_dir}} -G Ninja {{engine_flags}} >/dev/null
    {{env}} cmake -S installer -B {{installer_dir}} -G Ninja >/dev/null

# macOS: clang-tidy wheel for tools/gates/tidy.py in work/venv-tidy (Linux uses the ROCm toolchain's)
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

# Reproduce a GPU hang under the watchdog (once beforehand: sudo scripts/debug-setup.sh)
hang-repro rounds="3" *env_vars: build
    scripts/hang-repro.sh {{rounds}} {{env_vars}}

# Fuzz the parsers (api, gguf, minisig; libFuzzer, ASan, UBSan): seconds per target, optionally the targets; one folder per target in work/fuzz, crashes in <target>/findings
fuzz secs="300" *targets:
    scripts/fuzz.sh {{secs}} {{targets}}

# Check RAM after installing modules (memtester through sudo)
check-ram *args:
    scripts/check-ram.sh {{args}}

# Measure the yardsticks: llamacpp-hip, llamacpp-vulkan
baseline target="llamacpp-hip":
    scripts/baseline.sh {{target}}

# Usage of RAM, the model's page cache and VRAM
mem:
    @free -h | head -2
    @fincore -b {{model}} | tail -1 | awk '{printf "Model part 1 in page cache: %.1f GB\n", $1/1e9}'
    @echo "VRAM in use: $(( $(cat /sys/class/drm/card1/device/mem_info_vram_used) / 1048576 )) MiB of $(( $(cat /sys/class/drm/card1/device/mem_info_vram_total) / 1048576 )) MiB"
