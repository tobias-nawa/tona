# Tona, the inference engine for exactly one machine (gfx1200). Build:
#   cmake -S engine -B work/build/engine -G Ninja && cmake --build work/build/engine && ctest --test-dir work/build/engine
# The GPU part needs the toolchain: `source toolchain/env.sh` first.
cmake_minimum_required(VERSION 3.28)
project(tona LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
# Set before any target exists: unit tests that compile server sources which the GPU part below marks as HIP would
# otherwise get an empty HIP_ARCHITECTURES (it is initialized when the target is created)
if(DEFINED ENV{ROCM_PATH} AND DEFINED ENV{TONA_HIPCXX})
    set(CMAKE_HIP_ARCHITECTURES gfx1200)
endif()
# GCC objects are also linked by the HIP linker (lld), which expects PIE
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
# One platform: Zen 5 with AVX-512. Only CI runners and contributors without a Zen 5 CPU change TONA_MARCH, e.g. to
# x86-64-v3; the default leaves every compiler command of the reference build as it was.
# Apple silicon (docs/use/macos.md): clang takes the CPU there through -mcpu, and x86 target names do not exist. The x86 branch is
# the reference line unchanged.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    set(TONA_MCPU apple-m1 CACHE STRING "CPU target of the host code on arm64 (-mcpu)")
    add_compile_options(-mcpu=${TONA_MCPU} -Wall -Wextra -Werror)
else()
    set(TONA_MARCH znver5 CACHE STRING "CPU target of the host code (-march)")
    add_compile_options(-march=${TONA_MARCH} -Wall -Wextra -Werror)
endif()
# macOS has no io_uring: the same DirectRows interface over pread (docs/use/macos.md)
if(APPLE)
    set(TONA_ROWS_SOURCE io/uring_pread.cpp)
else()
    set(TONA_ROWS_SOURCE io/uring.cpp)
endif()

# Release build (scripts/release-build.sh): the programs contain no absolute path, so that the same commit gives
# bit-identical programs in any checkout directory, and they find the ROCm runtime in ../lib next to bin/ (the install
# layout of docs/maintain/release.md). Without the option nothing below changes a compiler or linker command.
option(TONA_RELEASE "Release build: no absolute paths in the programs, ROCm runtime found through $ORIGIN/../lib" OFF)
if(TONA_RELEASE)
    get_filename_component(TONA_SOURCE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/.. ABSOLUTE)
    # When prefixes overlap, the last matching map wins (GCC and Clang): the build directory therefore comes after the
    # source tree it may lie in
    # -Wdate-time (with -Werror): __DATE__ and __TIME__ would make the programs differ from build to build
    add_compile_options(-ffile-prefix-map=${TONA_SOURCE_ROOT}=. -ffile-prefix-map=${CMAKE_BINARY_DIR}=build -Wdate-time)
    # The toolchain directory shows up through the header paths in the debug info and differs between machines; the
    # ROCm path lies inside the venv and comes after it
    if(DEFINED ENV{VIRTUAL_ENV})
        add_compile_options(-ffile-prefix-map=$ENV{VIRTUAL_ENV}=/toolchain)
    endif()
    if(DEFINED ENV{ROCM_PATH})
        add_compile_options(-ffile-prefix-map=$ENV{ROCM_PATH}=/rocm)
    endif()
    # Content-derived build id instead of a random one
    add_link_options(-Wl,--build-id=sha1)
    # --hip-link would add an absolute rpath to the toolchain's lib directory (clang HIP driver)
    add_link_options($<$<LINK_LANGUAGE:HIP>:-fno-rtlib-add-rpath>)
    # -fopenmp (HIP link of tona_gpu) would add the toolchain's lib directories the same way
    add_link_options($<$<LINK_LANGUAGE:HIP>:-fno-openmp-implicit-rpath>)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
    set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
endif()

# Hardening of the host code (engine/cmake/hardening.cmake); the release build always has it. Not part of the
# development build: the reference build keeps every compiler and linker command as it was.
option(TONA_HARDEN "Harden the host code: FORTIFY_SOURCE=3, stack protector, CET, PIE, full RELRO" OFF)
if(TONA_RELEASE OR TONA_HARDEN)
    include(cmake/hardening.cmake)
endif()

# Test builds (never the reference build): sanitizers, and line coverage at -O0 (gcov is only reliable unoptimized)
option(TONA_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(TONA_COVERAGE "Build with gcov instrumentation at -O0" OFF)
if(TONA_SANITIZE)
    add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()
if(TONA_COVERAGE)
    # atomic counter updates: threads in the tests otherwise corrupt gcov counters (negative hits stop gcovr, GCC bug 68080)
    add_compile_options(-O0 --coverage -fprofile-update=atomic)
    add_link_options(--coverage)
endif()

add_library(tona_model model/gguf.cpp model/quant.cpp model/quant_tile.cpp model/qwen4exp.cpp model/qwen35moe.cpp model/nemotron_h_moe.cpp model/arch.cpp model/binder.cpp model/blocks.cpp model/model_registry.cpp model/tokenizer.cpp model/pretok_pixtral.cpp model/progress.cpp ${TONA_ROWS_SOURCE})
target_include_directories(tona_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# The reference must match ggml bit for bit: no FMA contraction
set_source_files_properties(model/quant.cpp model/quant_tile.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
# CPU experts must round bit-identically to the GPU (cpu/detmath.h): explicit fma only

if(APPLE)
    # Apple clang ships no OpenMP runtime; Homebrew's libomp provides it (docs/use/macos.md)
    include(macos_openmp.cmake)
endif()
find_package(OpenMP REQUIRED)
include(openmp_pragmas.cmake)  # tona_openmp_for_pragmas(): -fopenmp for HIP sources that have a pragma
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    # experts.cpp is AVX-512 only; on arm64 every expert entry point throws (docs/use/macos.md)
    add_library(tona_cpu cpu/ref_factory.cpp cpu/ref_ops.cpp cpu/ref_blocks.cpp cpu/ref_helper.cpp cpu/ref_qwen4exp.cpp cpu/ref_qwen35moe.cpp cpu/ref_mamba2.cpp cpu/ref_nemotron_h.cpp cpu/dflash.cpp cpu/experts_unsupported.cpp)
else()
    add_library(tona_cpu cpu/ref_factory.cpp cpu/ref_ops.cpp cpu/ref_blocks.cpp cpu/ref_helper.cpp cpu/ref_qwen4exp.cpp cpu/ref_qwen35moe.cpp cpu/ref_mamba2.cpp cpu/ref_nemotron_h.cpp cpu/dflash.cpp cpu/experts.cpp)
endif()
set_source_files_properties(cpu/experts.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
if(NOT TONA_MARCH STREQUAL "znver5" AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    # experts.cpp uses AVX-512 intrinsics (VNNI, VBMI, ...) directly and needs the full Zen 5 instruction set to
    # compile; it only has to run on such a CPU, so no unit test calls it (test_cpu_experts needs the model anyway)
    set_property(SOURCE cpu/experts.cpp APPEND PROPERTY COMPILE_OPTIONS -march=znver5)
endif()
target_link_libraries(tona_cpu PUBLIC tona_model OpenMP::OpenMP_CXX)

add_executable(tona-ple-io tools/tona_ple_io.cpp)
target_link_libraries(tona-ple-io PRIVATE tona_model OpenMP::OpenMP_CXX)
add_executable(tona-gguf tools/tona_gguf.cpp)
target_link_libraries(tona-gguf PRIVATE tona_model)
add_executable(tona-model tools/tona_model.cpp)
target_link_libraries(tona-model PRIVATE tona_model)
add_executable(tona-tokenize tools/tona_tokenize.cpp)
target_link_libraries(tona-tokenize PRIVATE tona_model)
add_executable(tona-ref tools/tona_ref.cpp)
target_link_libraries(tona-ref PRIVATE tona_cpu)
add_executable(tona-layer-skip tools/tona_layer_skip.cpp)
target_link_libraries(tona-layer-skip PRIVATE tona_cpu)

add_library(tona_spec spec/suffix.cpp)
target_include_directories(tona_spec PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
add_executable(tona-sfx-policy tools/tona_sfx_policy.cpp server/suffix.cpp)
target_link_libraries(tona-sfx-policy PRIVATE tona_model)
add_executable(tona-spec-sim tools/tona_spec_sim.cpp)
target_link_libraries(tona-spec-sim PRIVATE tona_spec tona_model)

enable_testing()
# Unit tests: no model, no GPU, no network. The label `unit` is what CI and `just coverage` run (ctest -L unit).
function(tona_unit_test name)
    cmake_parse_arguments(T "" "" "SOURCES;LIBS;ARGS" ${ARGN})
    add_executable(test_${name} ${T_SOURCES})
    target_include_directories(test_${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_${name} PRIVATE ${T_LIBS})
    add_test(NAME ${name} COMMAND test_${name} ${T_ARGS})
    set_tests_properties(${name} PROPERTIES LABELS unit)
endfunction()

add_executable(test_gguf tests/test_gguf.cpp)
target_link_libraries(test_gguf PRIVATE tona_model)
add_test(NAME gguf COMMAND test_gguf)
# Offer data shared with tona setup (installer/CMakeLists.txt includes the same file)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
include(${CMAKE_CURRENT_SOURCE_DIR}/offer/offer.cmake)
tona_unit_test(offer SOURCES tests/test_offer.cpp LIBS tona_offer)
tona_unit_test(model_arch SOURCES tests/test_model_arch.cpp LIBS tona_model)
tona_unit_test(ref_ops SOURCES tests/test_ref_ops.cpp LIBS tona_cpu)
tona_unit_test(qwen35moe SOURCES tests/test_qwen35moe.cpp LIBS tona_cpu)
tona_unit_test(nemotron_h SOURCES tests/test_nemotron_h.cpp LIBS tona_cpu)
add_executable(test_suffix tests/test_suffix.cpp)
target_link_libraries(test_suffix PRIVATE tona_spec)
add_test(NAME suffix COMMAND test_suffix)
add_executable(test_suffix_index tests/test_suffix_index.cpp server/suffix.cpp)
target_include_directories(test_suffix_index PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME suffix_index COMMAND test_suffix_index)
add_executable(test_draft_calib tests/test_draft_calib.cpp)
target_include_directories(test_draft_calib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME draft_calib COMMAND test_draft_calib)

# Tests with the pinned model (tools/get_model.py coder-iq1_m), if present
set(TONA_MODEL ${CMAKE_CURRENT_SOURCE_DIR}/../work/models/coder-iq1_m/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf
    CACHE FILEPATH "Part 1 of the model for tests")
add_library(tona_chat_lib server/chat.cpp server/chat_format.cpp server/chat_qwen3_render.cpp server/chat_qwen3_parser.cpp server/chat_nemotron_render.cpp server/api.cpp
    server/api_anthropic.cpp server/api_openai.cpp)
target_include_directories(tona_chat_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# Fuzz entry (scripts/fuzz.sh builds the fuzzer with clang): an object target only, so the gates see the file
add_library(tona_fuzz_api OBJECT EXCLUDE_FROM_ALL tests/fuzz/fuzz_api.cpp)
target_link_libraries(tona_fuzz_api PRIVATE tona_chat_lib)
add_library(tona_fuzz_gguf OBJECT EXCLUDE_FROM_ALL tests/fuzz/fuzz_gguf.cpp)
target_link_libraries(tona_fuzz_gguf PRIVATE tona_model)
add_executable(tona-chat tools/tona_chat.cpp)
target_link_libraries(tona-chat PRIVATE tona_chat_lib tona_model)
add_executable(test_chat tests/test_chat.cpp)
target_link_libraries(test_chat PRIVATE tona_chat_lib)
add_test(NAME chat COMMAND test_chat)
set_tests_properties(gguf suffix suffix_index draft_calib chat PROPERTIES LABELS unit)
tona_unit_test(mtp_scorer SOURCES tests/test_mtp_scorer.cpp backend/mtp_scorer.cpp LIBS tona_model OpenMP::OpenMP_CXX)
tona_unit_test(api_anthropic SOURCES tests/test_api_anthropic.cpp LIBS tona_chat_lib)
tona_unit_test(api_openai SOURCES tests/test_api_openai.cpp LIBS tona_chat_lib)
tona_unit_test(api_common SOURCES tests/test_api_common.cpp LIBS tona_chat_lib)
tona_unit_test(wait_clock SOURCES tests/test_wait_clock.cpp)
tona_unit_test(evict_counter SOURCES tests/test_evict_counter.cpp)
if(NOT APPLE)  # KFD and CLOCK_BOOTTIME are Linux only
    tona_unit_test(kfd_smi SOURCES tests/test_kfd_smi.cpp gpu/kfd_smi.cpp gpu/trace_path.cpp)
endif()
tona_unit_test(runtime_env SOURCES tests/test_runtime_env.cpp)
tona_unit_test(gtt_budget SOURCES tests/test_gtt_budget.cpp)
tona_unit_test(kv_place SOURCES tests/test_kv_place.cpp)
tona_unit_test(prefill_order SOURCES tests/test_prefill_order.cpp)
tona_unit_test(cu_mask SOURCES tests/test_cu_mask.cpp gpu/cu_mask.cpp)
tona_unit_test(omp_guard SOURCES tests/test_omp_guard.cpp LIBS OpenMP::OpenMP_CXX)
tona_unit_test(pcie_link SOURCES tests/test_pcie_link.cpp gpu/pcie_link.cpp)
tona_unit_test(cpu_cost SOURCES tests/test_cpu_cost.cpp)
tona_unit_test(pf_check SOURCES tests/test_pf_check.cpp)
tona_unit_test(class_lru SOURCES tests/test_class_lru.cpp)
tona_unit_test(route_rank SOURCES tests/test_route_rank.cpp)
tona_unit_test(expert_layout SOURCES tests/test_expert_layout.cpp)
tona_unit_test(expert_parts SOURCES tests/test_expert_parts.cpp LIBS tona_model)
tona_unit_test(mega_extra SOURCES tests/test_mega_extra.cpp gpu/mega_extra.cpp)
tona_unit_test(tokenizer SOURCES tests/test_tokenizer.cpp LIBS tona_model)
tona_unit_test(sampling SOURCES tests/test_sampling.cpp server/sampling.cpp)
tona_unit_test(nan_guard SOURCES tests/test_nan_guard.cpp server/sampling.cpp)
tona_unit_test(quant SOURCES tests/test_quant.cpp LIBS tona_model)
tona_unit_test(shared_host SOURCES tests/test_shared_host.cpp LIBS tona_model)
tona_unit_test(mtp_shared SOURCES tests/test_mtp_shared.cpp)
tona_unit_test(ple_prefetch SOURCES tests/test_ple_prefetch.cpp)
tona_unit_test(expert_route SOURCES tests/test_expert_route.cpp ARGS default)
add_test(NAME expert_route_variants COMMAND test_expert_route variants)
set_tests_properties(expert_route_variants PROPERTIES LABELS unit)
tona_unit_test(expert_host SOURCES tests/test_expert_host.cpp)
tona_unit_test(expert_tiers SOURCES tests/test_expert_tiers.cpp)
tona_unit_test(vram_governor SOURCES tests/test_vram_governor.cpp)
tona_unit_test(quant_tile SOURCES tests/test_quant_tile.cpp LIBS tona_model)
tona_unit_test(http_util SOURCES tests/test_http_util.cpp server/http_util.cpp LIBS pthread)
tona_unit_test(request_guard SOURCES tests/test_request_guard.cpp server/request_guard.cpp)
tona_unit_test(env_budget SOURCES tests/test_env_budget.cpp ARGS limited)
add_test(NAME env_budget_unlimited COMMAND test_env_budget unlimited)
tona_unit_test(ram_governor SOURCES tests/test_ram_governor.cpp)
set_tests_properties(env_budget_unlimited PROPERTIES LABELS unit)
tona_unit_test(stall SOURCES tests/test_stall.cpp)
tona_unit_test(hold_policy SOURCES tests/test_hold_policy.cpp)
tona_unit_test(trace_path SOURCES tests/test_trace_path.cpp gpu/trace_path.cpp)
tona_unit_test(prompt_list SOURCES tests/test_prompt_list.cpp tools/tona_prompt_list.cpp)
tona_unit_test(rope_yarn SOURCES tests/test_rope_yarn.cpp ARGS ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/yarn_ref.txt)
tona_unit_test(route_policy SOURCES tests/test_route_policy.cpp ARGS ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/route_ref.txt)
tona_unit_test(progress SOURCES tests/test_progress.cpp model/progress.cpp LIBS pthread)
tona_unit_test(uring SOURCES tests/test_uring.cpp ${TONA_ROWS_SOURCE} LIBS pthread)
set_tests_properties(uring PROPERTIES SKIP_RETURN_CODE 77)
tona_unit_test(pyjson SOURCES tests/test_pyjson.cpp LIBS tona_chat_lib)
tona_unit_test(checkpoint_file SOURCES tests/test_checkpoint_file.cpp server/checkpoint_file.cpp)
tona_unit_test(run_split SOURCES tests/test_run_split.cpp)
tona_unit_test(drafter_options SOURCES tests/test_drafter_options.cpp LIBS tona_model)
tona_unit_test(job SOURCES tests/test_job.cpp server/job.cpp LIBS pthread)
# Environment read once per process: two scenarios as two processes
tona_unit_test(request_options SOURCES tests/test_request_options.cpp server/request_options.cpp LIBS tona_chat_lib pthread
             ARGS defaults)
add_test(NAME request_options_temperature COMMAND test_request_options temperature)
set_tests_properties(request_options_temperature PROPERTIES LABELS unit)
add_executable(test_cpu_experts tests/test_cpu_experts.cpp)
target_link_libraries(test_cpu_experts PRIVATE tona_cpu)
if(EXISTS ${TONA_MODEL})
    add_test(NAME model_shapes COMMAND tona-model ${TONA_MODEL})
    add_test(NAME cpu_experts COMMAND test_cpu_experts)
    # jinja2 from the system Python (the toolchain Python does not have it)
    find_program(TONA_SYS_PYTHON python3 PATHS /usr/bin NO_DEFAULT_PATH)
    if(TONA_SYS_PYTHON)
        add_test(NAME chat_vs_jinja COMMAND ${TONA_SYS_PYTHON} -I ${CMAKE_CURRENT_SOURCE_DIR}/tests/chat_vs_jinja.py
                 $<TARGET_FILE:tona-chat> ${TONA_MODEL} ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/chat_cases.json)
    endif()
    add_test(NAME chat_tokens COMMAND tona-chat --tokens ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/chat_cases.json ${TONA_MODEL})
    set_tests_properties(cpu_experts PROPERTIES ENVIRONMENT TONA_MODEL=${TONA_MODEL})
else()
    # otherwise `just test` is green with few tests and the core paths go unchecked
    message(WARNING "Model missing (${TONA_MODEL}): model_shapes, cpu_experts, chat_vs_jinja, chat_tokens, tokenizer_llamacpp "
                    "and dequant_ggml_model are skipped")
endif()

# Tests with the Bolt file (nemotron_h_moe), if present: the module binds every tensor, the chat format equals the GGUF's
# template byte for byte, the pieces tokenize like the whole text
set(TONA_BOLT_MODEL ${CMAKE_CURRENT_SOURCE_DIR}/../work/models/nemotron-bolt-74b-a9b-24gb/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf
    CACHE FILEPATH "Nemotron 3.5 Bolt GGUF for tests")
if(EXISTS ${TONA_BOLT_MODEL})
    find_program(TONA_SYS_PYTHON python3 PATHS /usr/bin NO_DEFAULT_PATH)
    add_test(NAME model_shapes_bolt COMMAND tona-model ${TONA_BOLT_MODEL})
    if(TONA_SYS_PYTHON)
        add_test(NAME chat_vs_jinja_nemotron COMMAND ${TONA_SYS_PYTHON} -I ${CMAKE_CURRENT_SOURCE_DIR}/tests/chat_vs_jinja.py
                 $<TARGET_FILE:tona-chat> ${TONA_BOLT_MODEL} ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/chat_cases_nemotron.json)
    endif()
    add_test(NAME chat_tokens_nemotron COMMAND tona-chat --model ${TONA_BOLT_MODEL} --tokens
             ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/chat_cases_nemotron.json ${TONA_BOLT_MODEL})
endif()

# Ornith's file: its chat template against ours, byte for byte (the family comes from the model)
set(TONA_ORNITH_MODEL ${CMAKE_CURRENT_SOURCE_DIR}/../work/models/ornith-1.5-35b-a3b/Ornith-1.5-35B-Q4_K_M.gguf
    CACHE FILEPATH "Ornith model for tests")
if(EXISTS ${TONA_ORNITH_MODEL})
    find_program(TONA_SYS_PYTHON python3 PATHS /usr/bin NO_DEFAULT_PATH)
    if(TONA_SYS_PYTHON)
        add_test(NAME chat_vs_jinja_ornith COMMAND ${TONA_SYS_PYTHON} -I ${CMAKE_CURRENT_SOURCE_DIR}/tests/chat_vs_jinja.py
                 $<TARGET_FILE:tona-chat> ${TONA_ORNITH_MODEL} ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/chat_cases.json)
    endif()
    add_test(NAME chat_tokens_ornith COMMAND tona-chat --model ${TONA_ORNITH_MODEL} --tokens
             ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/chat_cases.json ${TONA_ORNITH_MODEL})
endif()

# Cross-check against ggml from the pinned llama.cpp build (scripts/build-llamacpp.sh), tests only
set(TONA_LLAMACPP_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../work/src/llama.cpp CACHE PATH "llama.cpp sources")
set(TONA_LLAMACPP_BUILD ${CMAKE_CURRENT_SOURCE_DIR}/../work/build/llama.cpp-vulkan CACHE PATH "llama.cpp build")
if(EXISTS ${TONA_LLAMACPP_BUILD}/bin/libggml-base.so)
    add_executable(test_dequant_ggml tests/test_dequant_ggml.cpp)
    target_include_directories(test_dequant_ggml PRIVATE ${TONA_LLAMACPP_SRC}/ggml/include)
    target_link_libraries(test_dequant_ggml PRIVATE tona_model ${TONA_LLAMACPP_BUILD}/bin/libggml-base.so)
    set_target_properties(test_dequant_ggml PROPERTIES BUILD_RPATH ${TONA_LLAMACPP_BUILD}/bin)
    add_test(NAME dequant_ggml COMMAND test_dequant_ggml)
    find_package(Python3 COMPONENTS Interpreter)
    if(EXISTS ${TONA_MODEL} AND Python3_FOUND AND EXISTS ${TONA_LLAMACPP_BUILD}/bin/llama-tokenize)
        add_test(NAME tokenizer_llamacpp COMMAND ${Python3_EXECUTABLE} -I
                 ${CMAKE_CURRENT_SOURCE_DIR}/tests/tokenizer_vs_llamacpp.py $<TARGET_FILE:tona-tokenize>
                 ${TONA_LLAMACPP_BUILD}/bin/llama-tokenize ${TONA_MODEL}
                 ${CMAKE_CURRENT_SOURCE_DIR}/../bench/corpus/python-stdlib-3.14.txt --fuzz 40)
        set_tests_properties(tokenizer_llamacpp PROPERTIES TIMEOUT 600)
    endif()
    if(EXISTS ${TONA_MODEL})
        add_test(NAME dequant_ggml_model COMMAND test_dequant_ggml)
        set_tests_properties(dequant_ggml_model PROPERTIES ENVIRONMENT TONA_MODEL=${TONA_MODEL})
    endif()
else()
    message(WARNING "llama.cpp build missing, test_dequant_ggml and tokenizer_llamacpp are skipped (scripts/build-llamacpp.sh vulkan)")
endif()

# Backend selection (backend/registry.h): every program that creates a backend links this and registers its backends
add_library(tona_backend backend/registry.cpp)
target_include_directories(tona_backend PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tona_backend PUBLIC tona_model)
tona_unit_test(backend_registry SOURCES tests/test_backend_registry.cpp LIBS tona_backend)
option(TONA_BACKEND_METAL "Metal backend for Apple silicon (tier 3, engine/backend/metal/AGENTS.md)" OFF)
if(APPLE AND TONA_BACKEND_METAL)
    add_subdirectory(backend/metal)
endif()
set(TONA_SERVER_SOURCES
    server/server_main.cpp server/drafter_factory.cpp server/routes_common.cpp server/routes_status.cpp
    server/routes_anthropic.cpp server/routes_openai.cpp server/http_util.cpp server/request_guard.cpp server/job.cpp
    server/request_options.cpp server/system_status.cpp server/engine.cpp server/hold_controller.cpp server/engine_report.cpp
    server/generator.cpp server/prefill.cpp server/prefill_warm.cpp server/prompt_cache.cpp server/checkpoint_file.cpp server/draft.cpp server/verify.cpp
    server/sampling.cpp)

# GPU part (HIP, gfx1200): only with the toolchain active (source toolchain/env.sh)
if(DEFINED ENV{ROCM_PATH} AND DEFINED ENV{TONA_HIPCXX})
    set(CMAKE_HIP_COMPILER $ENV{TONA_HIPCXX})
    set(CMAKE_HIP_ARCHITECTURES gfx1200)
    set(CMAKE_HIP_STANDARD 23)
    list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH})
    enable_language(HIP)
    if(COMMAND tona_harden_hip_runtime)
        tona_harden_hip_runtime()
    endif()
    # Fedora's /usr/include/hip (7.1) must never be picked up
    set(TONA_HIP_ISYSTEM -isystem $ENV{ROCM_PATH}/include)
    add_library(tona_gpu gpu/matvec.hip gpu/ops.hip gpu/forward.hip gpu/forward_moe.hip gpu/arch/qwen4exp/kernel.hip gpu/arch/qwen4exp/kernel_512.hip gpu/arch/qwen35moe/kernel.hip gpu/mega_weight_kernels.hip gpu/gemm_f16.hip gpu/gemm_q.hip gpu/gemm_fp8.hip gpu/gemm_dequant.hip gpu/gemm_dequant_bolt.hip gpu/gemm_q_bolt.hip gpu/prefill_hc.hip gpu/prefill_res.hip gpu/prefill_dn.hip gpu/prefill_qsa.hip gpu/prefill_topk.hip gpu/prefill_attn.hip gpu/prefill_fattn.hip gpu/prefill_moe.hip gpu/prefill_route_sig.hip gpu/prefill_ple.hip gpu/prefill_head.hip gpu/mega_dump.cpp gpu/mega_launch.cpp gpu/prefill_plan.cpp gpu/prefill_entry.cpp gpu/prefill_run.cpp gpu/prefill_dense.cpp gpu/arch/qwen4exp/prefill_layer.cpp gpu/arch/qwen4exp/prefill_residual.cpp gpu/arch/qwen4exp/prefill_mtp.cpp gpu/arch/qwen35moe/prefill_layer.cpp gpu/arch/qwen35moe/prefill_mtp.cpp gpu/prefill_moe_run.cpp gpu/prefill_mtp.cpp
        gpu/mega_alloc.cpp gpu/alloc_trace.cpp gpu/trace_path.cpp gpu/kfd_smi.cpp gpu/mega_seq.cpp gpu/mega_init.cpp gpu/cu_mask.cpp gpu/pcie_link.cpp gpu/gtt.cpp gpu/mega_build_ops.cpp gpu/mega_build_bufs.cpp gpu/mega_build_slots.cpp gpu/mega_extra.cpp gpu/mega_elastic.cpp gpu/mega_vram.cpp gpu/arch/qwen4exp/layer.cpp gpu/arch/qwen4exp/moe.cpp backend/mtp_scorer.cpp gpu/arch/qwen4exp/mtp.cpp gpu/arch/qwen4exp/mtp_dyn.cpp gpu/mega_mtp_dyn.cpp gpu/arch/qwen4exp/draft.cpp gpu/mega_build.cpp gpu/arch/qwen4exp/program.cpp gpu/arch/qwen4exp/buffers.cpp gpu/arch/qwen35moe/program.cpp gpu/arch/qwen35moe/buffers.cpp gpu/arch/qwen35moe/layers.cpp gpu/arch/qwen35moe/moe.cpp gpu/arch/qwen35moe/mtp.cpp gpu/arch/qwen4exp/residual.cpp gpu/mega_upload.cpp gpu/mega_select.cpp gpu/mega_select_cpu.cpp gpu/mega_experts.cpp gpu/mega_prefetch.cpp gpu/mega_serve.cpp gpu/mega_tap.cpp gpu/evict_counter.cpp gpu/mega_abort.cpp gpu/mega_mtp.cpp gpu/mega_state.cpp gpu/guarded_copy.cpp gpu/clock_hold.cpp)
    set_source_files_properties(gpu/mega_dump.cpp gpu/mega_launch.cpp gpu/prefill_plan.cpp gpu/prefill_entry.cpp gpu/prefill_run.cpp gpu/prefill_dense.cpp gpu/arch/qwen4exp/prefill_layer.cpp gpu/arch/qwen4exp/prefill_residual.cpp gpu/arch/qwen4exp/prefill_mtp.cpp gpu/arch/qwen35moe/prefill_layer.cpp gpu/arch/qwen35moe/prefill_mtp.cpp gpu/prefill_moe_run.cpp gpu/prefill_mtp.cpp gpu/mega_alloc.cpp gpu/alloc_trace.cpp gpu/mega_seq.cpp gpu/mega_init.cpp gpu/cu_mask.cpp gpu/pcie_link.cpp gpu/gtt.cpp gpu/mega_build_ops.cpp gpu/mega_build_bufs.cpp gpu/mega_build_slots.cpp gpu/mega_extra.cpp gpu/mega_elastic.cpp gpu/mega_vram.cpp gpu/arch/qwen4exp/layer.cpp gpu/arch/qwen4exp/moe.cpp gpu/arch/qwen4exp/mtp.cpp gpu/arch/qwen4exp/mtp_dyn.cpp gpu/mega_mtp_dyn.cpp gpu/arch/qwen4exp/draft.cpp gpu/mega_build.cpp gpu/arch/qwen4exp/program.cpp gpu/arch/qwen4exp/buffers.cpp gpu/arch/qwen35moe/program.cpp gpu/arch/qwen35moe/buffers.cpp gpu/arch/qwen35moe/layers.cpp gpu/arch/qwen35moe/moe.cpp gpu/arch/qwen35moe/mtp.cpp gpu/arch/qwen4exp/residual.cpp gpu/mega_upload.cpp gpu/mega_select.cpp gpu/mega_select_cpu.cpp gpu/mega_experts.cpp gpu/mega_prefetch.cpp gpu/mega_serve.cpp gpu/mega_tap.cpp gpu/evict_counter.cpp gpu/mega_abort.cpp gpu/mega_mtp.cpp gpu/mega_state.cpp gpu/guarded_copy.cpp gpu/clock_hold.cpp PROPERTIES LANGUAGE HIP)
    # Bolt (nemotron_h_moe): Mamba-2 prefill kernels and the layer state
    target_sources(tona_gpu PRIVATE gpu/prefill_mamba.hip gpu/prefill_mamba_ssd.hip gpu/arch/nemotron_h/state.cpp)
    set_source_files_properties(gpu/arch/nemotron_h/state.cpp PROPERTIES LANGUAGE HIP)
    # Bolt (nemotron_h_moe): latent MoE block and its prefill kernels
    target_sources(tona_gpu PRIVATE gpu/prefill_moe_latent.hip gpu/arch/nemotron_h/moe_prefill.cpp gpu/arch/nemotron_h/moe.cpp)
    # Bolt (nemotron_h_moe): the architecture module (program, blocks, kernel unit, prefill sequence)
    target_sources(tona_gpu PRIVATE gpu/arch/nemotron_h/kernel.hip gpu/prefill_kvstore.hip gpu/arch/nemotron_h/program.cpp gpu/arch/nemotron_h/blocks.cpp gpu/arch/nemotron_h/buffers.cpp gpu/arch/nemotron_h/prefill_layer.cpp)
    set_source_files_properties(gpu/arch/nemotron_h/program.cpp gpu/arch/nemotron_h/blocks.cpp gpu/arch/nemotron_h/buffers.cpp gpu/arch/nemotron_h/prefill_layer.cpp PROPERTIES LANGUAGE HIP)
    set_source_files_properties(gpu/arch/nemotron_h/moe_prefill.cpp gpu/arch/nemotron_h/moe.cpp PROPERTIES LANGUAGE HIP)
    target_include_directories(tona_gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(tona_gpu PUBLIC $<$<COMPILE_LANGUAGE:HIP>:${TONA_HIP_ISYSTEM}>)
    # Line tables in GPU code: on a hang rocgdb shows the source line of every wave (scripts/gpu-watch.sh)
    target_compile_options(tona_gpu PUBLIC $<$<COMPILE_LANGUAGE:HIP>:-gline-tables-only>)
    target_link_libraries(tona_gpu PUBLIC tona_model tona_cpu OpenMP::OpenMP_CXX)
    tona_openmp_for_pragmas(tona_gpu)
    # The HIP link of the executables needs the OpenMP runtime of the toolchain: find_package(OpenMP) names the one of the
    # system compiler, and the host code of the HIP sources now calls __kmpc_*
    target_link_options(tona_gpu PUBLIC $<$<LINK_LANGUAGE:HIP>:-fopenmp>)
    # Register cap of the matvec chain (amdgpu_waves_per_eu), measured in research/measurements/2026-10-23-megakernel
    set(TONA_CHAIN_WAVES 8 CACHE STRING "Waves per SIMD for k_chain_* and k_mega (measurements/2026-10-23-megakernel)")
    target_compile_definitions(tona_gpu PRIVATE CHAIN_WAVES=${TONA_CHAIN_WAVES})
    option(TONA_WAIT_DIAG "Diagnostics of the megakernel waits (TONA_MEGA_GAP_HIST, TONA_TEST_FREEZE_*); costs registers" OFF)
    if(TONA_WAIT_DIAG)
        target_compile_definitions(tona_gpu PRIVATE TONA_WAIT_DIAG)
    endif()
    set(TONA_GPU_GEN_SOURCES tools/tona_gpu_gen.cpp tools/tona_gpu_gen_teacher.cpp tools/tona_gpu_gen_tail.cpp tools/tona_gpu_gen_batch.cpp
        tools/tona_gpu_gen_multi.cpp tools/tona_gpu_gen_spec.cpp tools/tona_gpu_gen_spec_report.cpp
        tools/tona_gpu_gen_plain.cpp tools/tona_prompt_list.cpp)
    add_executable(tona-gpu-gen ${TONA_GPU_GEN_SOURCES})
    set_source_files_properties(${TONA_GPU_GEN_SOURCES} PROPERTIES LANGUAGE HIP)
    target_link_libraries(tona-gpu-gen PRIVATE tona_gpu tona_cpu)
    add_executable(tona-server ${TONA_SERVER_SOURCES} server/suffix.cpp)
    set_source_files_properties(${TONA_SERVER_SOURCES} PROPERTIES LANGUAGE HIP)
    target_link_libraries(tona-server PRIVATE tona_gpu tona_cpu tona_chat_lib tona_backend)
    tona_openmp_for_pragmas(tona-server)
    target_compile_definitions(tona-server PRIVATE TONA_HAVE_REFERENCE=1)
    # The same backend check as on macOS, here against the reference: its output is the golden file other backends
    # compare with where cpu/ref has no counterpart (the MTP head)
    add_executable(tona-check tools/tona_check.cpp tools/tona_check_mtp.cpp)
    set_source_files_properties(tools/tona_check.cpp PROPERTIES LANGUAGE HIP)
    target_link_libraries(tona-check PRIVATE tona_gpu tona_cpu tona_backend)
    target_compile_definitions(tona-check PRIVATE TONA_HAVE_REFERENCE=1)
    # The logits of a fixed token sequence, to compare with another engine's (tools/logits_compare.py)
    add_executable(tona-logits-dump tools/tona_logits_dump.cpp)
    set_source_files_properties(tools/tona_logits_dump.cpp PROPERTIES LANGUAGE HIP)
    target_link_libraries(tona-logits-dump PRIVATE tona_gpu tona_cpu tona_backend)
    target_compile_definitions(tona-logits-dump PRIVATE TONA_HAVE_REFERENCE=1)
    # N2c of Bolt: every block of the GPU program against the oracle's trace (needs the model and the GPU)
    add_executable(tona-bolt-layers tools/tona_bolt_layers.cpp)
    set_source_files_properties(tools/tona_bolt_layers.cpp PROPERTIES LANGUAGE HIP)
    target_link_libraries(tona-bolt-layers PRIVATE tona_gpu tona_cpu tona_backend)
    target_compile_definitions(tona-bolt-layers PRIVATE TONA_HAVE_REFERENCE=1)
    # Speed of prefill and decode through the generic backend interface (any architecture), with the step profile
    add_executable(tona-bolt-bench tools/tona_bolt_bench.cpp)
    set_source_files_properties(tools/tona_bolt_bench.cpp PROPERTIES LANGUAGE HIP)
    target_link_libraries(tona-bolt-bench PRIVATE tona_gpu tona_cpu tona_backend)
    target_compile_definitions(tona-bolt-bench PRIVATE TONA_HAVE_REFERENCE=1)
    add_executable(tona-gpu-chain tools/tona_gpu_chain.hip)
    target_link_libraries(tona-gpu-chain PRIVATE tona_gpu tona_model)
    add_executable(tona-gpu-matvec tools/tona_gpu_matvec.hip)
    target_link_libraries(tona-gpu-matvec PRIVATE tona_gpu tona_model OpenMP::OpenMP_CXX)
    tona_openmp_for_pragmas(tona-gpu-matvec)
    add_executable(tona-expert-check tools/tona_expert_check.hip)
    target_link_libraries(tona-expert-check PRIVATE tona_gpu tona_cpu tona_model)
    # CPU and GPU experts bit-identical; needs the GPU (small kernels, no megakernel)
    if(EXISTS ${TONA_MODEL})
        add_test(NAME expert_bitexact COMMAND tona-expert-check ${TONA_MODEL})
    endif()
    # k_topk against a CPU reference of the selection rule (cells and ncells byte for byte); needs the GPU, no model
    add_executable(test_gpu_topk tests/test_gpu_topk.hip)
    target_include_directories(test_gpu_topk PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_gpu_topk PRIVATE tona_gpu)
    add_test(NAME gpu_topk COMMAND test_gpu_topk)
    set_tests_properties(gpu_topk PROPERTIES LABELS gpu)
    # k_route with NaN, +inf and mixed rows: experts in range, counts add up, NaN rows counted; needs the GPU
    add_executable(test_gpu_route tests/test_gpu_route.hip)
    target_include_directories(test_gpu_route PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_gpu_route PRIVATE tona_gpu)
    add_test(NAME gpu_route COMMAND test_gpu_route)
    set_tests_properties(gpu_route PROPERTIES LABELS gpu)
    # The sigmoid routing policy on the GPU (step and prefill kernel) against the CPU side and the float64 reference
    add_library(tona_gpu_probe_route STATIC gpu/probe/route_sig_step.hip)
    target_include_directories(tona_gpu_probe_route PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(tona_gpu_probe_route PUBLIC $<$<COMPILE_LANGUAGE:HIP>:${TONA_HIP_ISYSTEM}>)
    target_compile_definitions(tona_gpu_probe_route PRIVATE CHAIN_WAVES=${TONA_CHAIN_WAVES})
    target_link_libraries(tona_gpu_probe_route PUBLIC tona_gpu)
    add_executable(test_gpu_route_sig tests/test_gpu_route_sig.hip)
    target_include_directories(test_gpu_route_sig PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(test_gpu_route_sig PRIVATE tona_gpu_probe_route)
    add_test(NAME gpu_route_sig COMMAND test_gpu_route_sig ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/route_ref.txt)
    set_tests_properties(gpu_route_sig PROPERTIES LABELS gpu)
    # op_mtp_pick and op_route of the megakernel with NaN and inf logits; needs the GPU, no model
    add_executable(test_gpu_mtp_nan tests/test_gpu_mtp_nan.hip)
    target_include_directories(test_gpu_mtp_nan PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_gpu_mtp_nan PRIVATE tona_gpu)
    add_test(NAME gpu_mtp_nan COMMAND test_gpu_mtp_nan)
    set_tests_properties(gpu_mtp_nan PROPERTIES LABELS gpu)
    # steps of the dynamic MTP head rows (projection, scores and selection, gather) against a CPU reference
    add_executable(test_gpu_mtp_dyn tests/test_gpu_mtp_dyn.hip)
    target_include_directories(test_gpu_mtp_dyn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_gpu_mtp_dyn PRIVATE tona_gpu)
    add_test(NAME gpu_mtp_dyn COMMAND test_gpu_mtp_dyn)
    set_tests_properties(gpu_mtp_dyn PROPERTIES LABELS gpu)
    # op_idx_score of the megakernel against the loop it replaced (scores bit for bit); needs the GPU, no model
    add_executable(test_gpu_idx_score tests/test_gpu_idx_score.hip)
    target_include_directories(test_gpu_idx_score PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_gpu_idx_score PRIVATE tona_gpu)
    add_test(NAME gpu_idx_score COMMAND test_gpu_idx_score)
    set_tests_properties(gpu_idx_score PROPERTIES LABELS gpu)
    # k_attn_wmma against its frozen predecessor, byte for byte (tests/attn_wmma_ref.h); needs the GPU, no model
    add_executable(test_gpu_attn_wmma tests/test_gpu_attn_wmma.hip)
    target_include_directories(test_gpu_attn_wmma PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(test_gpu_attn_wmma PRIVATE tona_gpu)
    add_test(NAME gpu_attn_wmma COMMAND test_gpu_attn_wmma)
    set_tests_properties(gpu_attn_wmma PROPERTIES LABELS gpu)

    # The real k_mega with test-only profiles (gpu/probe/), as KernelSets; only tests link this
    add_library(tona_gpu_probes STATIC gpu/probe/fattn_kernels.hip gpu/probe/resnorm_kernels.hip gpu/probe/quant_matvec_kernels.hip gpu/probe/mamba_kernels.hip gpu/probe/latent_moe_kernels.hip)
    target_include_directories(tona_gpu_probes PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(tona_gpu_probes PUBLIC $<$<COMPILE_LANGUAGE:HIP>:${TONA_HIP_ISYSTEM}>)
    target_compile_definitions(tona_gpu_probes PRIVATE CHAIN_WAVES=${TONA_CHAIN_WAVES})
    target_link_libraries(tona_gpu_probes PUBLIC tona_gpu)
    # Full-context attention (mega_op_fattn.h, prefill_fattn.hip) against the oracle's primitives, inside the real k_mega
    # with a test-only profile; needs the GPU, no model
    add_executable(test_gpu_fattn tests/test_gpu_fattn.hip)
    target_include_directories(test_gpu_fattn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(test_gpu_fattn PRIVATE tona_gpu_probes tona_cpu)
    add_test(NAME gpu_fattn COMMAND test_gpu_fattn)
    set_tests_properties(gpu_fattn PROPERTIES LABELS gpu)
    # The plain residual step (mega_op_resnorm.h, prefill_res.hip) against float64; needs the GPU, no model
    add_executable(test_gpu_resnorm tests/test_gpu_resnorm.hip)
    target_include_directories(test_gpu_resnorm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(test_gpu_resnorm PRIVATE tona_gpu_probes)
    add_test(NAME gpu_resnorm COMMAND test_gpu_resnorm)
    set_tests_properties(gpu_resnorm PROPERTIES LABELS gpu)
    # NVFP4 and IQ2_XXS (matvec in k_mega, wide pass, dequant_f16, gemm_q) against the CPU dequantization; needs the GPU;
    # with TONA_MODEL also the Bolt file's own tensors
    add_executable(test_gpu_bolt_quant tests/test_gpu_bolt_quant.hip)
    target_include_directories(test_gpu_bolt_quant PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(test_gpu_bolt_quant PRIVATE tona_gpu_probes tona_cpu tona_model)
    add_test(NAME gpu_bolt_quant COMMAND test_gpu_bolt_quant)
    set_tests_properties(gpu_bolt_quant PROPERTIES LABELS gpu)
    # The Mamba-2 steps (mega_op_mamba.h, prefill_mamba.hip) against the oracle's state-space step, the replay commit
    add_executable(test_gpu_mamba tests/test_gpu_mamba.hip)
    target_include_directories(test_gpu_mamba PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(test_gpu_mamba PRIVATE tona_gpu_probes tona_cpu)
    add_test(NAME gpu_mamba COMMAND test_gpu_mamba)
    set_tests_properties(gpu_mamba PROPERTIES LABELS gpu)
    # The latent MoE steps (mega_op_moe_latent.h, prefill_moe_latent.hip) against the oracle's formulas; needs the GPU
    add_executable(test_gpu_moe_latent tests/test_gpu_moe_latent.hip)
    target_include_directories(test_gpu_moe_latent PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(test_gpu_moe_latent PRIVATE tona_gpu_probes tona_cpu)
    add_test(NAME gpu_moe_latent COMMAND test_gpu_moe_latent)
    set_tests_properties(gpu_moe_latent PROPERTIES LABELS gpu)
    add_executable(tona-gemm-check tools/tona_gemm_check.hip)
    target_link_libraries(tona-gemm-check PRIVATE tona_gpu tona_model)
    add_executable(tona-gemm-q-bench tools/tona_gemm_q_bench.hip)
    target_link_libraries(tona-gemm-q-bench PRIVATE tona_gpu tona_model)
    add_executable(tona-wide-probe tools/tona_wide_probe.hip)
    target_link_libraries(tona-wide-probe PRIVATE tona_gpu tona_model)
    # Reproducible builds: clang derives the HIP compilation-unit id from the output path, so two build directories
    # give different binaries. A fixed id per source file (hash of its path in the tree) makes them bit-identical.
    get_property(TONA_TARGETS DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    foreach(t IN LISTS TONA_TARGETS)
        get_target_property(srcs ${t} SOURCES)
        foreach(src IN LISTS srcs)
            get_source_file_property(lang ${src} LANGUAGE)
            get_source_file_property(done ${src} TONA_CUID)
            if((lang STREQUAL "HIP" OR src MATCHES "\\.hip$") AND NOT done)
                string(MD5 cuid "${src}")
                string(SUBSTRING ${cuid} 0 16 cuid)
                set_property(SOURCE ${src} APPEND PROPERTY COMPILE_OPTIONS -cuid=${cuid})
                set_property(SOURCE ${src} PROPERTY TONA_CUID ${cuid})
            endif()
        endforeach()
    endforeach()
else()
    message(STATUS "ROCM_PATH/TONA_HIPCXX missing: GPU part skipped (source toolchain/env.sh)")
endif()

# macOS: the same server as plain C++; its backends link in below (docs/use/macos.md)
if(APPLE)
    add_executable(tona-server ${TONA_SERVER_SOURCES} server/suffix.cpp)
    target_link_libraries(tona-server PRIVATE tona_cpu tona_chat_lib tona_backend)
    if(TONA_BACKEND_METAL)
        target_link_libraries(tona-server PRIVATE tona_backend_metal)
        # Any backend against the CPU reference, step by step (docs/backends/README.md#checks)
        add_executable(tona-check tools/tona_check.cpp tools/tona_check_mtp.cpp)
        target_link_libraries(tona-check PRIVATE tona_cpu tona_backend tona_backend_metal)
        # Speed through the Backend interface alone (research/measurements/2026-11-11-mac-synth.md)
        add_executable(tona-bench tools/tona_bench.cpp)
        target_link_libraries(tona-bench PRIVATE tona_cpu tona_backend tona_backend_metal)
        # Quality of a lossy pass variant, teacher-forced against passes of 1 token (docs/backends/README.md#checks)
        add_executable(tona-width-nll tools/tona_width_nll.cpp)
        target_link_libraries(tona-width-nll PRIVATE tona_cpu tona_backend tona_backend_metal)
    endif()
endif()


# Updater (docs/use/update.md): signature check, download, install. Needs OpenSSL for HTTPS and Ed25519; only `tona` and
# the update tests link it, so no other program of the reference build gains the dependency. The httplib macro is
# PUBLIC on purpose: every translation unit of one program must see httplib.h with the same macros.
find_package(OpenSSL REQUIRED)
# The commit a build was made from, next to the binary (scripts/ab_plan.py reads tona-server.commit)
if(TARGET tona-server)
    add_custom_command(TARGET tona-server POST_BUILD
        COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_CURRENT_SOURCE_DIR} -DOUT=$<TARGET_FILE:tona-server>.commit
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/commit_stamp.cmake
        VERBATIM)
endif()

include(update/sources.cmake)  # the source list is shared with the installer project (installer/CMakeLists.txt)
add_library(tona_update ${TONA_UPDATE_SOURCES})
target_include_directories(tona_update PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(tona_update PUBLIC CPPHTTPLIB_OPENSSL_SUPPORT)
target_link_libraries(tona_update PUBLIC OpenSSL::SSL OpenSSL::Crypto pthread)
if(APPLE)
    # httplib takes the trusted roots from the macOS keychain (as from the system CA store on Linux)
    target_link_libraries(tona_update PUBLIC "-framework CoreFoundation" "-framework Security")
endif()
set(TONA_UPDATE_DATA ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/update)
tona_unit_test(update_minisig SOURCES tests/test_update_minisig.cpp LIBS tona_update ARGS ${TONA_UPDATE_DATA})
tona_unit_test(update_manifest SOURCES tests/test_update_manifest.cpp LIBS tona_update ARGS ${TONA_UPDATE_DATA})
tona_unit_test(update_fetch SOURCES tests/test_update_fetch.cpp LIBS tona_update)
tona_unit_test(update_install SOURCES tests/test_update_install.cpp LIBS tona_update)
tona_unit_test(update_switch SOURCES tests/test_update_switch.cpp LIBS tona_update)
tona_unit_test(update_updater SOURCES tests/test_update_updater.cpp LIBS tona_update)
add_executable(fake_tona_server tests/fake_tona_server.cpp)
target_include_directories(fake_tona_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fake_tona_server PRIVATE pthread)
tona_unit_test(update_script_service SOURCES tests/test_update_script_service.cpp LIBS tona_update
               ARGS $<TARGET_FILE:fake_tona_server>)
add_dependencies(test_update_script_service fake_tona_server)

# The program `tona`: control centre (ANSI, no UI library), the guided setup (FTXUI), commands without a screen, coding
# agents, background server. It reads and writes the settings with the installer's core (installer/core.cmake), so one
# parser, one catalog and one model download serve the installer and the client.
set(TONA_INSTALLER_CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../installer)
include(${TONA_INSTALLER_CORE_DIR}/core.cmake)
include(${TONA_INSTALLER_CORE_DIR}/setup.cmake)  # `tona setup` and FTXUI
set(TONA_CLIENT_SOURCES)
foreach(f text format server_view languages offer env control doctor settings ui keys keys_pages view view_util view_start
          view_states view_models view_agents view_settings view_modal screen history commands commands_more feed app
          app_actions term poller agent agent_claude agent_opencode agent_junie agent_pi agent_server agent_launch
          service_cmd foreground update_cmd manage_cmd)
    list(APPEND TONA_CLIENT_SOURCES client/${f}.cpp)
endforeach()
add_library(tona_client STATIC ${TONA_CLIENT_SOURCES})
target_include_directories(tona_client PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tona_client PUBLIC tona_install_core tona_update pthread)
add_executable(tona tools/tona.cpp)
target_link_libraries(tona PRIVATE tona_client tona_setup)
tona_unit_test(client_update SOURCES tests/test_client_update.cpp LIBS tona_client)
tona_unit_test(agent_commands SOURCES tests/test_agent_commands.cpp LIBS tona_client)
tona_unit_test(agent_launch SOURCES tests/test_agent_launch.cpp LIBS tona_client)
tona_unit_test(service_cmd SOURCES tests/test_service_cmd.cpp LIBS tona_client)
tona_unit_test(update_cmd SOURCES tests/test_update_cmd.cpp LIBS tona_client)
tona_unit_test(tona_main SOURCES tests/test_tona_main.cpp LIBS tona_client ARGS $<TARGET_FILE:tona>)
add_dependencies(test_tona_main tona)
tona_unit_test(client_text SOURCES tests/test_client_text.cpp LIBS tona_client)
tona_unit_test(client_state SOURCES tests/test_client_state.cpp LIBS tona_client
               ARGS ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/client)
tona_unit_test(client_views SOURCES tests/test_client_views.cpp LIBS tona_client
               ARGS ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/client)
tona_unit_test(client_commands SOURCES tests/test_client_commands.cpp LIBS tona_client
               ARGS ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/client)
tona_unit_test(client_center SOURCES tests/test_client_center.cpp LIBS tona_client)
tona_unit_test(client_stub SOURCES tests/test_client_stub.cpp LIBS tona_client
               ARGS ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/client)
# Fuzz entry (scripts/fuzz.sh builds the fuzzer with clang): an object target only, so the gates see the file
add_library(tona_fuzz_minisig OBJECT EXCLUDE_FROM_ALL tests/fuzz/fuzz_minisig.cpp)
target_link_libraries(tona_fuzz_minisig PRIVATE tona_update)
