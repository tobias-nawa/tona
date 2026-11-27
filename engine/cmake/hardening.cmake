# Hardening of the host code of the release programs (tona-server, tona). Included by
# engine/CMakeLists.txt (TONA_RELEASE or TONA_HARDEN) and installer/CMakeLists.txt (TONA_HARDEN); without the option
# no compiler or linker command changes, so the reference development build stays as it is (AGENTS.md rule 1).
# tools/check_hardening.py proves on the finished programs that every property below is there.
#
#   _FORTIFY_SOURCE=3   libc calls with a known destination size abort on overflow (needs -O, so not in Debug)
#   stack protector     canary in every function with an array or an address-taken local
#   stack clash         large frames are probed page by page: no jump over the guard page
#   CET / BTI           indirect-branch tracking and shadow stack marking (x86-64: -fcf-protection, arm64: -mbranch-protection)
#   PIE, full RELRO     ASLR for the program image, GOT read-only after start (-z now), no executable stack
#
# Device code (the HIP offload pass for gfx1200) cannot take these options (the AMDGPU target rejects -fcf-protection and
# has no stack protector), so HIP translation units get them through -Xarch_host, which clang applies to the host pass only.
include_guard(GLOBAL)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "hardening: only on Linux (ELF), skipped on ${CMAKE_SYSTEM_NAME}")
    return()
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(TONA_HARDEN_COMPILE -fstack-protector-strong -fstack-clash-protection)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
    list(APPEND TONA_HARDEN_COMPILE -mbranch-protection=standard)
else()
    list(APPEND TONA_HARDEN_COMPILE -fcf-protection=full)
endif()
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    # -U first: a distribution compiler may define a level already, and a redefinition is a warning (-Werror here)
    list(APPEND TONA_HARDEN_COMPILE -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=3)
endif()

foreach(flag IN LISTS TONA_HARDEN_COMPILE)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${flag}>" "$<$<COMPILE_LANGUAGE:HIP>:SHELL:-Xarch_host ${flag}>")
endforeach()
add_link_options(-pie LINKER:-z,relro LINKER:-z,now LINKER:-z,noexecstack)
# The ROCm clang links its startup objects (clang_rt.crtbegin/crtend) from compiler-rt for a HIP link; they carry no GNU
# property note, so the linker drops IBT and SHSTK from the whole program (`ld -z cet-report=warning` names exactly
# these two files). GCC's runtime objects carry the note, and libgcc has the builtins the HIP package asked compiler-rt
# for (__fp16, _Float16). tona_harden_hip_runtime() runs after enable_language(HIP), when hip-lang::device exists.
function(tona_harden_hip_runtime)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$" OR NOT TARGET hip-lang::device)
        return()
    endif()
    get_target_property(opts hip-lang::device INTERFACE_LINK_OPTIONS)
    string(REPLACE "--rtlib=compiler-rt" "--rtlib=libgcc" opts "${opts}")
    set_property(TARGET hip-lang::device PROPERTY INTERFACE_LINK_OPTIONS "${opts}")
endfunction()
