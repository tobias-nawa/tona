# OpenMP for the host code of sources compiled as HIP language.
#
# find_package(OpenMP) adds -fopenmp for C++ sources only. A source that is compiled as HIP (set_source_files_properties
# LANGUAGE HIP, or a .hip file) got no flag, so its `#pragma omp` compiled to nothing and the loop ran on one thread
# (the parallel verification scan of 2026-11-18 showed it: research/measurements/2026-11-18-b1-host-parallel.md).
# tona_openmp_for_pragmas(<target>) gives -fopenmp to every HIP source of the target that contains a pragma; the flag
# is per source so device code of the other files stays as it was. A source with a pragma that cannot get the flag
# (not in a target that calls this function) is caught by tools/gates/openmp.py on the compile database.
function(tona_openmp_for_pragmas target)
    get_target_property(srcs ${target} SOURCES)
    foreach(src IN LISTS srcs)
        get_source_file_property(lang ${src} LANGUAGE)
        if(NOT lang STREQUAL "HIP" AND NOT src MATCHES "\\.hip$")
            continue()
        endif()
        set(path "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
        if(NOT EXISTS "${path}")
            continue()
        endif()
        file(STRINGS "${path}" pragma LIMIT_COUNT 1 REGEX "^[ \t]*#[ \t]*pragma[ \t]+omp")
        if(pragma)
            set_property(SOURCE ${src} APPEND PROPERTY COMPILE_OPTIONS -fopenmp)
        endif()
    endforeach()
endfunction()
