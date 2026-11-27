# OpenMP on macOS (docs/use/macos.md): Apple clang understands -fopenmp only behind -Xpreprocessor and ships no runtime, so
# FindOpenMP needs the flags and Homebrew's libomp spelled out. HOMEBREW_PREFIX covers both Apple silicon
# (/opt/homebrew) and Intel (/usr/local) installs.
if(DEFINED ENV{HOMEBREW_PREFIX})
    set(TONA_LIBOMP $ENV{HOMEBREW_PREFIX}/opt/libomp)
else()
    set(TONA_LIBOMP /opt/homebrew/opt/libomp)
endif()
if(NOT EXISTS ${TONA_LIBOMP}/lib/libomp.dylib)
    message(FATAL_ERROR "libomp missing at ${TONA_LIBOMP}: brew install libomp (docs/use/macos.md)")
endif()
set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${TONA_LIBOMP}/include" CACHE STRING "")
set(OpenMP_CXX_LIB_NAMES omp CACHE STRING "")
set(OpenMP_omp_LIBRARY ${TONA_LIBOMP}/lib/libomp.dylib CACHE FILEPATH "")
