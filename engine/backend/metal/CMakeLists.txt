# Metal backend (Apple silicon), included by engine/CMakeLists.txt with -DTONA_BACKEND_METAL=ON (tier 3, off by
# default). One static library; tona-server and tona-check link it. The kernels are compiled at build time into
# tona.metallib next to the programs (xcrun metal; needs `xcodebuild -downloadComponent MetalToolchain`).
enable_language(OBJCXX)
add_library(tona_backend_metal STATIC arena.mm arena_cold.mm arena_govern.mm arena_purge.mm backend.mm device.mm keepalive.mm mtp_vocab.mm pass.mm pass_layers.mm pass_mtp.mm profile.mm runner_cold.mm
    runner.mm runner_mtp.mm seq.mm weights.mm)
target_include_directories(tona_backend_metal PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_options(tona_backend_metal PRIVATE -fobjc-arc)
target_link_libraries(tona_backend_metal PUBLIC tona_model "-framework Metal" "-framework Foundation")
target_compile_definitions(tona_backend_metal INTERFACE TONA_HAVE_METAL=1)

set(TONA_METAL_KERNELS all.metal params.h decode.metal matvec.metal matvec_mma.metal gemm.metal moe_table.metal ops.metal deltanet.metal qsa.metal qsa_prefill.metal moe.metal
    ple.metal mtp_pick.metal)
list(TRANSFORM TONA_METAL_KERNELS PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/kernels/)
set(TONA_METAL_GEN ${CMAKE_CURRENT_BINARY_DIR}/metal-gen)
add_custom_command(
    OUTPUT ${TONA_METAL_GEN}/quant_tables.metal
    COMMAND ${CMAKE_COMMAND} -DIN=${PROJECT_SOURCE_DIR}/model/quant_tables.h -DOUT=${TONA_METAL_GEN}/quant_tables.metal
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tables.cmake
    DEPENDS ${PROJECT_SOURCE_DIR}/model/quant_tables.h ${CMAKE_CURRENT_SOURCE_DIR}/tables.cmake)
# Without fast math, so that every pass width is bit-identical (kernels/all.metal); fast math only as a measurement arm
option(TONA_METAL_FAST_MATH "Compile the Metal kernels with fast math (measurements only; breaks width bit-identity)" OFF)
if(TONA_METAL_FAST_MATH)
    set(TONA_METAL_MATH -ffast-math)
else()
    set(TONA_METAL_MATH -fno-fast-math)
endif()
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/tona.metallib
    COMMAND xcrun metal -std=metal3.2 -O2 ${TONA_METAL_MATH} -I ${TONA_METAL_GEN} -I ${CMAKE_CURRENT_SOURCE_DIR}/kernels
            ${CMAKE_CURRENT_SOURCE_DIR}/kernels/all.metal -o ${CMAKE_BINARY_DIR}/tona.metallib
    DEPENDS ${TONA_METAL_KERNELS} ${TONA_METAL_GEN}/quant_tables.metal)
add_custom_target(tona_metallib ALL DEPENDS ${CMAKE_BINARY_DIR}/tona.metallib)
add_dependencies(tona_backend_metal tona_metallib)

# Host parts without a GPU (RoPE table, n-gram rows, indexer cells)
tona_unit_test(metal_host SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_metal_host.cpp LIBS tona_model)
tona_unit_test(gpu_gaps SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_gpu_gaps.cpp LIBS tona_model)
tona_unit_test(keepalive SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_keepalive.mm
    ${CMAKE_CURRENT_SOURCE_DIR}/keepalive.mm LIBS tona_model "-framework Metal" "-framework Foundation")
target_compile_options(test_keepalive PRIVATE -fobjc-arc)
