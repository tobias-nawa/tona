# The installer core (detection, catalog, config, steps, models), a library without a UI. Included by
# installer/CMakeLists.txt (the tests) and engine/CMakeLists.txt (the client tona uses the same config, catalog
# and model download, so there is one parser and one download path). The including project defines `tona_update` first
# finds OpenSSL and Python3 (Interpreter) and includes engine/offer/offer.cmake (tona_offer).
get_filename_component(TONA_INSTALLER_DIR ${CMAKE_CURRENT_LIST_DIR} ABSOLUTE)
get_filename_component(TONA_REPO_DIR ${CMAKE_CURRENT_LIST_DIR}/.. ABSOLUTE)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
# The model catalog and the expert profiles go into the program (signed with it), generated from
# toolchain/models.toml and toolchain/profiles/
file(GLOB TONA_PROFILES CONFIGURE_DEPENDS ${TONA_REPO_DIR}/toolchain/profiles/*.bin)
set(CATALOG_DATA ${CMAKE_CURRENT_BINARY_DIR}/catalog_data.cpp)
add_custom_command(OUTPUT ${CATALOG_DATA}
    COMMAND ${Python3_EXECUTABLE} -I ${TONA_REPO_DIR}/tools/gen_installer_data.py ${CATALOG_DATA}
    DEPENDS ${TONA_REPO_DIR}/tools/gen_installer_data.py
            ${TONA_REPO_DIR}/toolchain/models.toml ${TONA_PROFILES}
    COMMENT "Embedding the model catalog and the expert profiles")

# Core without a UI: detection, packages, recommendation, estimate, steps, the flow of the pages and the plan on the
# command line, backend modules
set(TONA_CORE_SOURCES)
foreach(f sys hw hw_ram speed disk pkgs plan config steps context_plan report hw_macos catalog model_fetch unit release_steps
         autostart flow space room describe step_text cli_args cli_plan start port progress shell_path)
    list(APPEND TONA_CORE_SOURCES ${TONA_INSTALLER_DIR}/core/${f}.cpp)
endforeach()
foreach(f registry hip_gfx1200 metal_apple)
    list(APPEND TONA_CORE_SOURCES ${TONA_INSTALLER_DIR}/modules/${f}.cpp)
endforeach()
add_library(tona_install_core STATIC ${TONA_CORE_SOURCES} ${CATALOG_DATA})
target_include_directories(tona_install_core PUBLIC ${TONA_INSTALLER_DIR})
target_include_directories(tona_install_core SYSTEM PUBLIC ${TONA_REPO_DIR}/engine/third_party)
target_link_libraries(tona_install_core PUBLIC tona_update tona_offer)
target_compile_options(tona_install_core PRIVATE -Wall -Wextra -Werror)

