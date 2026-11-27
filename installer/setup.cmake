# The guided setup of `tona setup`: the pages (FTXUI), the commands without a screen and the entry point run_setup().
# Included by engine/CMakeLists.txt (the program tona) and installer/CMakeLists.txt (the tests), after core.cmake. FTXUI
# is pinned through FetchContent (v7.0.3).
include(FetchContent)
FetchContent_Declare(ftxui
    GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI.git
    GIT_TAG f921fad208912747c17d129a8ef75ec7624b6eec  # v7.0.3
    SYSTEM)
set(FTXUI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(FTXUI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FTXUI_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(ftxui)
# The engine project compiles with -Werror for all targets; the pinned third-party code is not ours to fix
foreach(t screen dom component)
    if(TARGET ${t})
        target_compile_options(${t} PRIVATE -Wno-error)
    endif()
endforeach()

add_library(tona_setup STATIC
    ${TONA_INSTALLER_DIR}/setup.cpp ${TONA_INSTALLER_DIR}/headless.cpp ${TONA_INSTALLER_DIR}/tui.cpp
    ${TONA_INSTALLER_DIR}/tui_pages.cpp ${TONA_INSTALLER_DIR}/tui_room.cpp ${TONA_INSTALLER_DIR}/tui_states.cpp
    ${TONA_INSTALLER_DIR}/tui_install.cpp ${TONA_INSTALLER_DIR}/tui_finish.cpp ${TONA_INSTALLER_DIR}/tui_dump.cpp ${TONA_INSTALLER_DIR}/tui_widgets.cpp)
target_include_directories(tona_setup PUBLIC ${TONA_INSTALLER_DIR})
target_link_libraries(tona_setup PUBLIC tona_install_core ftxui::screen ftxui::dom ftxui::component)
target_compile_options(tona_setup PRIVATE -Wall -Wextra -Werror)
