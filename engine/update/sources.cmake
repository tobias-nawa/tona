# Sources of the updater library. Included by engine/CMakeLists.txt and installer/CMakeLists.txt (tona setup does the
# first installation with the same code, so there is one signature check and one download path).
set(TONA_UPDATE_SOURCES)
foreach(f crypto minisig manifest keys url config fetch fsutil state release activate install switch script_service
        updater)
    list(APPEND TONA_UPDATE_SOURCES ${CMAKE_CURRENT_LIST_DIR}/${f}.cpp)
endforeach()
