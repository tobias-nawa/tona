# Writes `git describe` of the source tree to a stamp file next to tona-server, when tona-server is (re)linked.
# scripts/ab_plan.py reads it, so a measurement names the commit that was built, not the HEAD at run time.
# Run as: cmake -DSRC=<source dir> -DOUT=<stamp file> -P commit_stamp.cmake
execute_process(
    COMMAND git -C "${SRC}" describe --always --dirty --abbrev=10
    OUTPUT_VARIABLE described
    ERROR_QUIET
    RESULT_VARIABLE rc
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT rc EQUAL 0 OR described STREQUAL "")
    set(described "unknown")  # no git or no repository (a source archive)
endif()
file(WRITE "${OUT}" "${described}\n")
