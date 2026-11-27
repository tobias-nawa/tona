# The offer library (tona_offer): tasks, recommendations and model cards, generated from data/modeldb/offer.json and
# toolchain/models.toml. Included by engine/CMakeLists.txt and installer/CMakeLists.txt, so tona setup and the client
# link the same code and show the same numbers. The including project needs Python 3 (find_package(Python3 REQUIRED
# COMPONENTS Interpreter)) and defines tona_offer once.
set(TONA_OFFER_DIR ${CMAKE_CURRENT_LIST_DIR})
get_filename_component(TONA_OFFER_REPO ${CMAKE_CURRENT_LIST_DIR}/../.. ABSOLUTE)
set(TONA_OFFER_DATA ${CMAKE_CURRENT_BINARY_DIR}/offer_data.cpp)
add_custom_command(OUTPUT ${TONA_OFFER_DATA}
    COMMAND ${Python3_EXECUTABLE} -I ${TONA_OFFER_REPO}/tools/gen_offer_data.py ${TONA_OFFER_DATA}
    DEPENDS ${TONA_OFFER_REPO}/tools/gen_offer_data.py ${TONA_OFFER_REPO}/data/modeldb/offer.json
            ${TONA_OFFER_REPO}/toolchain/models.toml
    COMMENT "Embedding the offer data")
add_library(tona_offer STATIC ${TONA_OFFER_DIR}/offer.cpp ${TONA_OFFER_DATA})
target_include_directories(tona_offer PUBLIC ${TONA_OFFER_DIR}/..)
