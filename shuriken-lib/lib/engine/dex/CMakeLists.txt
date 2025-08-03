target_sources(LIB_SHURIKEN PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/dex_engine.cpp
)

add_subdirectory(parser)
add_subdirectory(disassembler)
add_subdirectory(analysis)