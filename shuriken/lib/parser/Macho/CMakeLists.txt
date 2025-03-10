target_sources(macho-obj PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/parser.cpp
        ${CMAKE_CURRENT_LIST_DIR}/macho_header.cpp
        ${CMAKE_CURRENT_LIST_DIR}/macho_commands.cpp
        ${CMAKE_CURRENT_LIST_DIR}/macho_sections.cpp
)