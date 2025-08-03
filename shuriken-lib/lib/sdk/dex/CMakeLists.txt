target_sources(LIB_SHURIKEN PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/dvm_types.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dvm_prototypes.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/field.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/method.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/class.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/instructions.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/field_id.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/method_id.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/external_field.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/external_method.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/basic_blocks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/control_flow_graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dex.cpp
)