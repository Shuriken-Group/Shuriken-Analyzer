cmake_minimum_required(VERSION 3.5)
# Information from Shuriken project
project(
        ShurikenLib
        LANGUAGES CXX
        VERSION 2.0
        DESCRIPTION "Library for the analysis of bytecodes"
        HOMEPAGE_URL https://github.com/Shuriken-Group/Shuriken-Analyzer/
)

# Global variables, to specify one variable, add a proper comment specifying why
# the variable is needed and how it must be used.

# DEV_MODE: by default, only the sdk folder from the `include/shuriken` will be
# exported and installed with the library, in case the user wants to dig into
# the shuriken internals and use it in any program, the DEV_MODE must be set to
# ON
option(DEV_MODE "Enable the developer mode adding the internal headers to the installation" OFF)


# SHURIKEN Source Definition
# ==========================
add_library(LIB_SHURIKEN
    OBJECT
)

set_target_properties(LIB_SHURIKEN
    PROPERTIES
    POSITION_INDEPENDENT_CODE 1
)

add_subdirectory(lib/)

# SHURIKEN includes
# =================

set(SHURIKEN_PUBLIC_INCLUDE_DIR
        ${CMAKE_CURRENT_SOURCE_DIR}/include/shuriken/sdk/
)

set(SHURIKEN_PRIVATE_INCLUDE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/src/
)

target_include_directories(LIB_SHURIKEN
        PUBLIC
            "$<BUILD_INTERFACE:${SHURIKEN_PUBLIC_INCLUDE_DIR}>"
            "$<INSTALL_INTERFACE:include/shuriken/sdk>"
        PRIVATE "${SHURIKEN_PRIVATE_INCLUDE_DIR}")

# SHURIKEN testing
# ================

add_subdirectory(tests/)