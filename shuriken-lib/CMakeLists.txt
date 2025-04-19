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
# Test configuration
option(BUILD_TESTS "Build the Shuriken tests" ON)


# Configure CXX standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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


if (BUILD_TESTS)
    add_subdirectory(tests/)
endif ()

# Create shared library
# =====================
add_library(shuriken SHARED)
target_link_libraries(shuriken PUBLIC LIB_SHURIKEN)

# Installation
# ===========
include(GNUInstallDirs)

# Install the library targets
install(TARGETS shuriken LIB_SHURIKEN
        EXPORT shuriken-targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Install headers - conditionally based on DEV_MODE
if(DEV_MODE)
    # Install all headers including internal ones
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/shuriken
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            FILES_MATCHING PATTERN "*.hpp"
            PATTERN "*.h")
else()
    # Install only the SDK headers
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/shuriken/sdk
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/shuriken
            FILES_MATCHING PATTERN "*.hpp"
            PATTERN "*.h")
endif()

# Export the targets
install(EXPORT shuriken-targets
        FILE ShurikenTargets.cmake
        NAMESPACE Shuriken::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/shuriken)

# Create and install the config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ShurikenConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/ShurikenConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/shuriken
)

# Create and install the version file
write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/ShurikenConfigVersion.cmake
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
)

# Install the config and version files
install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/ShurikenConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/ShurikenConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/shuriken
)

# Add uninstall target
configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        @ONLY)

add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)