cmake_minimum_required(VERSION 3.16)
# Shuriken-dump project using shuriken-lib
project(
        Shuriken-dump
        LANGUAGES CXX
        VERSION 2.0
        DESCRIPTION "Simple command line tool to dump information from analyzed files"
        HOMEPAGE_URL https://github.com/Shuriken-Group/Shuriken-Analyzer/
)

# Configure CXX standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Use FetchContent to automatically download fmt if not found
include(FetchContent)

# Try to find fmt first
find_package(fmt QUIET)

if(NOT fmt_FOUND)
    message(STATUS "fmt not found, downloading...")
    FetchContent_Declare(
            fmt
            GIT_REPOSITORY https://github.com/fmtlib/fmt.git
            GIT_TAG 10.2.1  # Use a specific version
    )
    FetchContent_MakeAvailable(fmt)
endif()

# Create the executable target from the single source file
add_executable(shuriken-dump shuriken-dump.cpp)

# Include headers from shuriken-lib for development builds
target_include_directories(shuriken-dump
        PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/../shuriken-lib/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../shuriken-lib/include/shuriken/sdk"
)

# Link libraries
target_link_libraries(shuriken-dump
        PRIVATE
        shuriken
        fmt
)

# Set installation paths
include(GNUInstallDirs)
set(binary_install_path ${CMAKE_INSTALL_BINDIR})
set(include_install_path ${CMAKE_INSTALL_INCLUDEDIR})

# Install the executable
install(TARGETS shuriken-dump
        RUNTIME DESTINATION ${binary_install_path}
)