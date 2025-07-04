cmake_minimum_required(VERSION 3.5)
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

# Find required packages
find_package(fmt REQUIRED)

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
        fmt::fmt
)

# Set installation paths
include(GNUInstallDirs)
set(binary_install_path ${CMAKE_INSTALL_BINDIR})
set(include_install_path ${CMAKE_INSTALL_INCLUDEDIR})

# Install the executable
install(TARGETS shuriken-dump
        RUNTIME DESTINATION ${binary_install_path}
)