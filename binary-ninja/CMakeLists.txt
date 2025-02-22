CMAKE_MINIMUM_REQUIRED(VERSION 3.20)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/binjaplugin-build")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/binjaplugin-build")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/binjaplugin-build")

add_subdirectory(binja-api)
add_subdirectory(view-dex)
add_subdirectory(shuriken-service)