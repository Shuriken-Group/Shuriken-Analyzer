cmake_minimum_required(VERSION 3.5)
project(
        ShurikenProject
        DESCRIPTION "Shuriken Analyzer project for analysis of file formats"
        HOMEPAGE_URL https://github.com/Shuriken-Group/Shuriken-Analyzer/
)

# Set C++20 as the standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()
add_subdirectory(shuriken-lib/)