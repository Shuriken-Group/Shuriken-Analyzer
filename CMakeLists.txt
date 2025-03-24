cmake_minimum_required(VERSION 3.29)
project(
        ShurikenProject
        DESCRIPTION "Shuriken Analyzer project for analysis of file formats"
        HOMEPAGE_URL https://github.com/Shuriken-Group/Shuriken-Analyzer/
)

set(CMAKE_CXX_STANDARD 20)

enable_testing()
add_subdirectory(shuriken-lib/)