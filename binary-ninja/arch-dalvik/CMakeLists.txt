CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

project(arch-dalvik)

include_directories(./inc)

set(CMAKE_CXX_STANDARD 20)

add_library(${PROJECT_NAME} SHARED  src/plugin.cpp
                                    src/arch-dalvik.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE binaryninjaapi shuriken-service)


target_include_directories(${PROJECT_NAME} PUBLIC ./inc)

