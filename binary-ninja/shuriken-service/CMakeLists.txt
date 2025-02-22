CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

project(shuriken-service)

include_directories(./inc)

set(CMAKE_CXX_STANDARD 20)

add_definitions("-DSHURIKENSERVICE_EXPORTS")
add_library(${PROJECT_NAME} SHARED  src/shurikenservice.cpp
                                    src/shurikenview.cpp
                                    src/render.cpp)

target_link_libraries(${PROJECT_NAME} PUBLIC binaryninjaapi shuriken)


target_include_directories(${PROJECT_NAME} PUBLIC ./inc)

