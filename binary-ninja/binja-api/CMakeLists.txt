CMAKE_MINIMUM_REQUIRED(VERSION 3.14)

project(binja-api)

include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH ${BN_QT})

# --Binary Ninja API Setup-------------------------------------------
set (BINJA-API-STABLE-3.5 "59e569906828e91e4884670c2bba448702f5a31d")
set (BINJA-API-STABLE-4.0.4958 "b7c8e9bfbae0eec8d7be47cd42f61c14542cc28c")
set (BINJA-API-STABLE-4.1.5747 "ab5c0b34b473fa6cdc5ad55635e670e27688d71c")
set (BINJA-API-STABLE-4.1.5902 "967ffc9bd705f6820c4dbfe97f301d52facb1417")
set (BINJA-API-STABLE-4.2.6455 "9229ebde590febc9635d824ae9284ae170dee9da")
set (BINJA_API_VERSION ${BINJA-API-STABLE-4.2.6455})
message(STATUS "Using Binary-Ninja API: ${BINJA_API-VERSION}")

FetchContent_Declare(
    binja-api
    GIT_REPOSITORY https://github.com/Vector35/binaryninja-api.git
    GIT_TAG ${BINJA_API_VERSION}
)

if(MSVC)
    add_compile_options(/wd4005 /wd4996 /wd4244)
endif()

set(HEADLESS 1 CACHE INTERNAL "Enable headless mode")

FetchContent_MakeAvailable(binja-api)
