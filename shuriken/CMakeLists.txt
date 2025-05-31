cmake_minimum_required(VERSION 3.5)

# Information from Shuriken project
project(
        ShurikenLib
        LANGUAGES CXX
        VERSION 0.0.1
        DESCRIPTION "Library for the analysis of bytecodes"
        HOMEPAGE_URL https://github.com/Fare9/Shuriken-Analyzer/
)

add_definitions(-DSHURIKENLIB_EXPORTS)
##############################################################
# C++ Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# enable compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        add_compile_options(-Wall -Wextra -Werror -Wno-unused-parameter)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
                add_compile_options(-Wno-zero-length-array)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
                add_compile_options(-Wno-strict-flex-arrays -Wno-multichar)
        endif()
elseif(MSVC)
        add_compile_options(/W4 /WX /wd4200 /wd4458 /wd4456 /wd4297)
        add_compile_definitions(_SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING)
endif()

# Compiler for the project
if(CMAKE_CXX_COMPILER_LOADED)
        message(STATUS "C++ compiler ID: ${CMAKE_CXX_COMPILER_ID}")
        message(STATUS "C++ compiler version: ${CMAKE_CXX_COMPILER_VERSION}")
endif()

# Build type
if(NOT CMAKE_BUILD_TYPE)
        message(STATUS "Build type not set, setting to Release")
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

# for debug configuration set specific flags
if(NOT WIN32)
        set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O0 -DNDEBUG")
endif()
##############################################################


##############################################################
# options for compilation, Shuriken will be intended to provide the user
# with the options to compile only part of the project
option(DEX "Disable the DEX analysis from the library (enable by default)" ON)
option(DEX_TESTS "Compile all the tests for DEX files" OFF)
option(SANITIZE "Enable sanitizers for the project" OFF)
option(MJOLNIR "Enable the compilation of MjolnIR, it needs LLVM and MLIR on the system" OFF)
option(MACHO "Enable the Mach-O analysis from the library" OFF)
option(MACHO_TESTS "Compile all the tests for Mach-O files" OFF)

##############################################################


if((CMAKE_BUILD_TYPE STREQUAL "Debug") AND SANITIZE)
        message(STATUS "Debug build")
        message(STATUS "Sanitizers enabled by debug mode")
        # Set up sanitizer for address and undefined behavior
        set(SANITIZER_FLAGS "-fsanitize=address,undefined")
        message(STATUS "Sanitizer flags: ${SANITIZER_FLAGS}")
        add_compile_options(${SANITIZER_FLAGS} -fno-omit-frame-pointer)
        add_link_options(${SANITIZER_FLAGS} -fno-omit-frame-pointer)
endif()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

# Shuriken will contain only the basic classes, nothing more
# the analyses for the different files are included optionally
# we will include all the content in an object file
message(STATUS "Generating Shuriken object files")

add_library(shuriken-obj
        OBJECT
)

set_target_properties(shuriken-obj
        PROPERTIES
        POSITION_INDEPENDENT_CODE 1
)

# subdirectories for shuriken
add_subdirectory(lib/common/)

target_link_libraries(shuriken-obj PRIVATE spdlog::spdlog)
target_link_libraries(shuriken-obj PRIVATE zip)

# more information of object files
list(APPEND LIB_MODULES
        $<TARGET_OBJECTS:shuriken-obj>
)

# Compile the object files for DEX
if(DEX)
        message(STATUS "Generating Shuriken DEX object files")

        add_library(dex-obj
                OBJECT
        )

        set_target_properties(dex-obj
                PROPERTIES
                POSITION_INDEPENDENT_CODE 1
        )

        # Dex files
        add_subdirectory(lib/parser/Dex)
        add_subdirectory(lib/disassembler/Dex)
        add_subdirectory(lib/analysis/Dex)
        add_subdirectory(lib/api/Dex)
        # APK files
        add_subdirectory(lib/parser/Apk)

        target_link_libraries(dex-obj PRIVATE spdlog::spdlog)
        target_link_libraries(dex-obj PRIVATE zip)

        target_include_directories(dex-obj
                BEFORE
                PUBLIC ${SHURIKEN_INCLUDE_PATH}
        )

        list(APPEND LIB_MODULES
                $<TARGET_OBJECTS:dex-obj>
        )
endif()

# Compile the object files for Mach-O
if (MACHO)
    message(STATUS "Generating Shuriken Mach-O object files")

    add_library(macho-obj
            OBJECT
    )

    set_target_properties(macho-obj
            PROPERTIES
            POSITION_INDEPENDENT_CODE 1
    )

    # Mach-O files
    add_subdirectory(lib/parser/Macho)
    # IPA files
    add_subdirectory(lib/parser/Ipa)

    target_link_libraries(macho-obj PRIVATE spdlog::spdlog)
    target_link_libraries(macho-obj PRIVATE zip)

    target_include_directories(macho-obj
            BEFORE
            PUBLIC ${SHURIKEN_INCLUDE_PATH}
    )

    list(APPEND LIB_MODULES
            $<TARGET_OBJECTS:macho-obj>
    )
endif ()

message(STATUS "Generating Shuriken shared library")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
add_library(shuriken SHARED
        ${LIB_MODULES}
)

message(STATUS "Linking libraries into Shuriken library")

target_link_libraries(shuriken PRIVATE spdlog::spdlog)
target_link_libraries(shuriken PRIVATE zip)

target_include_directories(shuriken
        PUBLIC
        "${CMAKE_SOURCE_DIR}/shuriken/include/"
)

# MJOLNIR 
#############################################################

if(MJOLNIR)
        add_subdirectory(mjolnir)
endif(MJOLNIR)




#############################################################
include(CTest)

if(DEX_TESTS)
        message(STATUS "Test option used but not compilers for DEX provided, using precompiled DEX")
        set(DEX_FILES_FOLDER
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/compiled/dex_files/")
        set(APK_FILES_FOLDER "${CMAKE_CURRENT_SOURCE_DIR}/tests/compiled/apk_files/")
endif()

message(STATUS "Path to DEX Folder: ${DEX_FILES_FOLDER}")

if (MACHO_TESTS)
    message(STATUS "Using precompiled Mach-O")
    set(MACHO_FILES_FOLDER
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/compiled/macho_files/")
    set(IPA_FILES_FOLDER 
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/compiled/ipa_files/")
endif ()

message(STATUS "Path to Mach-O Folder: ${MACHO_FILES_FOLDER}")

add_subdirectory(tests)

# Assuming this variable points to your project's include directory
set(SHURIKEN_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Ensure install paths are properly set
set_target_properties(shuriken PROPERTIES
        INSTALL_NAME_DIR "${library_install_path}"
        BUILD_WITH_INSTALL_RPATH TRUE
)

# Install the Shuriken library's binary
install(TARGETS shuriken
        LIBRARY DESTINATION ${library_install_path}
)

# Install the Shuriken library's header files and def files
install(DIRECTORY ${SHURIKEN_INCLUDE_PATH}/
        DESTINATION ${include_install_path}
        FILES_MATCHING 
        PATTERN "*.h*"
        PATTERN "*.def"
)