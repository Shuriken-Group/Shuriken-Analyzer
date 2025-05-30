cmake_minimum_required(VERSION 3.5)

# Compiler detection and version-specific flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
        OUTPUT_VARIABLE GCC_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(GCC_VERSION VERSION_GREATER_EQUAL 14.0)
        message(STATUS "GCC ${GCC_VERSION} detected, adjusting compilation flags for dangling reference warnings")
        set(COMPILER_SPECIFIC_OPTIONS "-Wno-dangling-reference -Wno-error=dangling-reference")
        add_compile_options(${COMPILER_SPECIFIC_OPTIONS})
    endif()
endif()

include(FetchContent)

# By default, all libraries built here must be static
set(SHURIKEN_EXTERNAL_LIB_TYPE ${BUILD_SHARED_LIBS}) 
message("-- Preserve build shared libs to into SHURIKEN_EXTERNAL_LIB_TYPE, setting it to OFF")

set(BUILD_SHARED_LIBS OFF)
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

set(FETCHCONTENT_QUIET off)

# check if we have connection using PING
if(CMAKE_HOST_WIN32)
    execute_process(
            COMMAND ping www.google.com -n 2
            ERROR_QUIET
            RESULT_VARIABLE NO_CONNECTION
    )
else()
    execute_process(
            COMMAND ping www.google.com -c 2
            ERROR_QUIET
            RESULT_VARIABLE NO_CONNECTION
    )
endif()

if(NOT NO_CONNECTION EQUAL 0)
    set(FETCHCONTENT_FULLY_DISCONNECTED ON)
    message(WARNING "Fetch offline mode: requires already populated _deps")
else()
    set(FETCHCONTENT_FULLY_DISCONNECTED OFF)
    message(STATUS "Fetch will be used to download external dependencies")
endif()

if (GITHUB_ACTION OR (DEFINED ENV{GITHUB_ACTION} OR ENV{GITHUB_ACTION} EQUAL ON))
    message(WARNING "Running as a github action, download fetch the content")
    set(FETCHCONTENT_FULLY_DISCONNECTED OFF)
endif ()

# ------------------------------------
# Download the Logging utility from spdlog
# repo
# ------------------------------------
# Add this before the spdlog FetchContent_Declare
if(MSVC)
    add_compile_definitions(_SILENCE_STDEXT_ARR_ITERS_DEPRECATION_WARNING)
endif()

FetchContent_Declare(
    extern_spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.12.0
)

FetchContent_GetProperties(extern_spdlog)

if(NOT extern_spdlog_POPULATED)
    message(STATUS "Cloning spdlog")
    set(SPDLOG_BUILD_SHARED OFF)
    
    # First approach: Set compiler flags for spdlog
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND GCC_VERSION VERSION_GREATER_EQUAL 14.0)
        message(STATUS "Setting specific compiler flags for spdlog with GCC ${GCC_VERSION}")
        set(SPDLOG_WFLAGS "-Wno-dangling-reference -Wno-error=dangling-reference")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SPDLOG_WFLAGS}")
        
        # Modify CMAKE_CXX_FLAGS in the spdlog subdirectory
        set(PREV_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-dangling-reference -Wno-error=dangling-reference")
    endif()
    
    FetchContent_Populate(extern_spdlog)
    
    # Second approach: Patch the spdlog source for GCC 14
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND GCC_VERSION VERSION_GREATER_EQUAL 14.0)
        message(STATUS "Additionally patching spdlog source for GCC ${GCC_VERSION} compatibility")
        
        # Path to the file with the issue
        set(CORE_H_PATH "${extern_spdlog_SOURCE_DIR}/include/spdlog/fmt/bundled/core.h")
        
        # Read the file content
        file(READ "${CORE_H_PATH}" CONTENT)
        
        # Check if the problematic line exists (to avoid patching multiple times)
        if(CONTENT MATCHES "const auto& arg = arg_mapper<Context>\\(\\)\\.map\\(FMT_FORWARD\\(val\\)\\);")
            # Replace the problematic line
            string(REPLACE 
                "const auto& arg = arg_mapper<Context>().map(FMT_FORWARD(val));" 
                "auto arg_copy = arg_mapper<Context>().map(FMT_FORWARD(val)); const auto& arg = arg_copy;" 
                FIXED_CONTENT "${CONTENT}")
            
            # Write the modified content back
            file(WRITE "${CORE_H_PATH}" "${FIXED_CONTENT}")
            
            message(STATUS "spdlog source patched successfully")
        else()
            message(STATUS "spdlog source already patched or line not found")
        endif()
    endif()
    
    # Add this before adding the subdirectory
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND GCC_VERSION VERSION_GREATER_EQUAL 14.0)
        # Create a custom CMake module to inject into spdlog's build
        file(WRITE "${extern_spdlog_BINARY_DIR}/gcc14_compatibility.cmake" 
            "set(CMAKE_CXX_FLAGS \"\${CMAKE_CXX_FLAGS} -Wno-dangling-reference -Wno-error=dangling-reference\")\n"
            "message(STATUS \"Applied GCC 14 compatibility flags in spdlog build\")\n"
        )
        
        # Prepend our module to CMAKE_MODULE_PATH
        set(CMAKE_MODULE_PATH "${extern_spdlog_BINARY_DIR};${CMAKE_MODULE_PATH}")
    endif()
    
    # Enable trace logging to see what happens during build
    set(CMAKE_MESSAGE_LOG_LEVEL DEBUG)
    
    add_subdirectory(
        ${extern_spdlog_SOURCE_DIR}
        ${extern_spdlog_BINARY_DIR}
    )
    
    # Apply compile options directly to the spdlog target after it's created
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND GCC_VERSION VERSION_GREATER_EQUAL 14.0)
        message(STATUS "Applying warning flags directly to spdlog target")
        target_compile_options(spdlog PRIVATE -Wno-dangling-reference -Wno-error=dangling-reference)
        
        # Also add to interface options to propagate to consumers
        get_target_property(SPDLOG_INTERFACE_COMPILE_OPTIONS spdlog INTERFACE_COMPILE_OPTIONS)
        if(NOT SPDLOG_INTERFACE_COMPILE_OPTIONS)
            set(SPDLOG_INTERFACE_COMPILE_OPTIONS "")
        endif()
        set_target_properties(spdlog PROPERTIES 
            INTERFACE_COMPILE_OPTIONS "${SPDLOG_INTERFACE_COMPILE_OPTIONS};-Wno-dangling-reference;-Wno-error=dangling-reference")
        
        # Restore original flags if needed
        if(DEFINED PREV_CMAKE_CXX_FLAGS)
            set(CMAKE_CXX_FLAGS ${PREV_CMAKE_CXX_FLAGS})
        endif()
    endif()
endif()

set_target_properties(spdlog
    PROPERTIES
    POSITION_INDEPENDENT_CODE 1
)

# Make the project available
FetchContent_MakeAvailable(extern_spdlog)

# ------------------------------------
# Download the printing utility fmt
# ------------------------------------
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt
    GIT_TAG 10.1.1
)

FetchContent_GetProperties(fmt)

if(NOT fmt_POPULATED)
    message(STATUS "Cloning fmt")
    FetchContent_Populate(fmt)
    add_subdirectory(
        ${fmt_SOURCE_DIR}
        ${fmt_BINARY_DIR}
    )
endif()

# Make the project available
FetchContent_MakeAvailable(fmt)

# ------------------------------------
# Download the zlib library
# ------------------------------------
FetchContent_Declare(
    zlib
    GIT_REPOSITORY https://github.com/madler/zlib
    GIT_TAG v1.3.1
)

FetchContent_GetProperties(zlib)

if(NOT zlib_POPULATED)
    message(STATUS "Cloning zlib")
    FetchContent_Populate(zlib)
    add_subdirectory(
        ${zlib_SOURCE_DIR}
        ${zlib_BINARY_DIR}
    )
    set_target_properties(zlibstatic PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# Make the project available
FetchContent_MakeAvailable(zlib)
add_library(ZLIB::ZLIB ALIAS zlibstatic)
set(ZLIB_LIBRARY zlibstatic)
set(ZLIB_INCLUDE_DIR ${zlib_SOURCE_DIR} ${zlib_BINARY_DIR})

# ------------------------------------
# Download the zip libary
# ------------------------------------
FetchContent_Declare(
    libzip
    GIT_REPOSITORY https://github.com/nih-at/libzip.git
    GIT_TAG        main  # Or a specific version tag
)

FetchContent_GetProperties(libzip)

if(NOT libzip_POPULATED)
    message(STATUS "Cloning libzip")
    set(BUILD_SHARED_LIBS ON)
    option(BUILD_SHARED_LIBS ON)
    set(ZLIB_LIBRARY ZLIB::ZLIB)
    set(ZLIB_INCLUDE_DIR ${zlib_SOURCE_DIR})
    FetchContent_Populate(libzip)
    add_subdirectory(
        ${libzip_SOURCE_DIR}
        ${libzip_BINARY_DIR}
    )
    set(BUILD_SHARED_LIBS OFF)
    option(BUILD_SHARED_LIBS OFF)
endif()

FetchContent_MakeAvailable(libzip)

function(check_static_lib lib_name)
    get_target_property(external_target_type ${lib_name} TYPE)
    if (NOT external_target_type STREQUAL STATIC_LIBRARY)
        message(FATAL_ERROR "${lib_name} is not a static library")
    endif()
endfunction()

function(check_position_independent_code lib_name)
    get_target_property(external_target_type ${lib_name} POSITION_INDEPDENT_CODE)
    if (external_target_type EQUAL False)
        message(FATAL_ERROR "${lib_name} is not a library built with POSITION_INDEPENDENT_CODE option")
    endif()
endfunction()

# check_static_lib(zip)
# check_position_independent_code(zip)
check_static_lib(fmt)
check_static_lib(spdlog)

message("-- Reversing build shared libs to SHURIKEN_EXTERNAL_LIB_TYPE, setting it to ${SHURIKEN_EXTERNAL_LIB_TYPE}")

set(BUILD_SHARED_LIBS ${SHURIKEN_EXTERNAL_LIB_TYPE})
option(BUILD_SHARED_LIBS "Build shared libraries" ${SHURIKEN_EXTERNAL_LIB_TYPE})