#------------------------------------------------------------- Installation Flags
# Default paths for installation
# Set CMake MacOS runtime path to find .so files on mac
set(CMAKE_MACOSX_RPATH 1)
# Append runtime path to find .so files on unix, needed before install
list( APPEND CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib" )

# Check for operating system and set install paths accordingly
# Conditional installation paths for different platforms
if(APPLE)
    find_program(HOMEBREW_FOUND brew)
    if(HOMEBREW_FOUND)
        execute_process(COMMAND brew --prefix
                OUTPUT_VARIABLE HOMEBREW_PREFIX
                 OUTPUT_STRIP_TRAILING_WHITESPACE)
        set(include_install_path "${HOMEBREW_PREFIX}/include")
        set(library_install_path "${HOMEBREW_PREFIX}/lib")
        set(CMAKE_INSTALL_PREFIX "${HOMEBREW_PREFIX}")
    endif()
elseif(UNIX AND NOT APPLE) # Explicitly differentiate UNIX from APPLE
    # Linux specific paths (already set as default)
    if(NOT DEFINED CMAKE_INSTALL_PREFIX)
        set(CMAKE_INSTALL_PREFIX "/usr/local" CACHE PATH "Default installation directory" FORCE)
    endif()
    # Define default include and library paths based on CMAKE_INSTALL_PREFIX
    set(include_install_path "${CMAKE_INSTALL_PREFIX}/include") # Default path
    set(library_install_path "${CMAKE_INSTALL_PREFIX}/lib") # Default path
elseif(WIN32)
    # Windows specific paths - use relative paths to work with install-time prefix
    set(include_install_path "include")
    set(library_install_path "lib")
endif()

#------------------------------------------------------ Uninstallation Flags
# Define the uninstall target
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_SOURCE_DIR}/cmake/cmake_install_uninstall.cmake.in"
        "${CMAKE_BINARY_DIR}/cmake/cmake_install_uninstall.cmake"
        IMMEDIATE @ONLY
    )

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_BINARY_DIR}/cmake/cmake_install_uninstall.cmake
    )
endif()