if(MACHO AND MACHO_TESTS)
    configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/macho-files-folder.in
            ${CMAKE_CURRENT_SOURCE_DIR}/macho-files-folder.inc
    )

    # First Test Executable
    message(STATUS "Compiling test headers-test-macho")
    add_executable(headers-test-macho
            ${CMAKE_CURRENT_SOURCE_DIR}/headers-test-macho.cpp
    )

    target_include_directories(headers-test-macho
            BEFORE
            PUBLIC ${SHURIKEN_INCLUDE_PATH}
    )

    # Add dependency on shuriken library
    add_dependencies(headers-test-macho shuriken)

    target_link_libraries(headers-test-macho shuriken)

    add_test(NAME headers-test-macho
            COMMAND headers-test-macho)
endif ()