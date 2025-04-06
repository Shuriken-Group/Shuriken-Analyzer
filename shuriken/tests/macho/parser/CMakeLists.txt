if(MACHO AND MACHO_TESTS)
    configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/macho-files-folder.in
            ${CMAKE_CURRENT_SOURCE_DIR}/macho-files-folder.inc
    )

    # First Test Executable
    message(STATUS "Compiling test parse-test-macho")
    add_executable(parse-test-macho
            ${CMAKE_CURRENT_SOURCE_DIR}/parse-test-macho.cpp
    )

    target_include_directories(parse-test-macho
            BEFORE
            PUBLIC ${SHURIKEN_INCLUDE_PATH}
    )

    # Add dependency on shuriken library
    add_dependencies(parse-test-macho shuriken)

    target_link_libraries(parse-test-macho shuriken)

    add_test(NAME parse-test-macho
            COMMAND parse-test-macho)
endif ()