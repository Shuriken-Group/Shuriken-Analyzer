if(DEX AND DEX_TESTS)
    configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/dex-files-folder.in
        ${CMAKE_CURRENT_SOURCE_DIR}/dex-files-folder.inc
    )

    #--------------------------------------------------------------
    # Basic test
    message(STATUS "Compiling test basic-test")
    add_executable(basic-test
        ${CMAKE_CURRENT_SOURCE_DIR}/test_basictest.cpp
    )

    target_include_directories(basic-test
        BEFORE
        PUBLIC ${SHURIKEN_INCLUDE_PATH}
    )

    add_dependencies(basic-test shuriken)
    target_link_libraries(basic-test shuriken)

    add_test(NAME basic-test
        COMMAND basic-test)

    #--------------------------------------------------------------
    # Field analysis test
    message(STATUS "Compiling test field-analysis-test")
    add_executable(field-analysis-test
        ${CMAKE_CURRENT_SOURCE_DIR}/test_fields.cpp
    )

    target_include_directories(field-analysis-test
        BEFORE
        PUBLIC ${SHURIKEN_INCLUDE_PATH}
    )

    add_dependencies(field-analysis-test shuriken)
    target_link_libraries(field-analysis-test shuriken)

    add_test(NAME field-analysis-test
        COMMAND field-analysis-test)

    #--------------------------------------------------------------
    # Method analysis test
    message(STATUS "Compiling test method-analysis-test")
    add_executable(method-analysis-test
        ${CMAKE_CURRENT_SOURCE_DIR}/test_methods.cpp
    )

    target_include_directories(method-analysis-test
        BEFORE
        PUBLIC ${SHURIKEN_INCLUDE_PATH}
    )

    add_dependencies(method-analysis-test shuriken)
    target_link_libraries(method-analysis-test shuriken)

    add_test(NAME method-analysis-test
        COMMAND method-analysis-test)

    #--------------------------------------------------------------
    # Parser test
    message(STATUS "Compiling test parser-test")
    add_executable(parser-test
        ${CMAKE_CURRENT_SOURCE_DIR}/test_parser.cpp
    )

    target_include_directories(parser-test
        BEFORE
        PUBLIC ${SHURIKEN_INCLUDE_PATH}
    )

    add_dependencies(parser-test shuriken)
    target_link_libraries(parser-test shuriken)

    add_test(NAME parser-test
        COMMAND parser-test)

    #--------------------------------------------------------------
    # Disassembler test
    message(STATUS "Compiling test disassembler-test")
    add_executable(disassembler-test
        ${CMAKE_CURRENT_SOURCE_DIR}/test_disassembler.cpp
    )

    target_include_directories(disassembler-test
        BEFORE
        PUBLIC ${SHURIKEN_INCLUDE_PATH}
    )

    add_dependencies(disassembler-test shuriken)
    target_link_libraries(disassembler-test shuriken)

    add_test(NAME disassembler-test
        COMMAND disassembler-test)

    #--------------------------------------------------------------
    # Exceptions test
    message(STATUS "Compiling test exceptions-test")
    add_executable(exceptions-test
        ${CMAKE_CURRENT_SOURCE_DIR}/test_exceptions.cpp
    )

    target_include_directories(exceptions-test
        BEFORE
        PUBLIC ${SHURIKEN_INCLUDE_PATH}
    )

    add_dependencies(exceptions-test shuriken)
    target_link_libraries(exceptions-test shuriken)

    add_test(NAME exceptions-test
        COMMAND exceptions-test)
endif()
