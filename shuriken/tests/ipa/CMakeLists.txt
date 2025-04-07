if(MACHO AND MACHO_TESTS)
    configure_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/ipa-files-folder.in
            ${CMAKE_CURRENT_SOURCE_DIR}/ipa-files-folder.inc
    )

    # Load core API
    message(STATUS "Compiling test parse-test-ipa")
    add_executable(parse-test-ipa
            ${CMAKE_CURRENT_SOURCE_DIR}/parse-test-ipa.cpp
    )

    target_include_directories(parse-test-ipa
            BEFORE
            PUBLIC ${SHURIKEN_INCLUDE_PATH}
    )

    # Add dependency on shuriken library
    add_dependencies(parse-test-ipa shuriken)

    target_link_libraries(parse-test-ipa shuriken)

    add_test(NAME parse-test-ipa
            COMMAND parse-test-ipa)

endif()