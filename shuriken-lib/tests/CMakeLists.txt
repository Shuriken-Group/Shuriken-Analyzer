
# Test configuration
option(BUILD_TESTS "Build the Shuriken tests" ON)


if (BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG release-1.12.1
    )
    FetchContent_MakeAvailable(googletest)

    # Create the test executable
    add_executable(shuriken_tests
            main.cpp
            dex/dvm_type_tests.cpp
            # Add other test files here
            dex/dvm_prototypes_tests.cpp
    )

    # Link against the Shuriken library
    target_link_libraries(shuriken_tests
            PRIVATE
            LIB_SHURIKEN
            GTest::gtest
            GTest::gtest_main
    )

    # Include directories for tests
    target_include_directories(shuriken_tests
            PRIVATE
            ${SHURIKEN_PUBLIC_INCLUDE_DIR}
            ${SHURIKEN_PRIVATE_INCLUDE_DIR}
    )

    # Enable CTest integration
    include(GoogleTest)
    gtest_discover_tests(shuriken_tests)

endif ()