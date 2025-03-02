# DEX analysis tests
# include the paths for the different tests
add_subdirectory(dex/parser)
add_subdirectory(dex/disassembler)
add_subdirectory(dex/analysis)

# Different tests for the C API for the dex analysis
add_subdirectory(dex/api/C)


# APK analysis tests
# include the paths for the different tests
add_subdirectory(apk)

# Mach-O tests
add_subdirectory(macho/parser)