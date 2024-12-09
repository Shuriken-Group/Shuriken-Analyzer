add_library(MjolnIRPasses
mjolnirtoopgraph.cpp
mjolir_to_smali.cpp
)

target_compile_options(MjolnIRPasses PRIVATE -fno-rtti)

target_link_libraries(MjolnIRPasses LLVMSupport
    MLIRAnalysis
    MLIRPass
    MLIRTransforms
    MLIRSupport
    MLIRFuncTransforms
    MLIRSCFDialect
    MLIRIR               
    MLIRDialect
)

add_dependencies(MjolnIRPasses ShurikenTblGen)
target_include_directories(MjolnIRPasses PRIVATE ${PROJECT_SOURCE_DIR}/include/)
target_include_directories(MjolnIRPasses PRIVATE ${PROJECT_BINARY_DIR}/include/)
