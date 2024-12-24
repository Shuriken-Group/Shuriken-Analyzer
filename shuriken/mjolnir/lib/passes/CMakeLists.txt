add_library(MjolnIRPasses
mjolnirtoopgraph.cpp
opt.cpp
opt_const_prop.cpp
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
