add_library(MjolnIRPasses
mjolnirtoopgraph.cpp

)

target_link_libraries(MjolnIRPasses
    MLIRAnalysis
    MLIRPass
    MLIRTransforms
    MLIRSupport
    MLIRFuncTransforms
    MLIRSCFDialect
)

add_dependencies(MjolnIRPasses ShurikenTblGen)
target_include_directories(MjolnIRPasses PRIVATE ${PROJECT_SOURCE_DIR}/include/)
target_include_directories(MjolnIRPasses PRIVATE ${PROJECT_BINARY_DIR}/include/)
