message(hello ${PROJECT_SOURCE_DIR})
add_mlir_dialect_library(MjolnIRLib
    MjolnIRTypes.cpp
    MjolnIRDialect.cpp
    MjolnIROps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/
    ${PROJECT_BINARY_DIR}/include/

    DEPENDS
    ShurikenTblGen

    LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRControlFlowInterfaces
    MLIRIR
    MLIRSideEffectInterfaces
    MLIRAnalysis
    MLIRFunctionInterfaces
    MLIRParser
    MLIRTransforms
)
target_include_directories(MjolnIRLib PRIVATE ${PROJECT_SOURCE_DIR}/include/)
target_include_directories(MjolnIRLib PRIVATE ${PROJECT_BINARY_DIR}/include/)
