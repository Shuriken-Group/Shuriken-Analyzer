
add_mlir_dialect_library(DalvikMLIRLib
    DalvikTypes.cpp
    DalvikDialect.cpp
    DalvikOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/
    ${PROJECT_BINARY_DIR}/include/

    DEPENDS
    DalvikTblgen
    DalvikPatternTblgen   

    LINK_LIBS PUBLIC
    LLVMSupport
    MLIRArithDialect
    MLIRFuncDialect
    MLIRControlFlowInterfaces
    MLIRSideEffectInterfaces
    MLIRAnalysis
    MLIRFunctionInterfaces
    MLIRParser
    MLIRTransforms
)
target_include_directories(DalvikMLIRLib PRIVATE ${PROJECT_SOURCE_DIR}/include/)
target_include_directories(DalvikMLIRLib PRIVATE ${PROJECT_BINARY_DIR}/include/)
