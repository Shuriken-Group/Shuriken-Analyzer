add_mlir_dialect_library(MjolnIRLib
    MjolnIRTypes.cpp
    MjolnIRDialect.cpp
    MjolnIROps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_BINARY_DIR}/mjolnir/include/mjolnir/include

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
