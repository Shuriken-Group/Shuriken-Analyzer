add_library(MjolnIRTransformLib lifter.cpp
  lift_instructions/Instruction10t.cpp
  lift_instructions/Instruction10x.cpp
  lift_instructions/Instruction11n.cpp
  lift_instructions/Instruction11x.cpp
  lift_instructions/Instruction12x.cpp
  lift_instructions/Instruction20t.cpp
  lift_instructions/Instruction21c.cpp
  lift_instructions/Instruction22c.cpp
  lift_instructions/Instruction21h.cpp
  lift_instructions/Instruction21s.cpp
  lift_instructions/Instruction21t.cpp
  lift_instructions/Instruction22b.cpp
  lift_instructions/Instruction22s.cpp
  lift_instructions/Instruction22t.cpp
  lift_instructions/Instruction22x.cpp
  lift_instructions/Instruction23x.cpp
  lift_instructions/Instruction30t.cpp
  lift_instructions/Instruction31c.cpp
  lift_instructions/Instruction31i.cpp
  lift_instructions/Instruction32x.cpp
  lift_instructions/Instruction35c.cpp
  lift_instructions/Instruction51l.cpp
  mjolnir_to_smali.cpp)

target_compile_options(MjolnIRTransformLib PRIVATE -fno-rtti) # https://makslevental.github.io/working-with-mlir/#fixing-missing-__ztvn4mlir4passe-ie-typeinfo-for-mlirpass
target_link_libraries(MjolnIRTransformLib LLVMSupport
    MjolnIRLib
    MLIRArithDialect
    MLIRControlFlowDialect
    MLIRControlFlowInterfaces
    MLIRSideEffectInterfaces
    MLIRAnalysis
    MLIRFunctionInterfaces
    MLIRParser
    MLIRTransforms
    MLIRFuncDialect
    MLIRPass
    MLIRAnalysis
    MLIRSupport
    fmt::fmt
)

add_dependencies(MjolnIRTransformLib ShurikenTblGen)
target_include_directories(MjolnIRTransformLib PRIVATE ${PROJECT_SOURCE_DIR}/include/)
target_include_directories(MjolnIRTransformLib PRIVATE ${PROJECT_BINARY_DIR}/include/)
