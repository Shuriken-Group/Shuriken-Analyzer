add_library(MjolnIRLifterLib lifter.cpp
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
  )

target_compile_options(MjolnIRLifterLib PRIVATE -fno-rtti) # https://makslevental.github.io/working-with-mlir/#fixing-missing-__ztvn4mlir4passe-ie-typeinfo-for-mlirpass
target_link_libraries(MjolnIRLifterLib LLVMSupport
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

add_dependencies(MjolnIRLifterLib ShurikenTblGen)
target_include_directories(MjolnIRLifterLib PRIVATE ${PROJECT_SOURCE_DIR}/include/)
target_include_directories(MjolnIRLifterLib PRIVATE ${PROJECT_BINARY_DIR}/include/)



add_library(MjolnIRLowerLib
  mjolnir_to_smali.cpp
  lower_instructions/arith_to_smali.cpp
  lower_instructions/mjolnir_to_smali.cpp
  lower_instructions/cf_to_smali.cpp
)
target_compile_options(MjolnIRLowerLib PRIVATE -fno-rtti) # https://makslevental.github.io/working-with-mlir/#fixing-missing-__ztvn4mlir4passe-ie-typeinfo-for-mlirpass
target_link_libraries(MjolnIRLowerLib LLVMSupport
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

add_dependencies(MjolnIRLowerLib ShurikenTblGen)
target_include_directories(MjolnIRLowerLib PRIVATE ${PROJECT_SOURCE_DIR}/include/)
target_include_directories(MjolnIRLowerLib PRIVATE ${PROJECT_BINARY_DIR}/include/)



add_library(MjolnIRRegAllocLib
  reg_alloc/linear_reg_alloc.cpp
)
target_compile_options(MjolnIRRegAllocLib PRIVATE -fno-rtti) 
target_link_libraries(MjolnIRRegAllocLib LLVMSupport
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

add_dependencies(MjolnIRRegAllocLib ShurikenTblGen)
target_include_directories(MjolnIRRegAllocLib PRIVATE ${PROJECT_SOURCE_DIR}/include/)
target_include_directories(MjolnIRRegAllocLib PRIVATE ${PROJECT_BINARY_DIR}/include/)
