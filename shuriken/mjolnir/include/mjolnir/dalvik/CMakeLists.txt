set(LLVM_TARGET_DEFINITIONS MjolnirIROps.td)

# Deals with the enum declarations and definitions
add_mlir_tablegen(MjolnIREnums.h.inc -gen-enum-decls)
add_mlir_tablegen(MjolnIREnums.cpp.inc -gen-enum-defs)
