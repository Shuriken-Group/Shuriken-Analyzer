set(LLVM_TARGET_DEFINITIONS MjolnIROps.td)

# Deals with the dialect decl and def
mlir_tablegen(MjolnIRDialect.h.inc -gen-dialect-decls -dialect=MjolnIR)
mlir_tablegen(MjolnIRDialect.cpp.inc -gen-dialect-defs -dialect=MjolnIR)

# Deals with the ops decl and def
mlir_tablegen(MjolnIROps.h.inc -gen-op-decls -dialect=MjolnIR)
mlir_tablegen(MjolnIROps.cpp.inc -gen-op-defs -dialect=MjolnIR)

# Deals with the enum declarations and definitions
mlir_tablegen(MjolnIREnums.h.inc -gen-enum-decls -dialect=MjolnIR)
mlir_tablegen(MjolnIREnums.cpp.inc -gen-enum-defs -dialect=MjolnIR)

# Deals with the types decl and def - note the different flag format
mlir_tablegen(MjolnIRTypes.h.inc -gen-typedef-decls --typedefs-dialect=MjolnIR)
mlir_tablegen(MjolnIRTypes.cpp.inc -gen-typedef-defs --typedefs-dialect=MjolnIR)

add_public_tablegen_target(ShurikenTblGen)