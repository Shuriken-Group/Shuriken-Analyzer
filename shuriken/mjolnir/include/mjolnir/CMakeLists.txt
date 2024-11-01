set(LLVM_TARGET_DEFINITIONS MjolnIROps.td)


# Deals with the dialect decl and def
mlir_tablegen(MjolnIRDialect.h.inc -gen-dialect-decls)
mlir_tablegen(MjolnIRDialect.cpp.inc -gen-dialect-defs)

# Deals with the ops decl and def
mlir_tablegen(MjolnIROps.h.inc -gen-op-decls)
mlir_tablegen(MjolnIROps.cpp.inc -gen-op-defs)

# Deals with the enum declarations and definitions
mlir_tablegen(MjolnIREnums.h.inc -gen-enum-decls)
mlir_tablegen(MjolnIREnums.cpp.inc -gen-enum-defs)

# Deals with the types decl and def
mlir_tablegen(MjolnIRTypes.h.inc -gen-typedef-decls)
mlir_tablegen(MjolnIRTypes.cpp.inc -gen-typedef-defs)


add_public_tablegen_target(ShurikenTblGen)
