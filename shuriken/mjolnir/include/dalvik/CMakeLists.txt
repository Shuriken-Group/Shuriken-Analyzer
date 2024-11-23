set(LLVM_TARGET_DEFINITIONS DalvikOps.td)

# Deals with the dialect decl and def
mlir_tablegen(DalvikDialect.h.inc -gen-dialect-decls)
mlir_tablegen(DalvikDialect.cpp.inc -gen-dialect-defs)

# Deals with the ops decl and def
mlir_tablegen(DalvikOps.h.inc -gen-op-decls)
mlir_tablegen(DalvikOps.cpp.inc -gen-op-defs)

# Deals with the enum declarations and definitions
mlir_tablegen(DalvikEnums.h.inc -gen-enum-decls)
mlir_tablegen(DalvikEnums.cpp.inc -gen-enum-defs)

# Deals with the types decl and def
mlir_tablegen(DalvikTypes.h.inc -gen-typedef-decls)
mlir_tablegen(DalvikTypes.cpp.inc -gen-typedef-defs)


add_public_tablegen_target(DalvikTblgen)
