set(LLVM_TARGET_DEFINITIONS DalvikOps.td)

# Deals with the dialect decl and def
mlir_tablegen(DalvikDialect.h.inc -gen-dialect-decls --dialect="Dalvik" -I=${PROJECT_SOURCE_DIR}/include)
mlir_tablegen(DalvikDialect.cpp.inc -gen-dialect-defs --dialect="Dalvik" -I=${PROJECT_SOURCE_DIR}/include)

# Deals with the ops decl and def
mlir_tablegen(DalvikOps.h.inc -gen-op-decls -I=${PROJECT_SOURCE_DIR}/include)
mlir_tablegen(DalvikOps.cpp.inc -gen-op-defs -I=${PROJECT_SOURCE_DIR}/include)

# Deals with the enum declarations and definitions
mlir_tablegen(DalvikEnums.h.inc -gen-enum-decls -I=${PROJECT_SOURCE_DIR}/include)
mlir_tablegen(DalvikEnums.cpp.inc -gen-enum-defs -I=${PROJECT_SOURCE_DIR}/include)

# Deals with the types decl and def
mlir_tablegen(DalvikTypes.h.inc -gen-typedef-decls -I=${PROJECT_SOURCE_DIR}/include)
mlir_tablegen(DalvikTypes.cpp.inc -gen-typedef-defs -I=${PROJECT_SOURCE_DIR}/include)


add_public_tablegen_target(DalvikTblgen)

set(LLVM_TARGET_DEFINITIONS DalvikPatterns.td)

mlir_tablegen(DalvikPatterns.h.inc -gen-rewriters -I=${PROJECT_SOURCE_DIR}/include)

add_public_tablegen_target(DalvikPatternTblgen )
