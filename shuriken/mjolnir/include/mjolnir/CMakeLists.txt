set(LLVM_TARGET_DEFINITIONS MjolnIROps.td)

# Deals with the dialect decl and def
mlir_tablegen(MjolnIRDialect.h.inc -gen-dialect-decls -I=${PROJECT_SOURCE_DIR}/include  -dialect=MjolnIR)
mlir_tablegen(MjolnIRDialect.cpp.inc -gen-dialect-defs -I=${PROJECT_SOURCE_DIR}/include -dialect=MjolnIR)

# Deals with the ops decl and def
mlir_tablegen(MjolnIROps.h.inc -gen-op-decls -I=${PROJECT_SOURCE_DIR}/include -dialect=MjolnIR)
mlir_tablegen(MjolnIROps.cpp.inc -gen-op-defs -I=${PROJECT_SOURCE_DIR}/include -dialect=MjolnIR)

# Deals with the enum declarations and definitions
mlir_tablegen(MjolnIREnums.h.inc -gen-enum-decls -I=${PROJECT_SOURCE_DIR}/include -dialect=MjolnIR)
mlir_tablegen(MjolnIREnums.cpp.inc -gen-enum-defs -I=${PROJECT_SOURCE_DIR}/include -dialect=MjolnIR)

# Deals with the types decl and def
mlir_tablegen(MjolnIRTypes.h.inc -gen-typedef-decls -I=${PROJECT_SOURCE_DIR}/include -typedefs-dialect=MjolnIR -dialect=MjolnIR)
mlir_tablegen(MjolnIRTypes.cpp.inc -gen-typedef-defs -I=${PROJECT_SOURCE_DIR}/include -typedefs-dialect=MjolnIR  -dialect=MjolnIR)

add_public_tablegen_target(ShurikenTblGen)
