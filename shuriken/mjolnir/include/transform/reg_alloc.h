
#pragma once
#include <cstddef>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LLVM.h>


namespace shuriken::MjolnIR {
    mlir::DenseMap<mlir::Operation *, mlir::DenseMap<mlir::Value, size_t>> linear_register_alloc(mlir::ModuleOp &cfg);
}
