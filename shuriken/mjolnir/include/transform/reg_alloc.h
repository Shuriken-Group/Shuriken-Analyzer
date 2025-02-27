
#pragma once
#include <cstddef>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LLVM.h>


namespace shuriken::MjolnIR {
    // INFO: from the theory of reigster allocation:
    //
    // The question we wanna ask is, for a specific operation, and the ssa value S in it, what register would S occupy.
    mlir::DenseMap<mlir::Operation *, mlir::DenseMap<mlir::Value, size_t>> linear_register_alloc(mlir::ModuleOp &&cfg);
}// namespace shuriken::MjolnIR
