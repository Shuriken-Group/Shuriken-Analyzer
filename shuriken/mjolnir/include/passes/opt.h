

#pragma once
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

namespace shuriken {
    namespace MjolnIR {
        namespace Opt {
            mlir::LogicalResult run(mlir::ModuleOp &&module);
        }
    }// namespace MjolnIR
}// namespace shuriken
