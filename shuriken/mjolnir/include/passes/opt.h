

#pragma once
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

namespace shuriken {
    namespace MjolnIR {
        namespace Opt {
            class Opt {
                mlir::MLIRContext context;

                mlir::PassManager pm;

            public:
                // TODO: explain PM::On
                Opt();
                mlir::LogicalResult remove_nop(mlir::ModuleOp &&module);
                mlir::LogicalResult const_prop(mlir::ModuleOp &&module);
            };
        }// namespace Opt
    }// namespace MjolnIR
}// namespace shuriken
