

#pragma once
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

namespace shuriken {
    namespace MjolnIR {

        class Opt {
            mlir::MLIRContext context;

            mlir::PassManager pm;

        public:
            // TODO: explain PM::On
            Opt();

            mlir::LogicalResult run(mlir::ModuleOp &&module);
        };
    }// namespace MjolnIR
}// namespace shuriken
