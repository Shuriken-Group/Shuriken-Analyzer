
#include "passes/opt.h"
#include "mjolnir/MjolnIRDialect.h"
#include "mlir/Rewrite/PatternApplicator.h"


#include "mjolnir/MjolnIROps.h"
#include "mjolnir/MjolnIRTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
/// MLIR includes
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>
namespace shuriken::MjolnIR::Opt {
    using namespace mlir;

    /// The Lower class applies all lowering patterns that is added to it when it enters/encounters a FuncOp
    class MjolnIRRemoveNop : public mlir::PassWrapper<MjolnIRRemoveNop, mlir::OperationPass<mlir::ModuleOp>> {

        class OptNop : public mlir::OpRewritePattern<mlir::shuriken::MjolnIR::Nop> {

        public:
            OptNop(mlir::MLIRContext *context) : OpRewritePattern(context, /*benefit=*/1) {}

            LogicalResult matchAndRewrite(mlir::shuriken::MjolnIR::Nop op, mlir::PatternRewriter &rewriter) const override {
                llvm::errs() << "Matching and rewriting FallthroughOp\n";
                llvm::errs() << "Successfully erased FallthroughOp\n";
                rewriter.eraseOp(op);
                return success();
            }
        };

    public:
        void runOnOperation() override {
            std::cerr << "RunOpOperation called \n";
            auto module = getOperation();
            mlir::MLIRContext &context = getContext();

            // Create a RewritePatternSet and add your patterns
            mlir::RewritePatternSet patterns(&context);
            patterns.add<OptNop>(&context);

            if (mlir::failed(mlir::applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
                return;
            }
        }
    };
    Opt::Opt() : context(mlir::MLIRContext()), pm(&context) {
        // Load required dialects
        context.loadAllAvailableDialects();
        context.getOrLoadDialect<::mlir::shuriken::MjolnIR::MjolnIRDialect>();
        context.getOrLoadDialect<::mlir::cf::ControlFlowDialect>();
        context.getOrLoadDialect<::mlir::arith::ArithDialect>();
        context.getOrLoadDialect<::mlir::func::FuncDialect>();
    }
    mlir::LogicalResult Opt::remove_nop(mlir::ModuleOp &&module) {
        pm.clear();
        pm.addPass(std::make_unique<MjolnIRRemoveNop>());
        std::cerr << "Running a module\n";
        auto result = pm.run(module);
        return result;
    }

}// namespace shuriken::MjolnIR::Opt
