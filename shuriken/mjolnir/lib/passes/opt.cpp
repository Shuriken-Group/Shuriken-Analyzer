
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
namespace {
    using namespace mlir;
    class OptAdd : public mlir::OpRewritePattern<mlir::arith::AddIOp> {
        OptAdd(mlir::MLIRContext *context) : OpRewritePattern(context, /*benefit=*/3) {}

        LogicalResult matchAndRewrite(mlir::arith::AddIOp op, mlir::PatternRewriter &rewriter) const override {
            auto rhs = op.getRhs().getDefiningOp();
            if (llvm::isa<mlir::arith::ConstantOp>(rhs)) return success();
            else
                return success();
        }
    };
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

    /// The Lower class applies all lowering patterns that is added to it when it enters/encounters a FuncOp
    class MjolnIROpt : public mlir::PassWrapper<MjolnIROpt, mlir::OperationPass<mlir::ModuleOp>> {
        void registerMjolnIRToDalvikLowering(mlir::MLIRContext &context, mlir::RewritePatternSet &s) {
            s.add<OptNop>(&context);
        }

    public:
        void runOnOperation() override {
            std::cerr << "RunOpOperation called \n";
            auto module = getOperation();
            mlir::MLIRContext &context = getContext();

            // Create a RewritePatternSet and add your patterns
            mlir::RewritePatternSet patterns(&context);
            registerMjolnIRToDalvikLowering(context, patterns);

            if (mlir::failed(mlir::applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
                return;
            }
        }
    };

}// namespace
namespace shuriken {
    namespace MjolnIR {

        Opt::Opt() : context(mlir::MLIRContext()), pm(&context) {
            // Load required dialects
            context.loadAllAvailableDialects();
            context.getOrLoadDialect<::mlir::shuriken::MjolnIR::MjolnIRDialect>();
            context.getOrLoadDialect<::mlir::cf::ControlFlowDialect>();
            context.getOrLoadDialect<::mlir::arith::ArithDialect>();
            context.getOrLoadDialect<::mlir::func::FuncDialect>();
            pm.addPass(std::make_unique<MjolnIROpt>());
            pm.addPass(mlir::createSCCPPass());
        }
        mlir::LogicalResult Opt::run(mlir::ModuleOp &&module) {
            std::cerr << "Running a module\n";
            auto result = pm.run(module);
            return result;
        }
    }// namespace MjolnIR
}// namespace shuriken
