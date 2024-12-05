

#ifndef LOWER_MJOLNIR_LOWER_HPP
#define LOWER_MJOLNIR_LOWER_HPP

#include "mjolnir/MjolnIRDialect.h"

#include "dalvik/DalvikPatterns.h.inc"
#include "mjolnir/MjolnIROps.h"
#include "mjolnir/MjolnIRTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
/// MLIR includes
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
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
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
namespace shuriken {
    namespace MjolnIR {
        class LowerAdd : public mlir::OpRewritePattern<mlir::arith::AddIOp> {
            using LogicalResult = llvm::LogicalResult;
            LowerAdd(mlir::MLIRContext *context) : OpRewritePattern(context, /*benefit=*/3) {}

            LogicalResult matchAndRewrite(mlir::arith::AddIOp op, mlir::PatternRewriter &rewriter) const override {
                auto rhs = op.getRhs();
                if (llvm::isa<mlir::arith::ConstantOp>(rhs)) return llvm::success();
                else
                    return llvm::success();
            }
        };
        class LowerFallThrough : public mlir::OpRewritePattern<mlir::shuriken::MjolnIR::FallthroughOp> {
            using LogicalResult = llvm::LogicalResult;
            LowerFallThrough(mlir::MLIRContext *context) : OpRewritePattern(context, /*benefit=*/1) {}

            LogicalResult matchAndRewrite(mlir::shuriken::MjolnIR::FallthroughOp op, mlir::PatternRewriter &rewriter) const override {
                rewriter.eraseOp(op);
                return llvm::success();
            }
        };

        /// The Lower class applies all lowering patterns that is added to it when it enters/encounters a FuncOp
        class DalvikLower : public mlir::PassWrapper<DalvikLower, mlir::OperationPass<mlir::func::FuncOp>> {

        private:
            void registerMjolnIRToDalvikLowering(mlir::MLIRContext &context, mlir::RewritePatternSet &s) {
                s.add<LowerFallThrough>(&context);
            }
            void runOnOperation() override {
                mlir::func::FuncOp func = getOperation();
                mlir::MLIRContext &context = getContext();

                // Create a RewritePatternSet and add your patterns
                mlir::RewritePatternSet patterns(&context);
                registerMjolnIRToDalvikLowering(context, patterns);

                // Apply the patterns to the function
                if (mlir::failed(mlir::applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
                    signalPassFailure();
                }
            }
        };

        class Lower {
        public:
            Lower() {
                mlir::MLIRContext context;
                // Load required dialects
                context.getOrLoadDialect<mlir::shuriken::MjolnIR::MjolnIRDialect>();
                context.getOrLoadDialect<mlir::func::FuncDialect>();
                mlir::PassManager pm(&context);
                pm.addPass(std::make_unique<DalvikLower>());
            }
        };
    }// namespace MjolnIR
}// namespace shuriken
#endif// LOWER_MJOLNIR_LOWER
