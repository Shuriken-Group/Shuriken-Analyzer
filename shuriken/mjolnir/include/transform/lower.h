

#ifndef LOWER_MJOLNIR_LOWER_HPP
#define LOWER_MJOLNIR_LOWER_HPP

#include "mjolnir/MjolnIRDialect.h"
#include "mlir/Rewrite/PatternApplicator.h"


#include "dalvik/DalvikPatterns.h.inc"
#include "mjolnir/MjolnIROps.h"
#include "mjolnir/MjolnIRTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
/// MLIR includes
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
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
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>
namespace shuriken {
    namespace MjolnIR {
        class LowerAdd : public mlir::OpRewritePattern<mlir::arith::AddIOp> {
            using LogicalResult = llvm::LogicalResult;
            LowerAdd(mlir::MLIRContext *context) : OpRewritePattern(context, /*benefit=*/3) {}

            LogicalResult matchAndRewrite(mlir::arith::AddIOp op, mlir::PatternRewriter &rewriter) const override {
                auto rhs = op.getRhs().getDefiningOp();
                if (llvm::isa<mlir::arith::ConstantOp>(rhs)) return llvm::success();
                else
                    return llvm::success();
            }
        };
        class LowerNop : public mlir::OpRewritePattern<mlir::shuriken::MjolnIR::Nop> {
            using LogicalResult = llvm::LogicalResult;

        public:
            LowerNop(mlir::MLIRContext *context) : OpRewritePattern(context, /*benefit=*/1) {}

            LogicalResult matchAndRewrite(mlir::shuriken::MjolnIR::Nop op, mlir::PatternRewriter &rewriter) const override {
                llvm::errs() << "Matching and rewriting FallthroughOp\n";
                llvm::errs() << "Successfully erased FallthroughOp\n";
                rewriter.eraseOp(op);
                return llvm::success();
            }
        };
        // class LowerFallThrough : public mlir::OpRewritePattern<mlir::shuriken::MjolnIR::FallthroughOp> {
        //     using LogicalResult = llvm::LogicalResult;
        //
        // public:
        //     LowerFallThrough(mlir::MLIRContext *context) : OpRewritePattern(context, /*benefit=*/1) {}
        //
        //     LogicalResult matchAndRewrite(mlir::shuriken::MjolnIR::FallthroughOp op, mlir::PatternRewriter &rewriter) const override {
        //         llvm::errs() << "Matching and rewriting FallthroughOp\n";
        //         llvm::errs() << "Successfully erased FallthroughOp\n";
        //         return llvm::success();
        //     }
        // };

        /// The Lower class applies all lowering patterns that is added to it when it enters/encounters a FuncOp
        class DalvikLower : public mlir::PassWrapper<DalvikLower, mlir::OperationPass<mlir::ModuleOp>> {
            void registerMjolnIRToDalvikLowering(mlir::MLIRContext &context, mlir::RewritePatternSet &s) {
                s.add<LowerNop>(&context);
            }

        public:
            void runOnOperation() override {
                std::cerr << "RunOpOperation called \n";
                auto module = getOperation();
                mlir::MLIRContext &context = getContext();

                // Create a RewritePatternSet and add your patterns
                mlir::RewritePatternSet patterns(&context);
                registerMjolnIRToDalvikLowering(context, patterns);

                // bool should_lower = false;
                // // First, let's log what operations we find
                // module.walk([&](mlir::Operation *op) {
                //     llvm::errs() << "Found operation: " << op->getName() << "\n";
                //     if (llvm::isa<mlir::shuriken::MjolnIR::FallthroughOp>(op)) {
                //         llvm::errs() << "Found a FallthroughOp to lower\n";
                //         should_lower = true;
                //     } else {
                //         llvm::errs() << "Not one of the supported lowering methods, skipping it \n";
                //         should_lower = false;
                //     }
                // });
                //
                // if (!should_lower) return;
                // Apply the patterns to the entire module once
                llvm::errs() << "Applying patterns\n";
                if (mlir::failed(mlir::applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
                    llvm::errs() << "Failed to apply patterns\n";
                    return;
                }
            }
        };

        class Lower {
            mlir::MLIRContext context;

            mlir::PassManager pm;

        public:
            // TODO: explain PM::On
            Lower() : context(mlir::MLIRContext()), pm(&context) {
                // Load required dialects
                context.loadAllAvailableDialects();
                context.getOrLoadDialect<::mlir::shuriken::MjolnIR::MjolnIRDialect>();
                context.getOrLoadDialect<::mlir::cf::ControlFlowDialect>();
                context.getOrLoadDialect<::mlir::arith::ArithDialect>();
                context.getOrLoadDialect<::mlir::func::FuncDialect>();
                pm.addPass(std::make_unique<DalvikLower>());
                pm.addPass(mlir::createSCCPPass());
            }

            mlir::LogicalResult run(mlir::ModuleOp &&module) {
                std::cerr << "Running a module\n";
                auto result = pm.run(module);
                return result;
            }
        };
    }// namespace MjolnIR
}// namespace shuriken
#endif// LOWER_MJOLNIR_LOWER
