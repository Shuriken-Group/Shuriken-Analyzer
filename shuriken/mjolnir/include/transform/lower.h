

#ifndef LIFTER_MJOLNIR_LIFTER_HPP
#define LIFTER_MJOLNIR_LIFTER_HPP

#include "mjolnir/MjolnIRDialect.h"

#include "mjolnir/MjolnIROps.h"
#include "mjolnir/MjolnIRTypes.h"

#include "dalvik/DalvikPatterns.h.inc"
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
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
using namespace mlir;
namespace shuriken {
    namespace MjolnIR {
        class LowerAdd : public mlir::OpRewritePattern<mlir::arith::AddIOp> {
            using LogicalResult = llvm::LogicalResult;
            LowerAdd(MLIRContext *context) : OpRewritePattern(context, /*benefit=*/3) {}

            LogicalResult matchAndRewrite(arith::AddIOp op, mlir::PatternRewriter &rewriter) const override {
                auto rhs = op.getRhs();
                if (llvm::isa<arith::ConstantOp>(rhs)) return llvm::success();
                else
                    return llvm::success();
            }
        };
        class LowerFallThrough : public mlir::OpRewritePattern<mlir::shuriken::MjolnIR::FallthroughOp> {
            using LogicalResult = llvm::LogicalResult;
            LowerFallThrough(MLIRContext *context) : OpRewritePattern(context, /*benefit=*/1) {}

            LogicalResult matchAndRewrite(mlir::shuriken::MjolnIR::FallthroughOp op, mlir::PatternRewriter &rewriter) const override {
                rewriter.eraseOp(op);
                return llvm::success();
            }
        };


        /// The Lower class applies all lowering patterns that is added to it when it enters/encounters a FuncOp
        class Lower : public mlir::PassWrapper<Lower, OperationPass<mlir::func::FuncOp>> {};
    }// namespace MjolnIR
}// namespace shuriken
#endif// LIFTER_MJOLNIR_LIFTER
