

#ifndef LIFTER_MJOLNIR_LIFTER_HPP
#define LIFTER_MJOLNIR_LIFTER_HPP

#include "mjolnir/MjolnIRDialect.h"

#include "mjolnir/MjolnIRTypes.h"

#include "dalvik/DalvikPatterns.h.inc"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "shuriken/analysis/Dex/analysis.h"
/// MLIR includes
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>


namespace shuriken {
    namespace MjolnIR {
        class LowerAdd : public mlir::OpRewritePattern<mlir::arith::AddIOp> {};
    }// namespace MjolnIR
}// namespace shuriken
#endif// LIFTER_MJOLNIR_LIFTER
