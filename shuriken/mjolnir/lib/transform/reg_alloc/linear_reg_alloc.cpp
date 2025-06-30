
#include "mjolnir/MjolnIROps.h"
#include <cstdint>
#include <cstdlib>
#include <fmt/core.h>
#include <iostream>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <string>
#include <vector>

// MP02 paper : https://link.springer.com/content/pdf/10.1007/3-540-45937-5_17.pdf
namespace shuriken::MjolnIR {
    // NOTE: We're not creating a seperate nameless namespace while in prototyping mode because of strict compiler warning.
    using namespace mlir;
    using namespace mlir::shuriken::MjolnIR;
    // todo: this should be llvm::Expected or llvm::Result i don't remember
    //
    // INFO: From 4.1 of the MP02 paper, as a helper function for gen_moves
    //
    // This function puts a newly created block between "pred" and "succ"
    //
    // It also makes the terminator from "pred" go to "between", and create a new terminator going from "between" to "succ"
    void glue_from_between_to(Block *pred, Block *between, Block *succ) {
        auto term = pred->getTerminator();
        int succIndex = -1;
        {
            int i = 0;
            for (auto p: term->getSuccessors()) {
                if (p == succ) {
                    succIndex = i;
                    break;
                }
            }
            if (succIndex == -1) {

                llvm::errs() << "Error in mjolnir to smali glue pred between\n";
                return;
            }
        }

        term->setSuccessor(between, succIndex);
        mlir::OpBuilder builder = mlir::OpBuilder(between, between->end());

        // we can call begin() like this because the newly created block can only have 1
        builder.create<mlir::cf::BranchOp>(builder.getUnknownLoc(), between, *between->getArguments().begin());
    }
    void add_move_and_rename_bb_args(Block *from, Block *to) {
        assert(from && to && "two blocks must be non null");
        auto *terminator = from->getTerminator();
        auto jump_args = terminator->getOperands();

        SmallVector<Value, 4> newArgs;
        mlir::OpBuilder builder = mlir::OpBuilder(from, std::prev(from->end()));
        for (auto arg: jump_args) {
            auto moved = builder.create<MjolnIR::MoveOp>(arg.getLoc(), arg.getType(), arg);
            newArgs.push_back(moved.getResult());
        }
        terminator->setOperands(newArgs);
    }

    // INFO: From 4.1 of the MP02 paper
    void gen_moves(MethodOp &region) {
        Block *n = nullptr;
        for (auto &b: region) {
            auto pred_begin = b.pred_begin();
            auto pred_end = b.pred_end();
            auto numPred = std::distance(pred_begin, pred_end);
            for (auto it = pred_begin; it != pred_end; it++) {
                auto p = *it;
                if (numPred > 1 and p->getNumSuccessors() > 1) {
                    n = region.addBlock();
                    // todo: how to glue from p to n, and then n to b,
                    // this function is not implemented rn
                    // clue: something terminator
                    // rename so that p's terminator arg that previously pointing to b is now pointing to n
                    //
                    glue_from_between_to(p, n, &b);
                } else
                    n = p;

                add_move_and_rename_bb_args(n, &b);
            }
        }
    }


    // INFO: From 4.2 of the MP02 paper
    mlir::DenseMap<mlir::Operation *, size_t> op_counter(MethodOp &method_op) {
        decltype(op_counter(method_op)) op_mapping;
        return op_mapping;
    }
    using LRStart = uint32_t;
    using LREnd = uint32_t;
    using LiveRange = std::pair<LRStart, LREnd>;// From above description

    // todo: do add range here
    void add_range(Operation *i, Block *b, uint32_t end) {
        // jasmine plz do this
    }

    // INFO: From 4.3 of the MP02 paper
    DenseMap<Value, SmallVector<LiveRange>> build_intervals(Region &region) {

        decltype(build_intervals(region)) live_intervals;
        // TODO: i know my ass will need the smali counter again soon. Maybe split the function here.

        // TODO: We get the value from the operand like this

        /*for b in blocks:*/
        /*    for op in b:*/
        /*        if v = op.getValue() then*/
        /*            smali_counter.getCounter(v);*/
        /**/
        // todo: we do this because although we visualize ssa as having an inherent counter, it is not visible in the
        // mlir api
        return live_intervals;
    }


    mlir::DenseMap<mlir::Operation *, mlir::DenseMap<mlir::Value, uint32_t>> linear_register_alloc(mlir::ModuleOp &&cfg) {
        mlir::DenseMap<mlir::Operation *, mlir::DenseMap<mlir::Value, uint32_t>> reg_mapping;
        cfg.walk([](MethodOp op) {
            gen_moves(op);
        });
        // TODO: Do something here to get the mapping
        return reg_mapping;
    };
}// namespace shuriken::MjolnIR
