
#include "transform/mjolnir_to_smali.h"
#include <mlir/Dialect/Arith/IR/Arith.h>

namespace {
    std::string arith_cmpipredicate_to_str(mlir::arith::CmpIPredicate pred) {
        std::string result = "";
        switch (pred) {

            case mlir::arith::CmpIPredicate::eq:
                result = "eq";
                break;
            case mlir::arith::CmpIPredicate::ne:
                result = "ne";
                break;
            case mlir::arith::CmpIPredicate::slt:
            case mlir::arith::CmpIPredicate::ult:
                result = "lt";
                break;
            case mlir::arith::CmpIPredicate::sle:
            case mlir::arith::CmpIPredicate::ule:
                result = "le";
                break;
            case mlir::arith::CmpIPredicate::sgt:
            case mlir::arith::CmpIPredicate::ugt:
                result = "gt";
                break;
            case mlir::arith::CmpIPredicate::sge:
            case mlir::arith::CmpIPredicate::uge:
                result = "ge";
                break;
        }
        return result;
    }
}// namespace

namespace shuriken::MjolnIR {

    /// INFO: Control flow dialect
    SmaliLines MjolnIRToSmali::from_cf_condbr(cf::CondBranchOp op) {
        auto true_dest = op.getSuccessor(0), false_dest = op.getSuccessor(1);
        auto false_block_id = block_counter.get_counter(false_dest);
        auto true_block_id = block_counter.get_counter(true_dest);
        //
        // if (previous_predicate == arith::CmpIPredicate::eq)
        //     return "";
        SmaliLines handling;
        {
            // INFO: Handling of the true branch
            auto operands = op.getTrueOperands();
            auto bb_operands = true_dest->getArguments();
            auto op_it = operands.begin();
            auto bb_op_it = bb_operands.begin();
            for (; op_it != operands.end() && bb_op_it != bb_operands.end(); op_it++, bb_op_it++) {
                handling.emplace_back(fmt::format("move {}, {}", get_smali_value(*bb_op_it), get_smali_value(*op_it)));
            }
            auto pred = op->getOperand(0);
            auto comparision_type = arith_cmpipredicate_to_str(*previous_predicate);
            auto temp_result = fmt::format("if-{} {}, :block_{}", comparision_type, get_smali_value(pred), true_block_id);
            handling.emplace_back(temp_result);
        }


        {
            // INFO: Handling of the false branch
            auto operands = op.getFalseOperands();
            auto bb_operands = false_dest->getArguments();
            auto op_it = operands.begin();
            auto bb_op_it = bb_operands.begin();
            for (; op_it != operands.end() && bb_op_it != bb_operands.end(); op_it++, bb_op_it++) {
                handling.emplace_back(fmt::format("move {}, {}", get_smali_value(*bb_op_it), get_smali_value(*op_it)));
            }
            auto temp_result = fmt::format("goto :block_{}", false_block_id);
            handling.emplace_back(temp_result);
        }

        return handling;
    }
    SmaliLines MjolnIRToSmali::from_cf_br(cf::BranchOp op) {
        auto operands = op.getOperands();
        auto dest = op.getDest();
        auto bb_operands = dest->getArguments();
        auto op_it = operands.begin();
        auto bb_op_it = bb_operands.begin();
        SmaliLines handling;
        for (; op_it != operands.end() && bb_op_it != bb_operands.end(); op_it++, bb_op_it++) {
            handling.emplace_back(fmt::format("move {}, {}", vrc.get_counter(*bb_op_it), vrc.get_counter(*op_it)));
        }
        auto temp_result = fmt::format("goto :block_{}", block_counter.get_counter(dest));
        handling.emplace_back(temp_result);
        return handling;
    }
}// namespace shuriken::MjolnIR
