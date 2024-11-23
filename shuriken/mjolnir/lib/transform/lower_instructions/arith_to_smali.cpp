
#include "transform/mjolnir_to_smali.h"
#include <cstdlib>
#include <mlir/Dialect/Arith/IR/Arith.h>

namespace shuriken::MjolnIR {
    /// INFO: ARITH
    SmaliLine MjolnIRToSmali::from_arith_constintop(arith::ConstantIntOp) { return ""; }
    SmaliLine MjolnIRToSmali::from_arith_addi(arith::AddIOp op) {
        auto lhs = op.getLhs();
        auto rhs = op.getRhs();
        auto res = op.getResult();

        return fmt::format("add-int {}, {}, {}", get_smali_value(res), get_smali_value(lhs), get_smali_value(rhs));

        return "";
    }
    SmaliLine MjolnIRToSmali::from_arith_muli(arith::MulIOp op) {
        auto lhs = op.getLhs();
        auto rhs = op.getRhs();
        auto res = op.getResult();

        return fmt::format("mul-int {}, {}, {}", get_smali_value(res), get_smali_value(lhs), get_smali_value(rhs));
    }
    SmaliLine MjolnIRToSmali::from_arith_divsi(arith::DivSIOp op) {
        auto lhs = op.getLhs();
        auto rhs = op.getRhs();
        auto res = op.getResult();

        return fmt::format("div-int {}, {}, {}", get_smali_value(res), get_smali_value(lhs), get_smali_value(rhs));
    }
    SmaliLine MjolnIRToSmali::from_arith_cmpi(arith::CmpIOp op) {
        auto lhs = op.getLhs();
        auto rhs = op.getRhs();
        auto res = op.getResult();

        previous_predicate = op.getPredicate();
        std::string pred_str = "cmp-long";
        return fmt::format("{} {}, {}, {}", pred_str, get_smali_value(res), get_smali_value(lhs), get_smali_value(rhs));
    }


}// namespace shuriken::MjolnIR
