
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

        return fmt::format("add-int v{}, v{}, v{}", vrc.get_counter(res), vrc.get_counter(lhs), vrc.get_counter(rhs));

        return "";
    }
    SmaliLine MjolnIRToSmali::from_arith_muli(arith::MulIOp op) {
        auto lhs = op.getLhs();
        auto rhs = op.getRhs();
        auto res = op.getResult();

        return fmt::format("mul-int v{}, v{}, v{}", vrc.get_counter(res), vrc.get_counter(lhs), vrc.get_counter(rhs));
    }
    SmaliLine MjolnIRToSmali::from_arith_divsi(arith::DivSIOp op) {
        auto lhs = op.getLhs();
        auto rhs = op.getRhs();
        auto res = op.getResult();

        return fmt::format("div-int v{}, v{}, v{}", vrc.get_counter(res), vrc.get_counter(lhs), vrc.get_counter(rhs));
    }
    SmaliLine MjolnIRToSmali::from_arith_cmpi(arith::CmpIOp op) {
        auto lhs = op.getLhs();
        auto rhs = op.getRhs();
        auto res = op.getResult();

        previous_predicate = op.getPredicate();
        std::string pred_str = "cmp-long";
        return fmt::format("{} v{}, v{}, v{}", pred_str, vrc.get_counter(res), vrc.get_counter(lhs), vrc.get_counter(rhs));
    }


}// namespace shuriken::MjolnIR
