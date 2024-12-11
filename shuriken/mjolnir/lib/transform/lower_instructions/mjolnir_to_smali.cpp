#include "transform/mjolnir_to_smali.h"
#include <cstdlib>
#include <iostream>
namespace shuriken::MjolnIR {
    /// INFO: MJOLNIR
    std::tuple<SmaliLines, SmaliLines> MjolnIRToSmali::from_mjolnir_method_op(MethodOp) {
        SmaliLines prologue = {".method"};
        SmaliLines epilogue = {".end method"};
        return {prologue, epilogue};
    }
    SmaliLine MjolnIRToSmali::from_mjolnir_return_op(ReturnOp op) {
        auto operands = op.getOperands();
        if (operands.size() == 0) {
            return "return";
        } else if (operands.size() == 1) {
            return fmt::format("return v{}", vrc.get_counter(operands.front()));
        }
        std::cerr << "Returning more than 1 operand, which is an impossible variant\n";
        std::abort();
    }

    SmaliLine MjolnIRToSmali::from_mjolnir_fallthrough(FallthroughOp) { return "nop"; }
    SmaliLine MjolnIRToSmali::from_mjolnir_loadfield(LoadFieldOp) { return "INCOMPLETE_LOAD_FIELD"; }
    SmaliLine MjolnIRToSmali::from_mjolnir_storefield(StoreFieldOp) { return "INCOMPLETE_STORE_FIELD"; }
    SmaliLine MjolnIRToSmali::from_mjolnir_loadvalue(LoadValue) { return ""; }
    SmaliLine MjolnIRToSmali::from_mjolnir_move(MoveOp op) {
        auto dest = op.getResult();
        auto operand = op.getOperand();


        return fmt::format("move v{}, v{}", vrc.get_counter(dest), vrc.get_counter(operand));
    }
    SmaliLine MjolnIRToSmali::from_mjolnir_invoke(InvokeOp) { return ""; }
    SmaliLine MjolnIRToSmali::from_mjolnir_new(NewOp) { return ""; }
    SmaliLine MjolnIRToSmali::from_mjolnir_getarray(GetArrayOp) { return ""; }

}// namespace shuriken::MjolnIR
