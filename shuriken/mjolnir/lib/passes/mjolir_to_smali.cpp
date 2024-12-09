#include "passes/mjolir_to_smali.h"
#include <cstdlib>
#include <llvm/Support/Casting.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
namespace shuriken::MjolnIR {
    /// INFO: ARITH
    std::string MjolnIRToSmali::from_arith_addi(arith::AddIOp) { return ""; }
    std::string MjolnIRToSmali::from_arith_muli(arith::MulIOp) { return ""; }
    std::string MjolnIRToSmali::from_arith_divsi(arith::DivSIOp) { return ""; }


    /// INFO: MJOLNIR
    std::string MjolnIRToSmali::from_mjolnir_method_op(MethodOp) { return ""; }
    std::string MjolnIRToSmali::from_mjolnir_return_op(ReturnOp) { return ""; }
    std::string MjolnIRToSmali::from_mjolnir_fallthrough(FallthroughOp) { return ""; }
    std::string MjolnIRToSmali::from_mjolnir_loadfield(LoadFieldOp) { return ""; }
    std::string MjolnIRToSmali::from_mjolnir_storefield(StoreFieldOp) { return ""; }


    /// INFO: Control flow dialect
    std::string MjolnIRToSmali::from_cf_condbr(cf::CondBranchOp) { return ""; }
    std::string MjolnIRToSmali::from_cf_br(cf::BranchOp) { return ""; }


    std::vector<std::string> MjolnIRToSmali::get_smali_lines() { return smali_lines; }

    size_t MjolnIRToSmali::get_virtual_reg(Value *v) {
        if (this->mlir_value_to_virtual_reg_map.find(v) == this->mlir_value_to_virtual_reg_map.end()) {
            this->mlir_value_to_virtual_reg_map[v] = vrc.get_new_virtual_register();
        }
        return this->mlir_value_to_virtual_reg_map[v];
    }

    void MjolnIRToSmali::runOnOperation() {
        auto *gen_op = getOperation();

        if (auto op = llvm::dyn_cast<arith::AddIOp>(gen_op))
            smali_lines.emplace_back(from_arith_addi(op));
        else if (auto op = llvm::dyn_cast<arith::MulIOp>(gen_op))
            smali_lines.emplace_back(from_arith_muli(op));
        else if (auto op = llvm::dyn_cast<arith::DivSIOp>(gen_op))
            smali_lines.emplace_back(from_arith_divsi(op));
        else if (auto op = llvm::dyn_cast<MethodOp>(gen_op))
            smali_lines.emplace_back(from_mjolnir_method_op(op));
        else if (auto op = llvm::dyn_cast<ReturnOp>(gen_op))
            smali_lines.emplace_back(from_mjolnir_return_op(op));
        else if (auto op = llvm::dyn_cast<FallthroughOp>(gen_op))
            smali_lines.emplace_back(from_mjolnir_fallthrough(op));
        else if (auto op = llvm::dyn_cast<LoadFieldOp>(gen_op))
            smali_lines.emplace_back(from_mjolnir_loadfield(op));
        else if (auto op = llvm::dyn_cast<StoreFieldOp>(gen_op))
            smali_lines.emplace_back(from_mjolnir_storefield(op));
        else if (auto op = llvm::dyn_cast<cf::CondBranchOp>(gen_op))
            smali_lines.emplace_back(from_cf_condbr(op));
        else if (auto op = llvm::dyn_cast<cf::BranchOp>(gen_op))
            smali_lines.emplace_back(from_cf_br(op));
        else {
            llvm::errs() << "Instruction not supported to lower to smali right now, speak to Jasmine or Edu\n";
            llvm::errs() << "Operation name: " << gen_op->getName() << "\n";
            std::abort();
        }
    }
}// namespace shuriken::MjolnIR
