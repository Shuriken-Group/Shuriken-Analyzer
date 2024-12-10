#include "transform/mjolnir_to_smali.h"
#include <cstdlib>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <string>
#include <vector>
/// MjolnIR headers
#include "mjolnir/MjolnIRDialect.h"
#include "mjolnir/MjolnIROps.h"
#include "mjolnir/MjolnIRTypes.h"

namespace {
    using namespace mlir;
    using namespace mlir::shuriken::MjolnIR;
    class VirtualRegisterCounter {
        size_t virtual_register_counter = 0;

    public:
        size_t get_new_virtual_register() { return virtual_register_counter++; }
    };

    class MjolnIRToSmali : public PassWrapper<MjolnIRToSmali, OperationPass<>> {
        std::vector<std::string> &smali_lines;
        VirtualRegisterCounter &vrc;
        std::map<Value *, int> &mlir_value_to_virtual_reg_map;
        void runOnOperation() override;

        /// INFO: ARITH
        std::string from_arith_addi(arith::AddIOp);
        std::string from_arith_muli(arith::MulIOp);
        std::string from_arith_divsi(arith::DivSIOp);


        /// INFO: MJOLNIR
        std::string from_mjolnir_method_op(MethodOp);
        std::string from_mjolnir_return_op(ReturnOp);
        std::string from_mjolnir_fallthrough(FallthroughOp);
        std::string from_mjolnir_loadfield(LoadFieldOp);
        std::string from_mjolnir_storefield(StoreFieldOp);


        /// INFO: Control flow dialect
        std::string from_cf_condbr(cf::CondBranchOp);
        std::string from_cf_br(cf::BranchOp);

        size_t get_virtual_reg(Value *v);

    public:
        MjolnIRToSmali(std::vector<std::string> &smali_lines, VirtualRegisterCounter &vrc, std::map<Value *, int> &mlir_value_to_virtual_reg_map) : smali_lines(smali_lines), vrc(vrc), mlir_value_to_virtual_reg_map(mlir_value_to_virtual_reg_map) {}
        std::vector<std::string> get_smali_lines();
    };

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
}// namespace
namespace shuriken::MjolnIR {
    std::vector<std::string> to_smali(std::vector<mlir::OwningOpRef<mlir::ModuleOp>> &modules) {
        /// INFO: Shared resources of ModuleOP, all of these ModuleOp supposedly come from the same file, and thus,
        /// share the same supposedly virtual register
        auto vrc = VirtualRegisterCounter();
        std::map<Value *, int> mlir_value_to_virtual_reg_map;
        std::vector<std::string> smali_lines;

        for (auto &module: modules) {
            mlir::PassManager pm(module.get()->getName());

            /// INFO: Since a pass manager loves to manage a pass's unique ptr, we let the ptr holds the references of shared resources instead
            /// and once its done running, we just return the smali lines
            auto pass = std::make_unique<MjolnIRToSmali>(smali_lines, vrc, mlir_value_to_virtual_reg_map);
            pm.addNestedPass<mlir::shuriken::MjolnIR::MethodOp>(std::move(pass));

            auto result = pm.run(*module);
            if (result.failed()) {
                llvm::errs() << "Failed in mjolnir to smali for " << module.get().getName() << " \n";
            }
        }
        return smali_lines;
    }
}// namespace shuriken::MjolnIR
