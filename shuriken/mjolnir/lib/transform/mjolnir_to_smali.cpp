#include "transform/mjolnir_to_smali.h"
#include <cstdlib>
#include <fmt/core.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <string>
#include <vector>

namespace shuriken::MjolnIR {
    using namespace mlir;
    using namespace mlir::shuriken::MjolnIR;


    SmaliLines MjolnIRToSmali::get_smali_lines() { return smali_lines; }


    void MjolnIRToSmali::runOnOperation() {
        auto *outer_op = getOperation();

        if (auto method_op = llvm::dyn_cast<MethodOp>(outer_op)) {
            // Handle the method operation itself
            auto [prologue, epilogue] = from_mjolnir_method_op(method_op);
            smali_lines.insert(smali_lines.end(), prologue.begin(), prologue.end());

            // Recurse into the method body
            for (Block &block: method_op.getBody()) {
                if (!block.isEntryBlock())
                    smali_lines.emplace_back(fmt::format(":block_{}", block_counter.get_counter(&block)));
                for (Operation &gen_op: block) {
                    bool matched_an_op = false;
                    // Process each nested operation
                    // INFO: ARITH
                    SmaliLine smali_line;
                    if (auto op = llvm::dyn_cast<arith::AddIOp>(gen_op)) {
                        smali_line = from_arith_addi(op);
                        matched_an_op = true;
                    } else if (auto op = llvm::dyn_cast<arith::MulIOp>(gen_op)) {
                        smali_line = from_arith_muli(op);
                        matched_an_op = true;
                    } else if (auto op = llvm::dyn_cast<arith::DivSIOp>(gen_op)) {
                        smali_line = from_arith_divsi(op);
                        matched_an_op = true;
                    } else if (auto op = llvm::dyn_cast<arith::CmpIOp>(gen_op)) {
                        smali_line = from_arith_cmpi(op);
                        matched_an_op = true;
                    }
                    // INFO: MJOLNIR
                    else if (auto op = llvm::dyn_cast<ReturnOp>(gen_op)) {
                        smali_line = from_mjolnir_return_op(op);
                        matched_an_op = true;
                    } else if (auto op = llvm::dyn_cast<FallthroughOp>(gen_op)) {
                        smali_line = from_mjolnir_fallthrough(op);
                        matched_an_op = true;
                    } else if (auto op = llvm::dyn_cast<LoadFieldOp>(gen_op)) {
                        smali_line = from_mjolnir_loadfield(op);
                        matched_an_op = true;
                    } else if (auto op = llvm::dyn_cast<StoreFieldOp>(gen_op)) {
                        smali_line = from_mjolnir_storefield(op);
                        matched_an_op = true;
                    } else if (auto op = llvm::dyn_cast<LoadValue>(gen_op)) {
                        smali_line = from_mjolnir_loadvalue(op);
                        matched_an_op = true;
                    } else if (auto op = llvm::dyn_cast<MoveOp>(gen_op)) {
                        smali_line = from_mjolnir_move(op);
                        matched_an_op = true;
                    } else if (auto op = llvm::dyn_cast<InvokeOp>(gen_op)) {
                        smali_line = from_mjolnir_invoke(op);
                        matched_an_op = true;
                    } else if (auto op = llvm::dyn_cast<NewOp>(gen_op)) {
                        smali_line = from_mjolnir_new(op);
                        matched_an_op = true;
                    } else if (auto op = llvm::dyn_cast<GetArrayOp>(gen_op)) {
                        smali_line = from_mjolnir_getarray(op);
                        matched_an_op = true;
                    }
                    if (matched_an_op) {
                        smali_lines.emplace_back(TAB + smali_line);
                        continue;
                    }
                    // INFO: Control flow

                    SmaliLines temp_smali_lines;
                    if (auto op = llvm::dyn_cast<cf::CondBranchOp>(gen_op)) {
                        temp_smali_lines = from_cf_condbr(op);
                        matched_an_op = true;
                    }

                    else if (auto op = llvm::dyn_cast<cf::BranchOp>(gen_op)) {
                        temp_smali_lines = from_cf_br(op);
                        matched_an_op = true;
                    }

                    if (matched_an_op) {
                        for (auto &temp_smali_line: temp_smali_lines)
                            smali_lines.emplace_back(TAB + temp_smali_line);
                        continue;
                    }

                    if (!matched_an_op) {

                        llvm::errs() << "Instruction not supported to lower to smali right now, speak to Jasmine or Edu\n";
                        llvm::errs() << "Operation name: " << gen_op.getName() << "\n";
                        std::abort();
                    }
                }
            }

            smali_lines.insert(smali_lines.end(), epilogue.begin(), epilogue.end());
        } else {
            llvm::errs() << "Encountered an outer operation different than MethodOp, speak to Jasmine or Edu";
            llvm::errs() << "Operation name: " << outer_op->getName() << "\n";
            std::abort();
        }
    }
}// namespace shuriken::MjolnIR
namespace shuriken::MjolnIR {
    std::vector<std::string> to_smali(std::vector<mlir::OwningOpRef<mlir::ModuleOp>> &modules) {
        /// INFO: Shared resources of ModuleOP, all of these ModuleOp supposedly come from the same file, and thus,
        /// share the same supposedly virtual register
        auto virtual_reg_counter = SmaliCounter<Value>();
        auto block_counter = SmaliCounter<Block *>();
        SmaliLines smali_lines;

        for (auto &module: modules) {
            mlir::PassManager pm(module.get()->getName());

            /// INFO: Since a pass manager loves to manage a pass's unique ptr, we let the ptr holds the references of shared resources instead
            /// and once its done running, we just return the smali lines
            auto pass = std::make_unique<MjolnIRToSmali>(smali_lines, virtual_reg_counter, block_counter);
            pm.addNestedPass<mlir::shuriken::MjolnIR::MethodOp>(std::move(pass));

            auto result = pm.run(*module);
            if (result.failed()) {
                llvm::errs() << "Failed in mjolnir to smali for " << module.get().getName() << " \n";
            }
        }
        return smali_lines;
    }
}// namespace shuriken::MjolnIR
