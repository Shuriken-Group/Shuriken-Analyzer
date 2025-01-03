#include "transform/mjolnir_to_smali.h"
#include "mjolnir/MjolnIROps.h"
#include <cstdlib>
#include <fmt/core.h>
#include <iostream>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LLVM.h>
#include <string>
#include <vector>

namespace shuriken::MjolnIR {
    using namespace mlir;
    using namespace mlir::shuriken::MjolnIR;


    void MjolnIRToSmali::emitOnMethodOp(MethodOp method_op) {

        // SUMMARY: We'll get the prologue and epilogue from mjolnir_method_op
        // as well as the lines from the body
        //
        // From the body, we calculate the number of locals used
        // Then we add to smali_lines in this order
        // prologue
        // number of locals used
        // body lines
        // epilogue


        // Handle the method operation itself
        auto [prologue, epilogue] = from_mjolnir_method_op(method_op);
        smali_lines.insert(smali_lines.end(), prologue.begin(), prologue.end());

        // Recurse into the method body
        SmaliLines body_lines;
        for (Block &block: method_op.getBody()) {
            if (!block.isEntryBlock())
                body_lines.emplace_back(fmt::format(":block_{}", block_counter.get_counter(&block)));
            if (block.isEntryBlock()) {
              /// Pre counting the argument of function 
              for (auto &arg : block.getArguments()) { func_arg_counter.get_counter(arg); }
            }
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
                } else if (auto op = llvm::dyn_cast<LoadString>(gen_op)) {
                    smali_line = from_mjolnir_loadstring(op);
                    matched_an_op = true;
                } else if (auto op = llvm::dyn_cast<Nop>(gen_op)) {
                    matched_an_op = true;
                }
                if (matched_an_op) {
                    body_lines.emplace_back(std::string(TAB) + smali_line);
                    continue;
                }
                // INFO: Control flow is different since it requires multiple lines
                SmaliLines temp_body_lines;
                if (auto op = llvm::dyn_cast<cf::CondBranchOp>(gen_op)) {
                    temp_body_lines = from_cf_condbr(op);
                    matched_an_op = true;
                }

                else if (auto op = llvm::dyn_cast<cf::BranchOp>(gen_op)) {
                    temp_body_lines = from_cf_br(op);
                    matched_an_op = true;
                }

                if (matched_an_op) {
                    for (auto &temp_smali_line: temp_body_lines)
                        body_lines.emplace_back(std::string(TAB) + temp_smali_line);
                    continue;
                }

                if (!matched_an_op) {

                    llvm::errs() << "Instruction not supported to lower to smali right now, speak to Jasmine or Edu\n";
                    llvm::errs() << "Operation name: " << gen_op.getName() << "\n";
                    std::abort();
                }
            }
            smali_lines.push_back(fmt::format("{}.locals {}", std::string(TAB), std::to_string(vrc.get_size())));
            smali_lines.insert(smali_lines.end(), body_lines.begin(), body_lines.end());
            smali_lines.insert(smali_lines.end(), epilogue.begin(), epilogue.end());


            // after every method, reset the counters
            vrc.clean_counter();
            func_arg_counter.clean_counter();
            block_counter.clean_counter();
        }
    }
    std::string MjolnIRToSmali::get_smali_value(mlir::Value val) {
        if (auto block_arg = llvm::dyn_cast<BlockArgument>(val)) {
            if (block_arg.getParentBlock()->isEntryBlock()) {
                return fmt::format("p{}", func_arg_counter.get_counter(val));
            }
        }
        return fmt::format("v{}", vrc.get_counter(val));
    }
    std::string get_smali_access_flag(dex::TYPES::access_flags acc_flag) {
        switch (acc_flag) {
            case dex::TYPES::NONE:
                return "";
            case dex::TYPES::ACC_PUBLIC:
                return "public";
            case dex::TYPES::ACC_PRIVATE:
                return "private";
            case dex::TYPES::ACC_PROTECTED:
                return "protected";
            case dex::TYPES::ACC_STATIC:
                return "static";
            case dex::TYPES::ACC_FINAL:
                return "final";
            case dex::TYPES::ACC_SYNCHRONIZED:
                return "synchronized";
            case dex::TYPES::ACC_VOLATILE:
                return "volatile";
            case dex::TYPES::ACC_TRANSIENT:
                return "transient";
            case dex::TYPES::ACC_NATIVE:
                return "native";
            case dex::TYPES::ACC_INTERFACE:
                return "interface";
            case dex::TYPES::ACC_ABSTRACT:
                return "abstract";
            case dex::TYPES::ACC_STRICT:
                return "strict";
            case dex::TYPES::ACC_SYNTHETIC:
                return "synthetic";
            case dex::TYPES::ACC_ANNOTATION:
                return "annotation";
            case dex::TYPES::ACC_ENUM:
                return "enum";
            case dex::TYPES::UNUSED:
                return "unused";
            case dex::TYPES::ACC_CONSTRUCTOR:
                return "constructor";
            case dex::TYPES::ACC_DECLARED_SYNCHRONIZED:
                return "declared_synchronized";
        }
        return "non-matching smali access flags, contact Jasmine or Edu";
    }
    void MjolnIRToSmali::runOnOperation() {
        auto *outer_op = getOperation();
        if (auto module_op = llvm::dyn_cast<ModuleOp>(outer_op)) {
            module_op.walk([this](MethodOp op) {
                emitOnMethodOp(op);
            });
        } else {
            llvm::errs() << "The outer op must be ModuleOp, instead got: Operation name: " << outer_op->getName() << "\n";
        }
        return;
    }
}// namespace shuriken::MjolnIR


namespace shuriken::MjolnIR {
    std::vector<std::string> to_smali(std::vector<mlir::OwningOpRef<mlir::ModuleOp>> &modules) {
        /// INFO: Shared resources of ModuleOP, all of these ModuleOp supposedly come from the same file, and thus,
        /// share the same supposedly virtual register
        SmaliLines smali_lines;

        for (auto &module: modules) {
            mlir::PassManager pm(module.get()->getName());

            /// INFO: Since a pass manager loves to manage a pass's unique ptr, we let the ptr holds the references of shared resources instead
            /// and once its done running, we just return the smali lines
            pm.addPass(std::make_unique<MjolnIRToSmali>(smali_lines));

            auto result = pm.run(*module);
            if (result.failed()) {
                llvm::errs() << "Failed in mjolnir to smali for " << module.get().getName() << " \n";
            }
        }
        return smali_lines;
    }
}// namespace shuriken::MjolnIR
