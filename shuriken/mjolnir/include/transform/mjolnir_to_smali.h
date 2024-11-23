#pragma once
#include <cstddef>
#include <filesystem>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Support/LLVM.h>
#include <optional>
#include <string>
#include <vector>

#include <cstdlib>
#include <fmt/core.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>


/// MjolnIR headers
#include "mjolnir/MjolnIRDialect.h"
#include "mjolnir/MjolnIROps.h"
#include "mjolnir/MjolnIRTypes.h"
#include "shuriken/common/Dex/dvm_types.h"
namespace shuriken::MjolnIR {
    using SmaliLine = std::string;
    using SmaliLines = std::vector<std::string>;
    SmaliLines to_smali(std::vector<mlir::OwningOpRef<mlir::ModuleOp>> &modules);

    using ::shuriken::MjolnIR::SmaliLine;
    using ::shuriken::MjolnIR::SmaliLines;
    inline constexpr std::string_view TAB = "    ";
    template<class Aspect>
    class SmaliCounter {
        size_t counter = 0;

        mlir::DenseMap<Aspect, size_t> counter_map;

    public:
        size_t get_counter(Aspect a) {

            if (this->counter_map.find(a) == this->counter_map.end())
                this->counter_map[a] = counter++;

            return this->counter_map[a];
        }

        void clean_counter() {
            counter = 0;
            counter_map.clear();
        }
    };


    using namespace mlir;
    using namespace mlir::shuriken::MjolnIR;

    class MjolnIRToSmali : public PassWrapper<MjolnIRToSmali, OperationPass<>> {
        std::optional<arith::CmpIPredicate> previous_predicate;

        /// Virtual register counter, which is different than parameter_counter
        SmaliCounter<Value> vrc;
        SmaliCounter<Block *> block_counter;
        SmaliCounter<Value> func_arg_counter;
        SmaliLines &smali_lines;
        void runOnOperation() override;

        /// INFO: ARITH
        SmaliLine from_arith_constintop(arith::ConstantIntOp);
        SmaliLine from_arith_addi(arith::AddIOp);
        SmaliLine from_arith_muli(arith::MulIOp);
        SmaliLine from_arith_divsi(arith::DivSIOp);
        SmaliLine from_arith_cmpi(arith::CmpIOp);

        /// INFO: MJOLNIR
        std::tuple<SmaliLines, SmaliLines> from_mjolnir_method_op(MethodOp);
        SmaliLine from_mjolnir_return_op(ReturnOp);
        SmaliLine from_mjolnir_fallthrough(FallthroughOp);
        SmaliLine from_mjolnir_loadfield(LoadFieldOp);
        SmaliLine from_mjolnir_storefield(StoreFieldOp);
        SmaliLine from_mjolnir_move(MoveOp);
        SmaliLine from_mjolnir_invoke(InvokeOp);
        SmaliLine from_mjolnir_loadvalue(LoadValue);
        SmaliLine from_mjolnir_new(NewOp);
        SmaliLine from_mjolnir_getarray(GetArrayOp);
        SmaliLine from_mjolnir_loadstring(LoadString);


        /// INFO: Control flow dialect
        SmaliLines from_cf_condbr(cf::CondBranchOp);
        SmaliLines from_cf_br(cf::BranchOp);


        void emitOnMethodOp(MethodOp);

        /// INFO: Get either vX or pX, depending on if the mlir value is block argument of the entry block or otherwise
        std::string get_smali_value(mlir::Value);
        std::string get_smali_access_flag(dex::TYPES::access_flags);

    public:
        MjolnIRToSmali(SmaliLines &smali_lines) : smali_lines(smali_lines) {}
    };
}// namespace shuriken::MjolnIR
