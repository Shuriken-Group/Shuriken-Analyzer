#include "iostream"
#include "mjolnir/MjolnIROps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Support/LLVM.h>
#include <string>
#include <vector>
using namespace mlir;
namespace shuriken::MjolnIR {

    using namespace mlir::shuriken::MjolnIR;

    class MjolnIRToSmali : public PassWrapper<MjolnIRToSmali, OperationPass<>> {
        class VirtualRegisterCounter {
            size_t virtual_register_counter = 0;

        public:
            size_t get_new_virtual_register() { return virtual_register_counter++; }
        };

        std::vector<std::string> smali_lines;
        VirtualRegisterCounter vrc;
        std::map<Value *, int> mlir_value_to_virtual_reg_map;
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
        std::vector<std::string> get_smali_lines();
    };

}// namespace shuriken::MjolnIR
