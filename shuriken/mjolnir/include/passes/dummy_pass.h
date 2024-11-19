#include "iostream"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
using namespace mlir;
struct MyOperationPass : public PassWrapper<MyOperationPass, OperationPass<>> {
    void runOnOperation() override {
        // Get the current operation being operated on.
        Operation *op = getOperation();

        std::cerr << op->getName().getStringRef().str() << std::endl;
    }
};
