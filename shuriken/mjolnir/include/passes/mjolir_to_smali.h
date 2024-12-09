#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Support/LLVM.h>
#include <string>
#include <vector>
using namespace mlir;
namespace shuriken::MjolnIR {
    std::vector<std::string> to_smali(std::vector<mlir::OwningOpRef<mlir::ModuleOp>> &modules);
}// namespace shuriken::MjolnIR
