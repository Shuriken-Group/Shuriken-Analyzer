//------------------------------------------------------------------- -*- cpp -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file MjolnIROps.cpp

#include "mjolnir/MjolnIROps.h"
#include "mjolnir/MjolnIRDialect.h"
#include "mjolnir/MjolnIRTypes.h"

// include from MLIR
#include <mlir/Bytecode/BytecodeReader.h>
#include <mlir/Bytecode/BytecodeWriter.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>

#include <mlir/Interfaces/FunctionImplementation.h>
#include <mlir/Support/LLVM.h>
#include <mlir/TableGen/Operator.h>
#include <mlir/Transforms/InliningUtils.h>
// include from LLVM
#include <llvm/ADT/TypeSwitch.h>


using namespace mlir;
using namespace ::mlir::shuriken::MjolnIR;

/***
 * Following the example from the Toy language from MLIR webpage
 * we will provide here some useful methods for managing parsing,
 * printing, and build constructors
 */

/// @brief Parser for binary operation and functions
/// @param parser parser object
/// @param result result
/// @return
#define GET_OP_CLASSES
#include "mjolnir/MjolnIROps.cpp.inc"

using namespace mlir;
using namespace ::mlir::shuriken::MjolnIR;

//===----------------------------------------------------------------------===//
// MethodOp
//===----------------------------------------------------------------------===//

void MethodOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                     llvm::StringRef name, mlir::FunctionType type,
                     llvm::ArrayRef<mlir::NamedAttribute> attrs) {
    // FunctionOpInterface provides a convenient `build` method that will populate
    // the stateof our MethodOp, and create an entry block
    buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
}

mlir::ParseResult MethodOp::parse(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
    auto buildFuncType =
            [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
               llvm::ArrayRef<mlir::Type> results,
               mlir::function_interface_impl::VariadicFlag,
               std::string &) { return builder.getFunctionType(argTypes, results); };

    return mlir::function_interface_impl::parseFunctionOp(
            parser, result, /*allowVariadic=*/false,
            getFunctionTypeAttrName(result.name), buildFuncType,
            getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void MethodOp::print(mlir::OpAsmPrinter &p) {
    // Dispatch to the FunctionOpInterface provided utility method that prints the
    // function operation.
    mlir::function_interface_impl::printFunctionOp(
            p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
            getArgAttrsAttrName(), getResAttrsAttrName());
}

/// Returns the region on the function operation that is callable.
mlir::Region *MethodOp::getCallableRegion() { return &getBody(); }

ArrayRef<Type> MethodOp::getArgumentTypes() { return getFunctionType().getInputs(); }

/// Returns the result types of this function.
ArrayRef<Type> MethodOp::getResultTypes() { return getFunctionType().getResults(); }

//===----------------------------------------------------------------------===//
// InvokeOp
//===----------------------------------------------------------------------===//

/// Return the callee of the generic call operation, this is required by the
/// call interface.
CallInterfaceCallable InvokeOp::getCallableForCallee() {
    return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

void InvokeOp::setCalleeFromCallable(CallInterfaceCallable callee) {
    (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
Operation::operand_range InvokeOp::getArgOperands() { return getInputs(); }

::mlir::MutableOperandRange InvokeOp::getArgOperandsMutable() { return getInputsMutable(); }
//===----------------------------------------------------------------------===//
// FallthroughOp
//===----------------------------------------------------------------------===//
void FallthroughOp::setDest(Block *block) { return setSuccessor(block); }

void FallthroughOp::eraseOperand(unsigned index) { (*this)->eraseOperand(index); }
