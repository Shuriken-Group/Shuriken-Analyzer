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
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/FunctionImplementation.h>
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
[[maybe_unused]]
static mlir::ParseResult parseBinaryOp(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
    SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
    SMLoc operandsLoc = parser.getCurrentLocation();
    Type type;
    if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(type))
        return mlir::failure();

    // If the type is a function type, it contains the input and result types of
    // this operation. mlir::dyn_cast<FunctionType>()
    if (FunctionType funcType = mlir::dyn_cast<FunctionType>(type)) {
        if (parser.resolveOperands(operands, funcType.getInputs(), operandsLoc,
                                   result.operands))
            return mlir::failure();
        result.addTypes(funcType.getResults());
        return mlir::success();
    }

    // Otherwise, the parsed type is the type of both operands and results.
    if (parser.resolveOperands(operands, type, result.operands))
        return mlir::failure();
    result.addTypes(type);
    return mlir::success();
}

[[maybe_unused]]
static void printBinaryOp(mlir::OpAsmPrinter &printer, mlir::Operation *op) {
    printer << " " << op->getOperands();
    printer.printOptionalAttrDict(op->getAttrs());
    printer << " : ";

    // If all of the types are the same, print the type directly
    Type resultType = *op->result_type_begin();
    if (llvm::all_of(op->getOperandTypes(),
                     [=](Type type) { return type == resultType; })) {
        printer << resultType;
        return;
    }

    // Otherwise, print a functional type
    printer.printFunctionalType(op->getOperandTypes(), op->getResultTypes());
}

//===----------------------------------------------------------------------===//
// MethodOp
//===----------------------------------------------------------------------===//

// [[maybe_unused]]
// void MethodOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
//                      llvm::StringRef name, mlir::FunctionType type,
//                      llvm::ArrayRef<mlir::NamedAttribute> attrs) {
//     // FunctionOpInterface provides a convenient `build` method that will populate
//     // the stateof our MethodOp, and create an entry block
//     buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
// }
//
// [[maybe_unused]]
// mlir::ParseResult MethodOp::parse(mlir::OpAsmParser &parser,
//                                   mlir::OperationState &result) {
//     auto buildFuncType =
//             [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
//                llvm::ArrayRef<mlir::Type> results,
//                mlir::function_interface_impl::VariadicFlag,
//                std::string &) { return builder.getFunctionType(argTypes, results); };
//
//     return mlir::function_interface_impl::parseFunctionOp(
//             parser, result, /*allowVariadic=*/false,
//             getFunctionTypeAttrName(result.name), buildFuncType,
//             getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
// }
// [[maybe_unused]]
// void MethodOp::print(mlir::OpAsmPrinter &p) {
//     // Dispatch to the FunctionOpInterface provided utility method that prints the
//     // function operation.
//     mlir::function_interface_impl::printFunctionOp(
//             p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
//             getArgAttrsAttrName(), getResAttrsAttrName());
// }
//
// /// Returns the region on the function operation that is callable.
// [[maybe_unused]]
// mlir::Region *MethodOp::getCallableRegion() { return &getBody(); }
//
// /// Returns results types that callable region produces when executed
// [[maybe_unused]]
// llvm::ArrayRef<mlir::Type> MethodOp::getCallableResults() {
//     return getFunctionType().getResults();
// }
// [[maybe_unused]]
// mlir::ArrayAttr MethodOp::getCallableArgAttrs() {
//     return mlir::ArrayAttr();
// }
// [[maybe_unused]]
// mlir::ArrayAttr MethodOp::getCallableResAttrs() {
//     return mlir::ArrayAttr();
// }
//
// //===----------------------------------------------------------------------===//
// // InvokeOp
// //===----------------------------------------------------------------------===//
// [[maybe_unused]]
// void InvokeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
//                      StringRef callee, ArrayRef<mlir::Value> arguments, MethodOp &method) {
//     state.addTypes(method.getResultTypes());
//     state.addOperands(arguments);
//     state.addAttribute("callee",
//                        mlir::SymbolRefAttr::get(builder.getContext(), callee));
// }
//
// /// Return the callee of the generic call operation, this is required by the
// /// call interface.
// [[maybe_unused]]
// CallInterfaceCallable InvokeOp::getCallableForCallee() {
//     return (*this)->getAttrOfType<SymbolRefAttr>("callee");
// }
//
// /// Get the argument operands to the called function, this is required by the
// /// call interface.
// [[maybe_unused]]
// Operation::operand_range InvokeOp::getArgOperands() { return getInputs(); }
//
// //===----------------------------------------------------------------------===//
// // FallthroughOp
// //===----------------------------------------------------------------------===//
// void FallthroughOp::setDest(Block *block) { return setSuccessor(block); }
//
// void FallthroughOp::eraseOperand(unsigned index) { (*this)->eraseOperand(index); }
//

#define GET_OP_CLASSES
#include "mjolnir/MjolnIROps.cpp.inc"
