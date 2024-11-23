
// @file MjolnIROps.cpp

#include "dalvik/DalvikOps.h"
#include "dalvik/DalvikDialect.h"
#include "dalvik/DalvikTypes.h"

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
using namespace ::mlir::shuriken::Dalvik;

#define GET_OP_CLASSES
#include "dalvik/DalvikOps.cpp.inc"
