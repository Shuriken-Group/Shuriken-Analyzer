//------------------------------------------------------------------- -*- cpp -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file MjolnIRDialect.cpp


#include "mjolnir/MjolnIRTypes.h"

#include "mjolnir/MjolnIRDialect.h"
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>

using namespace ::mlir::shuriken::MjolnIR;

#define GET_TYPEDEF_CLASSES
#include "mjolnir/MjolnIRTypes.cpp.inc"

void MjolnIRDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "mjolnir/MjolnIRTypes.cpp.inc"
            >();
}
