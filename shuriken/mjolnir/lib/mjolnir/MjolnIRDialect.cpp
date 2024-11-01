//------------------------------------------------------------------- -*- cpp -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file MjolnIRDialect.cpp

#include "mjolnir/MjolnIRDialect.h"
#include "mjolnir/MjolnIROps.h"
#include "mjolnir/MjolnIRTypes.h"

using namespace mlir;
using namespace ::mlir::shuriken::MjolnIR;

// import the cpp generated from tablegen
#include "mjolnir/MjolnIRDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// MjolnIR Dialect
//===----------------------------------------------------------------------===//

// initialize the operations from those generated
// with tablegen
void MjolnIRDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "mjolnir/MjolnIROps.cpp.inc"
            >();

    registerTypes();
}
