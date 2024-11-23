
// @file DalvikDialect.cpp

#include "dalvik/DalvikDialect.h"
#include "dalvik/DalvikOps.h"
#include "dalvik/DalvikTypes.h"

using namespace mlir;
using namespace ::mlir::shuriken::Dalvik;

// import the cpp generated from tablegen
#include "dalvik/DalvikDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Dalvik Dialect
//===----------------------------------------------------------------------===//

// initialize the operations from those generated
// with tablegen
void DalvikDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "dalvik/DalvikOps.cpp.inc"
            >();
}
