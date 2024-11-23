
// @file DalvikDialect.td
// @brief Main file for the Dialect of Dalvik with MLIR, here we will find basic
// definitions of main components, for example the dialect itself.

#ifndef SHURIKENPROJECT_DALVIK_TD
#define SHURIKENPROJECT_DALVIK_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Dalvik dialect definition.
//===----------------------------------------------------------------------===//

def Dalvik_Dialect : Dialect
{
    let name = "Dalvik";
    let summary = "Dialect for the instructions of Dalvik Virtual Machine";
    let description = [{
      This dialect tries to represent the instructions from the Dalvik Virtual Machine
      together with the instruction we will find also the different types found on
      this bytecode (Fields, Methods, Classes).
    }];

    let cppNamespace = "::mlir::shuriken::Dalvik";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}
// Base class for the operations of Dalvik. Operations will
// inherit from base `Op` class from MLIR. It provides:
//      * Parent dialect of operation
//      * Mnemonic for the operation, or name without dialect prefix
//      * List of traits for the operation
class Dalvik_Op<string mnemonic, list<Trait> traits = []> :
    Op<Dalvik_Dialect, mnemonic, traits>;
#endif // SHURIKENPROJECT_DALVIK_TD
