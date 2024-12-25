//------------------------------------------------------------------- -*- tablegen -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file MjolnIRDialect.td
// @brief Main file for the Dialect of Dalvik with MLIR, here we will find basic
// definitions of main components, for example the dialect itself.

#ifndef SHURIKENPROJECT_MJOLNIR_TD
#define SHURIKENPROJECT_MJOLNIR_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// MjolnIR dialect definition.
//===----------------------------------------------------------------------===//

def MjolnIR_Dialect : Dialect
{
    let name = "MjolnIR";
    let summary = "Dialect for the instructions of Dalvik Virtual Machine";
    let description = [{
      This dialect tries to represent the instructions from the Dalvik Virtual Machine
      together with the instruction we will find also the different types found on
      this bytecode (Fields, Methods, Classes).
    }];

    let cppNamespace = "::mlir::shuriken::MjolnIR";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

// Base class for the operations of MjolnIR. Operations will
// inherit from base `Op` class from MLIR. It provides:
//      * Parent dialect of operation
//      * Mnemonic for the operation, or name without dialect prefix
//      * List of traits for the operation
class MjolnIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<MjolnIR_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// MjolnIR_FieldType
// TODO: Put some kind of explanation for the existence of this field type
//
//===----------------------------------------------------------------------===//
def MjolnIR_FieldType : I32EnumAttr<
    "FieldType", "",
    [
        I32EnumAttrCase<"STATIC", 0, "static">,
        I32EnumAttrCase<"INSTANCE", 1, "instance">
    ]
> {
    let cppNamespace = "::mlir::shuriken::MjolnIR";
}


//===----------------------------------------------------------------------===//
// MjolnIR_InvokeTypeAttr
// Attribute used for specifying the type of invoke we are working with.
//===----------------------------------------------------------------------===//
def MjolnIR_InvokeTypeAttr : I32EnumAttr<
    "InvokeType", "Different types of invoke instructions in Dalvik",
    [
      I32EnumAttrCase<"VIRTUAL", 0, "virtual">,
      I32EnumAttrCase<"SUPER", 1, "super">,
      I32EnumAttrCase<"DIRECT", 2, "direct">,
      I32EnumAttrCase<"STATIC", 3, "static">,
      I32EnumAttrCase<"INTERFACE", 4, "interface">,
      I32EnumAttrCase<"NONE", 99, "none">
    ]> {
  let cppNamespace = "::mlir::shuriken::MjolnIR";
}

//===----------------------------------------------------------------------===//
// MjolnIR_FieldTypeAttr
// Attribute used for specifying the type of load/get of a field (instance or static)
//===----------------------------------------------------------------------===//
def MjolnIR_FieldTypeAttr : I32EnumAttr<
    "FieldAccessType", "Different types of load/get of a field",
    [
        I32EnumAttrCase<"INSTANCE", 0, "instance">,
        I32EnumAttrCase<"STATIC", 1, "static">,
        I32EnumAttrCase<"NONE", 99, "none">
    ]> {
    let cppNamespace = "::mlir::shuriken::MjolnIR";
}

//===----------------------------------------------------------------------===//
// MjolnIR_MethodFlags
// Attribute used for specifying the type of Method in a MethodOp
//===----------------------------------------------------------------------===//
def MjolnIR_MethodFlagsAttr : I32BitEnumAttr<
   "MethodFlags", "Access and modifier flags for methods",
   [
       I32BitEnumAttrCase<"NONE", 0x0, "none">,
       I32BitEnumAttrCase<"PUBLIC", 0x1, "public">,
       I32BitEnumAttrCase<"PRIVATE", 0x2, "private">, 
       I32BitEnumAttrCase<"PROTECTED", 0x4, "protected">,
       I32BitEnumAttrCase<"STATIC", 0x8, "static">,
       I32BitEnumAttrCase<"FINAL", 0x10, "final">,
       I32BitEnumAttrCase<"SYNCHRONIZED", 0x20, "synchronized">,
       I32BitEnumAttrCase<"VOLATILE", 0x40, "volatile">,
       I32BitEnumAttrCase<"TRANSIENT", 0x80, "transient">,
       I32BitEnumAttrCase<"NATIVE", 0x100, "native">,
       I32BitEnumAttrCase<"INTERFACE", 0x200, "interface">,
       I32BitEnumAttrCase<"ABSTRACT", 0x400, "abstract">,
       I32BitEnumAttrCase<"STRICT", 0x800, "strict">,
       I32BitEnumAttrCase<"SYNTHETIC", 0x1000, "synthetic">,
       I32BitEnumAttrCase<"ANNOTATION", 0x2000, "annotation">,
       I32BitEnumAttrCase<"ENUM", 0x4000, "enum">,
       I32BitEnumAttrCase<"UNUSED", 0x8000, "unused">,
       I32BitEnumAttrCase<"CONSTRUCTOR", 0x10000, "constructor">,
       I32BitEnumAttrCase<"DECLARED_SYNCHRONIZED", 0x20000, "declared_synchronized">
   ]> {
   let cppNamespace = "::mlir::shuriken::MjolnIR";
}


#endif // SHURIKENPROJECT_MJOLNIR_TD
