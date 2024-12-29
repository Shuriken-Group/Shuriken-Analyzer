//------------------------------------------------------------------- -*- tablegen -*-
// Kunai-static-analyzer: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file MjolnIRTypes.td
// @brief Definition of types from the Dalvik Virtual Machine.

#ifndef DALVIK_MJOLNIRTYPES_TD
#define DALVIK_MJOLNIRTYPES_TD

include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/AttrTypeBase.td"
include "mjolnir/MjolnIRDialect.td"

class MjolnIRType<string typeName, string typeMnemonic, list<Trait> traits = []>
  : TypeDef<MjolnIR_Dialect, typeName, traits>
{
  let summary = typeName;
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// Basic Types
//===----------------------------------------------------------------------===//

def DVMVoid : MjolnIRType<"DVMVoid", "dvmvoid"> {
  let summary = "Void type from Dalvik, returns nothing";
}

def DVMObject : MjolnIRType<"DVMObject", "object"> {
  let summary = "An object from any Java class";

  let parameters = (ins
    StringRefParameter<"the name of the class">:$value
  );

  let assemblyFormat = [{
    `<` $value `>`
  }];
}

def DVMArray : MjolnIRType<"DVMArray", "dvmarray"> {
  let summary = "Array type from dalvik";
  let description = [{
    Represents a Dalvik array type. Maps to Java array types.
    
    Examples:
      dvmarray<"I">      // int[]
      dvmarray<"String"> // String[]
  }];

  let parameters = (ins
        StringRefParameter<"the name of the type the array is holding">:$array_type
  );
  let assemblyFormat = [{
    `<` $array_type `>`
  }];
}

def DVM_BasicTypes : AnyTypeOf<[Builtin_Integer, Builtin_Float32, Builtin_Float64, DVMObject, DVMVoid, DVMArray]>;

#endif // DALVIK_MJOLNIRTYPES_TD
