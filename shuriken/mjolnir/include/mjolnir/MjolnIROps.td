//------------------------------------------------------------------- -*- tablegen -*-
// Shuriken: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//         Jasmine Tang <tanghocle456@gmail.com>
//
// @file MjolnIROps.td
// @brief Operations from Dalvik instructions

#ifndef DALVIK_MJOLNIROPS_TD
#define DALVIK_MJOLNIROPS_TD

include "MjolnIRTypes.td"
include "MjolnIRDialect.td"

include "mlir/IR/OpBase.td"
include "mlir/Bytecode/BytecodeOpInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"


// TODO: Should we restrict AnyType to at least MjolnirType

// TODO: Should we put pure on everything
class MjolnIR_UnaryOpBase<string mnemonic, list<Trait> traits = []> :
  MjolnIR_Op<mnemonic, !listconcat(traits, [Pure])>
  {
  let arguments = (ins AnyType:$src);
  let results = (outs AnyType:$dst);
  }


class MjolnIR_BinaryOpBase<string mnemonic, list<Trait> traits = []> :
  MjolnIR_Op<mnemonic, !listconcat(traits, [Pure])> 
  {

  }

class MjolnIR_ComparisonOpBase<string mnemonic, list<Trait> traits = []> :
    MjolnIR_Op<mnemonic, !listconcat(traits, [Pure])>
{
    let description = [{
        Different type of comparisons in Dalvik, the types of the arguments
        can be any of the Dalvik Machine, but always as result the operation
        returns a bool value for ControlFlowOps Dialect (I1 value)
    }];

    let arguments = (ins
        AnyType:$lhs,
        AnyType:$rhs
    );

    let results = (outs
        I1:$res
    );
}

def MjolnIR_Nop : MjolnIR_Op<"nop", [Pure]>
{
    let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//
def MjolnIR_MoveOp : MjolnIR_UnaryOpBase<"move">;
def MjolnIR_CastOp : MjolnIR_UnaryOpBase<"cast">;

def MjolnIR_Neg : MjolnIR_UnaryOpBase<"neg", [AllTypesMatch<["src", "dst"]>]>;
def MjolnIR_Not : MjolnIR_UnaryOpBase<"not", [AllTypesMatch<["src", "dst"]>]>;

//===----------------------------------------------------------------------===//
// Load values operations
//===----------------------------------------------------------------------===//
def MjolnIR_LoadValue : MjolnIR_Op<"load-value", [Pure, ConstantLike]>
{
    let summary = "Load a constant value";

    let description = [{
        In order to generate a SSA value from a constant one we have to create
        an instruction that has a value as attribute and returns a Value to any
        of the types from Dalvik, in this way the generated value is already in
        SSA value.

        Examples:

        ```mlir
        // Example of data loaded
        %0 = MjolnIR.load-value(33 : dvmint) : dvmint
        ```
    }];

    let arguments = (ins
        SI64Attr:$value
    );

    let results = (outs
        AnyType:$res
    );

    let assemblyFormat = "`(` $value `)` attr-dict `:` type($res)";
}
//
////===----------------------------------------------------------------------===//
//// Load string operations
////===----------------------------------------------------------------------===//
//def MjolnIR_LoadString : MjolnIR_Op<"load-string", [Pure, ConstantLike]>
//{
//    let summary = "Load a String into a String object";
//
//    let description = [{
//        In order to generate a SSA value from a loaded String, we have to create
//        an instruction which attribute will be the loaded string, and the return
//        will be a DVMObject with the attribute 'java/lang/String'.
//
//        Examples:
//
//        ```mlir
//        %0 = MjolnIR.load-string("ExampleString" : string) : dvmobject
//        ```
//    }];
//
//    let arguments = (ins
//        FlatSymbolRefAttr:$string,
//        UI32Attr:$stringRef
//    );
//
//    let results = (outs
//        AnyType:$res
//    );
//
//    let assemblyFormat = "`(` $string `)` attr-dict `:` type($res)";
//}

//===----------------------------------------------------------------------===//
// Binary operations
//===----------------------------------------------------------------------===//
def MjolnIR_AddOp : MjolnIR_BinaryOpBase<"add", [Commutative]>
{
    let summary = "Add two DVM registers";
}

def MjolnIR_SubOp : MjolnIR_BinaryOpBase<"sub">
{
    let summary = "Substract two DVM registers";
}

def MjolnIR_MulOp : MjolnIR_BinaryOpBase<"mul", [Commutative]>
{
    let summary = "Multiply two DVM registers";
}

def MjolnIR_DivOp : MjolnIR_BinaryOpBase<"div">
{
    let summary = "Divide two DVM registers";
}

def MjolnIR_RemOp : MjolnIR_BinaryOpBase<"rem">
{
    let summary = "Divide two DVM registers and get the remainder";
}

def MjolnIR_AndOp : MjolnIR_BinaryOpBase<"and">
{
    let summary = "Apply AND operation in two DVM registers";
}

def MjolnIR_OrOp  : MjolnIR_BinaryOpBase<"or">
{
    let summary = "Apply OR operation in two DVM registers";
}

def MjolnIR_XorOp : MjolnIR_BinaryOpBase<"xor">
{
    let summary = "Apply XOR operation in two DVM registers";
}

def MjolnIR_Shl   : MjolnIR_BinaryOpBase<"shl">
{
    let summary = "Shift Left a DVM register";
}

def MjolnIR_Shr   : MjolnIR_BinaryOpBase<"shr">
{
    let summary = "Shift Right a DVM register";
}

def MjolnIR_UShr  : MjolnIR_BinaryOpBase<"ushr">
{
    let summary = "Unsigned Shift Right a DVM register";
}

//===----------------------------------------------------------------------===//
// Comparison Operations
//===----------------------------------------------------------------------===//

def MjolnIR_CmpEq : MjolnIR_ComparisonOpBase<"cmp-eq", [Commutative]>
{
    let summary = "Compare if values are equal";
}

def MjolnIR_CmpNEq : MjolnIR_ComparisonOpBase<"cmp-neq", [Commutative]>
{
    let summary = "Compare if values are not equal";
}

def MjolnIR_CmpLt : MjolnIR_ComparisonOpBase<"cmp-lt">
{
    let summary = "Compare a value is lower than other";
}

def MjolnIR_CmpGe : MjolnIR_ComparisonOpBase<"cmp-ge">
{
    let summary = "Compare a value is greater or equal than other";
}

def MjolnIR_CmpGt : MjolnIR_ComparisonOpBase<"cmp-gt">
{
    let summary = "Compare a value is greater than other";
}

def MjolnIR_CmpLe : MjolnIR_ComparisonOpBase<"cmp-le">
{
    let summary = "Compare a value is lower or equal than other";
}

def MjolnIR_CmpEqz : MjolnIR_ComparisonOpBase<"cmp-eqz">
{
    let summary = "Compare a value is equal to zero";
}

def MjolnIR_CmpNeqz : MjolnIR_ComparisonOpBase<"cmp-neqz">
{
    let summary = "Compare a value is not equal to zero";
}

def MjolnIR_CmpLtz : MjolnIR_ComparisonOpBase<"cmp-ltz">
{
    let summary = "Compare a value is lower than zero";
}

def MjolnIR_CmpGez : MjolnIR_ComparisonOpBase<"cmp-gez">
{
    let summary = "Compare a value is greater or equal to zero";
}

def MjolnIR_CmpGtz : MjolnIR_ComparisonOpBase<"cmp-gtz">
{
    let summary = "Compare a value is greater than zero";
}

def MjolnIR_CmpLez : MjolnIR_ComparisonOpBase<"cmp-lez">
{
    let summary = "Compare a value is lower or equal to zero";
}

#endif
