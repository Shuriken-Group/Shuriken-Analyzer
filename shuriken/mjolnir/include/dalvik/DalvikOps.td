#ifndef DALVIK_DALVIKOPS_TD
#define DALVIK_DALVIKOPS_TD
include "DalvikDialect.td"

include "mlir/IR/OpBase.td"
include "mlir/Bytecode/BytecodeOpInterface.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

/// Base class for unary operations, we specify just one
/// argument value as entry, and one value as result
class Dalvik_UnaryOpBase<string mnemonic, list<Trait> traits = []> :
    Dalvik_Op<mnemonic, !listconcat(traits, [Pure])>
{
    let arguments = (ins
        AnyType:$src
    );

    let results = (outs
        AnyType:$dst
    );
}

class Dalvik_BinaryOpBase<string mnemonic, list<Trait> traits = []> :
    Dalvik_Op<mnemonic, !listconcat(traits, [Pure])>
{
    let description = [{
        Binary operation receives two Values of AnyTypes,
        each one of these operations return only one Value.
    }];

    let arguments = (ins
        AnyType:$lhs,
        AnyType:$rhs
    );

    let results = (outs
        AnyType:$res
    );
}

def Dalvik_Nop : Dalvik_Op<"nop", [Pure]>
{
    let assemblyFormat = "attr-dict";
}

def Dalvik_Instr_00x : Dalvik_Op<"Instr_00x", [Pure]> {}

def Dalvik_Instr_10x : Dalvik_Op<"Instr_10x", [Pure]> {}

def Dalvik_Instr_12x : Dalvik_Op<"Instr_12x", [Pure]> {}
def Dalvik_Instr_11n : Dalvik_Op<"Instr_11n", [Pure]> {}

def Dalvik_Instr_11x : Dalvik_Op<"Instr_11x", [Pure]> {}
def Dalvik_Instr_10t : Dalvik_Op<"Instr_10t", [Pure]> {}

def Dalvik_Instr_20t : Dalvik_Op<"Instr_20t", [Pure]> {}

def Dalvik_Instr_20bc : Dalvik_Op<"Instr_20bc", [Pure]> {}
def Dalvik_Instr_22x : Dalvik_Op<"Instr_22x", [Pure]> {}
def Dalvik_Instr_21t : Dalvik_Op<"Instr_21t", [Pure]> {}

def Dalvik_Instr_21h : Dalvik_Op<"Instr_21h", [Pure]> {}
def Dalvik_Instr_21c : Dalvik_Op<"Instr_21c", [Pure]> {}

def Dalvik_Instr_23x : Dalvik_Op<"Instr_23x", [Pure]> {}
def Dalvik_Instr_22b : Dalvik_Op<"Instr_22b", [Pure]> {}
def Dalvik_Instr_22t : Dalvik_Op<"Instr_22t", [Pure]> {}
def Dalvik_Instr_22s : Dalvik_Op<"Instr_22s", [Pure]> {}
def Dalvik_Instr_22c : Dalvik_Op<"Instr_22c", [Pure]> {}
def Dalvik_Instr_22cs : Dalvik_Op<"Instr_22cs", [Pure]> {}
def Dalvik_Instr_30t : Dalvik_Op<"Instr_30t", [Pure]> {}
def Dalvik_Instr_32x : Dalvik_Op<"Instr_32x", [Pure]> {}
def Dalvik_Instr_31i : Dalvik_Op<"Instr_31i", [Pure]> {}
def Dalvik_Instr_31t : Dalvik_Op<"Instr_31t", [Pure]> {}
def Dalvik_Instr_31c : Dalvik_Op<"Instr_31c", [Pure]> {}
def Dalvik_Instr_35c : Dalvik_Op<"Instr_35c", [Pure]> {}
def Dalvik_Instr_35ms : Dalvik_Op<"Instr_35ms", [Pure]> {}
def Dalvik_Instr_35mi : Dalvik_Op<"Instr_35mi", [Pure]> {}
def Dalvik_Instr_3rc : Dalvik_Op<"Instr_3rc", [Pure]> {}
def Dalvik_Instr_3rms : Dalvik_Op<"Instr_3rms", [Pure]> {}
def Dalvik_Instr_3rmi : Dalvik_Op<"Instr_3rmi", [Pure]> {}
def Dalvik_Instr_45cc : Dalvik_Op<"Instr_45cc", [Pure]> {}
def Dalvik_Instr_4rcc : Dalvik_Op<"Instr_4rcc", [Pure]> {}
def Dalvik_Instr_00x : Dalvik_Op<"Instr_00x", [Pure]> {}
#endif
