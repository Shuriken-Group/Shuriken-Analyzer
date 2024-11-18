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

def DEFAULT_WIDTH :    I32EnumAttrCase<"DEFAULT", 0>;
def WWIDTH:    I32EnumAttrCase<"WIDE", 1>;
def OBJECT_WIDTH :    I32EnumAttrCase<"OBJECT", 2>;
def BOOLEAN_WIDTH :    I32EnumAttrCase<"BOOLEAN", 3>;
def BYTE_WIDTH :    I32EnumAttrCase<"BYTE", 4>;
def CHAR_WIDTH :    I32EnumAttrCase<"CHAR",5>;
def SHORT_WIDTH :    I32EnumAttrCase<"SHORT", 6>;
def WidthEnum : I32EnumAttr<"WidthEnum", "To simulate stuff like aget-object and aget-short", 
        [DEFAULT_WIDTH, WWIDTH, OBJECT_WIDTH, BOOLEAN_WIDTH, BYTE_WIDTH, CHAR_WIDTH, SHORT_WIDTH]> {
  let cppNamespace = "::mlir::shuriken::MjolnIR";
  let stringToSymbolFnName = "ConvertToEnum";
  let symbolToStringFnName = "ConvertToString";
}
/// Base class for unary operations, we specify just one
/// argument value as entry, and one value as result
class MjolnIR_UnaryOpBase<string mnemonic, list<Trait> traits = []> :
    MjolnIR_Op<mnemonic, !listconcat(traits, [Pure])>
{
    let arguments = (ins
        DVM_BasicType:$src
    );

    let results = (outs
        DVM_BasicType:$dst
    );
}

class MjolnIR_BinaryOpBase<string mnemonic, list<Trait> traits = []> :
    MjolnIR_Op<mnemonic, !listconcat(traits, [Pure])>
{
    let description = [{
        Binary operation receives two Values of DVM_BasicTypes,
        each one of these operations return only one Value.
    }];

    let arguments = (ins
        DVM_BasicType:$lhs,
        DVM_BasicType:$rhs
    );

    let results = (outs
        DVM_BasicType:$res
    );
}

def MjolnIR_Nop : MjolnIR_Op<"nop", [Pure]>
{
    let arguments = (ins
        StrAttr:$from // called method
    );

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
        DVM_BasicType:$res
    );

    let assemblyFormat = "`(` $value `)` attr-dict `:` type($res)";
}

//===----------------------------------------------------------------------===//
// Load string operations
//===----------------------------------------------------------------------===//
def MjolnIR_LoadString : MjolnIR_Op<"load-string", [Pure, ConstantLike]>
{
    let summary = "Load a String into a String object";

    let description = [{
        In order to generate a SSA value from a loaded String, we have to create
        an instruction which attribute will be the loaded string, and the return
        will be a DVMObject with the attribute 'java/lang/String'.

        Examples:

        ```mlir
        %0 = MjolnIR.load-string("ExampleString" : string) : dvmobject
        ```
    }];

    let arguments = (ins
        FlatSymbolRefAttr:$string,
        UI32Attr:$stringRef
    );

    let results = (outs
        DVM_BasicType:$res
    );

    let assemblyFormat = "`(` $string `)` attr-dict `:` type($res)";
}

//===----------------------------------------------------------------------===//
// Binary operations
//===----------------------------------------------------------------------===//
// Badumbatish: THIS IS EMPTY AS IT IS REPLACED BY MLIR's arith dialect


//===----------------------------------------------------------------------===//
// GetArrayOp
//===----------------------------------------------------------------------===//
def GetArrayOp: MjolnIR_Op<"getarray",
    [Pure]>
{
    let summary = "Get a value from the array with an index";


    let arguments = (ins
        AnyType :$dvm_array, // class of Array
        AnyType: $array_index, // reference to Array in Dalvik code
        WidthEnum : $width
    );

    let results = (outs
        DVM_BasicType               // any of the DVM Basic Types
    );
}
//===----------------------------------------------------------------------===//
// LoadFieldOp
//===----------------------------------------------------------------------===//
def LoadFieldOp : MjolnIR_Op<"loadfield",
    [ConstantLike, Pure]>
{
    let summary = "Load a field value";

    let description = [{
        In order to produces an SSA value from a symbol reference
        we have this operation that takes a symbol field name and
        returns a value with the type of the field.

        ```mlir
            %2 = MjolnIR.loadfield @MyField (3) :  () -> DVMType
        ```

        It is not possible to create a kind of generic variable so
        this will be a way to obtain a value from a non-SSA type
        since the nature of a Field is not being SSA.
    }];

    let arguments = (ins
        FlatSymbolRefAttr:$fieldName, // name of Field
        FlatSymbolRefAttr:$fieldClass, // class of Field
        UI32Attr: $fieldRef // reference to Field in Dalvik code
    );

    let results = (outs
        DVM_BasicType               // any of the DVM Basic Types
    );

    let assemblyFormat = "attr-dict $fieldClass`->`$fieldName `(` $fieldRef `)` `:` type(results)";
}

//===----------------------------------------------------------------------===//
// StoreFieldOp
//===----------------------------------------------------------------------===//
def StoreFieldOp : MjolnIR_Op<"storefield", []>
{
    let summary = "Store value in field";

    let description = [{
        In order to store a value in a field we have to represent the field
        as a Symbol, since there are not global variables. In that case we
        will obtain the SSA value that will be stored in the field, and we will
        obtain the name of the field and reference.

        ```mlir
            MjolnIR.storeField 
        ```
    }];

    let arguments = (ins
        DVM_BasicType:$value,   // value to store in the Field
        FlatSymbolRefAttr:$fieldName, // name of Field
        FlatSymbolRefAttr:$fieldClass, // class of Field
        UI32Attr: $fieldRef // reference to Field in Dalvik code
    );

    let assemblyFormat = "attr-dict $value`:`type($value)`,` $fieldClass`->`$fieldName `(`$fieldRef`)`";
}


//===----------------------------------------------------------------------===//
// MethodOp
//===----------------------------------------------------------------------===//
def MethodOp : MjolnIR_Op<"method", [
    DeclareOpInterfaceMethods<CallableOpInterface>,
    FunctionOpInterface,
    IsolatedFromAbove
    ]>
{
    let summary = "Method from Dalvik";

    let description = [{
        A mjolnir.method describes a method from a class from Dalvik, defined
        by the programmer. These will contain instructions from MjolnIR.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs
    );

    let regions = (region AnyRegion:$body);

    let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
    >];
    
    let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    // TODO: FunctionOpInterface methods are automatically declared, at least in mlir 18.
    // making these redundant, consider removing them.

    /// Returns the argument types of this function.
    // ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    // ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
    }];

    let hasCustomAssemblyFormat = 1;
    
    let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// InvokeOp
//===----------------------------------------------------------------------===//
def InvokeOp : MjolnIR_Op<"invoke", [
    DeclareOpInterfaceMethods<CallOpInterface>
]>
{
    let summary = "Invoke operation from Dalvik";

    let description = [{
        Dalvik machine has different invoke-* operations we will use just one,
        and the few corner cases will be transformed to special operations in that
        case.
    }];

    let arguments = (ins
        FlatSymbolRefAttr:$callee, // called method
        UI32Attr:$methodRef,
        UI32Attr:$isStatic,
        Variadic<DVM_BasicType>: $inputs // the inputs that must be the same
                                         // of those from the callee
    );

    // The return type must also be the same than the returned type
    // from the callee
    let results = (outs
        Optional<DVM_BasicType>
    );

    let assemblyFormat = [{
        $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
    }];

    let builders = [
        OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments, "MethodOp& ":$method)>,
        OpBuilder<(ins "MethodOp":$callee, "IntegerAttr":$methodRef, "IntegerAttr":$isStatic, CArg<"ValueRange", "{}">:$operands), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee", SymbolRefAttr::get(callee));
            $_state.addAttribute("methodRef", methodRef);
            $_state.addAttribute("isStatic", isStatic);
            $_state.addTypes(callee.getFunctionType().getResults());
        }]>,
        OpBuilder<(ins "Type":$result, "StringAttr":$callee, "IntegerAttr":$methodRef, "IntegerAttr":$isStatic, CArg<"ValueRange", "{}">:$operands), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee", SymbolRefAttr::get(callee));
            $_state.addAttribute("methodRef", methodRef);
            $_state.addAttribute("isStatic", isStatic);
            if (!mlir::isa<::mlir::shuriken::MjolnIR::DVMVoidType>(result))
                $_state.addTypes(result);
        }]>,
    ];
}

//===----------------------------------------------------------------------===//
// FallthroughOp
//===----------------------------------------------------------------------===//

def FallthroughOp : MjolnIR_Op<"fallthrough", [
    Pure, Terminator
  ]> {
  let summary = "FallThrough jump";
  let description = [{
    Fallthrough instruction is used as a fake jump for implementing the logic
    of control flow graph, two blocks must be connected by a terminator
    instruction, and a fallthrough in the dalvik graph is not any kind of jump.
    For this we need to add a fallthrough as an unconditional jump.

    Example:

    ```mlir
    ^bb2:
      %2 = call @someFn()
      MjolnIR.fallthrough ^bb3(%2 : tensor<*xf32>)
    ^bb3(%3: tensor<*xf32>):
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$destOperands);
  let successors = (successor AnySuccessor:$dest);

  let builders = [
    OpBuilder<(ins "Block *":$dest,
                   CArg<"ValueRange", "{}">:$destOperands), [{
      $_state.addSuccessors(dest);
      $_state.addOperands(destOperands);
    }]>];

  let extraClassDeclaration = [{
    void setDest(Block *block);

    /// Erase the operand at 'index' from the operand list.
    void eraseOperand(unsigned index);
  }];

  let assemblyFormat = [{
    $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// New operation
//===----------------------------------------------------------------------===//
def NewOp : MjolnIR_Op<"new">
{
    let summary = "Create a new object";

    let description = [{
        This instruction creates a SSA Value for an object
        the object will have as one of its parameter the
        name of the class.
    }];

    let results = (outs
        AnyType // object created with new
    );
}


#endif
