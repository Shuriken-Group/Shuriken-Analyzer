//------------------------------------------------------------------- -*- tablegen -*-
// Shuriken: library for doing analysis of dalvik files
// @author Farenain <kunai.static.analysis@gmail.com>
//         Jasmine Tang <tanghocle456@gmail.com>
//
// @file MjolnIROps.td
// @brief Operations from Dalvik instructions

#ifndef DALVIK_MJOLNIROPS_TD
#define DALVIK_MJOLNIROPS_TD

include "MjolnIR/Dalvik/MjolnIRTypes.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"


// TODO: Should we restrict AnyType to at least MjolnirType

class MjolnIR_UnaryOpBase<string mnemonic, list<Trait> traits = []> :
  MjolnIR_Op<mnemonic, !listconcat(traits, [Pure])>
  {
  let arguments = (ins AnyType:$src)
  let results = (outs AnyType:$dst)
  }


class MjolnIR_BinaryOpBase<string mnemonic, list<Trait> traits = []> :
  MjolnIR_Op<mnemonic, !listconcat(traits, [Pure])> 
  {

  }
