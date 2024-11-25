include "mlir/Dialect/Complex/IR/ComplexOps.td"
include "mlir/IR/PatternBase.td"
include "DalvikOps.td"
include "mjolnir/MjolnIROps.td"
include "mlir/Dialect/Arith/IR/ArithOps.td"

def HasOneUse: Constraint<CPred<"$_self.hasOneUse()">, "has one use">;

def SimplePat : Pat<
(MjolnIR_AddOp $lhs, (arith.constant $rhs)),
(Dalvik_Instr_22s $final)
  >;
