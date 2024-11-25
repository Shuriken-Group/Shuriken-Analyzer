include "mlir/Dialect/Complex/IR/ComplexOps.td"
include "mlir/IR/PatternBase.td"
include "DalvikOps.td"
include "mjolnir/MjolnIROps.td"


def HasOneUse: Constraint<CPred<"$_self.hasOneUse()">, "has one use">;
