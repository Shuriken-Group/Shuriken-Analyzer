//------------------------------------------------------------------- -*- tablegen -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file mjolnir.td
// @brief Global definitions from MjolnIR Dialect

#ifndef SHURIKENPROJECT_MJOLNIR_TD
#define SHURIKENPROJECT_MJOLNIR_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

//===-----------------------------------------------------===//
// MjolnIR dialect definition, it will be used as the definition
// of the global namespace
//===-----------------------------------------------------===//

def MjolnIR_Dialect : Dialect
{
    // Namespace of the dialect. It is used to encapsulate the
    // sub-components of the dialect.
    let name = "MjolnIR";
    let summary = "Dialect for supporting the analysis of bytecode";
    let description = [{
        Dialect for the analysis of bytecode, right now mostly
        targeted to an analysis of Java.
    }];

    // Namespace used in the C++ code
    let cppNamespace = "mlir::shuriken::mjolnir";

    // Use the provided printer parser
    let useDefaultTypePrinterParser = 1;
    // Other declarations to add code to
    let extraClassDeclaration = [{

    }]
}

#endif // SHURIKENPROJECT_MJOLNIR_TD