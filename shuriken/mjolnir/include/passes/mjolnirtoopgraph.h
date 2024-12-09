//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file mjolnirtoopgraph.h
// @brief A Pass to transform a Method from MjolnIR to an operand graph.

#pragma once

#include <memory>

#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>

namespace shuriken::MjolnIR {
    /**
       * Return a Pass for creating a graph from a MjolnIR method.
       * 
       * @param os output stream where to print the graph.
       * @return a Pass object to create a graph.
    */
    std::unique_ptr<mlir::Pass> create_mjolnir_op_graph_pass(mlir::raw_ostream &os = llvm::errs());

    mlir::LogicalResult generate_functions_cfg(mlir::raw_ostream &os, mlir::OwningOpRef<mlir::ModuleOp> &module);
}// namespace shuriken::MjolnIR
