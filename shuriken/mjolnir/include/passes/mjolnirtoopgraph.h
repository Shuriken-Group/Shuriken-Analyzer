//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>
//
// @file mjolnirtoopgraph.h
// @brief A Pass to transform a Method from MjolnIR to an operand graph.

#ifndef MJOLNIR_TO_OP_GRAPH_HPP
#define MJOLNIR_TO_OP_GRAPH_HPP

#include <memory>

#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <llvm/Support/raw_ostream.h>

// We need to declare the Pass class
// from MLIR
namespace mlir {
class Pass;
} // namespace mlir

namespace shuriken::MjolnIR {
/**
 * Return a Pass for creating a graph from a MjolnIR method.
 * 
 * @param os output stream where to print the graph.
 * @return a Pass object to create a graph.
 */
std::unique_ptr<mlir::Pass> create_mjolnir_op_graph_pass(mlir::raw_ostream &os = llvm::errs());
}

#endif