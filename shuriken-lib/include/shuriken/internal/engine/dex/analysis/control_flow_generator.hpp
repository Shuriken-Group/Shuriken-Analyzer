//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include "shuriken/sdk/dex/method.hpp"
#include "shuriken/internal/sdk/dex/control_flow_graph_impl.hpp"
#include "shuriken/sdk/dex/control_flow_graph.hpp"

namespace shuriken::dex {

class ControlFlowGeneratorPass {
public:
    ControlFlowGeneratorPass() = default;
    ~ControlFlowGeneratorPass() = default;

    /// Generate control flow graph for the given method
    /// @param method_impl The method implementation to analyze
    /// @return Unique pointer to the generated control flow graph
    std::unique_ptr<ControlFlowGraph> generate_control_flow_graph(Method::Impl* method_impl);

private:
    /// Create basic blocks from instruction sequence
    /// @param method_impl The method implementation containing instructions
    void create_basic_blocks(Method::Impl* method_impl);

    /// Build edges between basic blocks based on control flow
    /// @param method_impl The method implementation
    void build_control_flow_edges(Method::Impl* method_impl);

    /// Handle exception flow in the control flow graph
    /// @param method_impl The method implementation
    void handle_exception_flow(Method::Impl* method_impl);

    /// Find basic block leaders (first instruction of each block)
    /// @param method_impl The method implementation
    /// @return Set of instruction addresses that start basic blocks
    std::set<std::uint64_t> find_basic_block_leaders(Method::Impl* method_impl);

    /// Create a basic block starting at the given address
    /// @param start_address Starting address of the basic block
    /// @param end_address Ending address of the basic block
    /// @param method_impl The method implementation
    /// @return Pointer to the created basic block
    DVMBasicBlock* create_basic_block(std::uint64_t start_address, std::uint64_t end_address, Method::Impl* method_impl);

    /// Temporary storage for control flow graph being built
    std::unique_ptr<ControlFlowGraph> current_cfg_;
    ControlFlowGraph::Impl * current_cfg_impl;

    /// Map from instruction address to basic block
    std::unordered_map<std::uint64_t, DVMBasicBlock*> address_to_block_;
    std::unordered_map<DVMBasicBlock*, DVMBasicBlock::Impl*> block_to_implementation_;
};

}