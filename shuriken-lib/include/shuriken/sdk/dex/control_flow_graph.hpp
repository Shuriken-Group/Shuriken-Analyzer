//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include "shuriken/sdk/dex/custom_types.hpp"
#include "shuriken/sdk/common/iterator_range.hpp"

#include <memory>
#include <unordered_map>
#include <set>

namespace shuriken {
namespace dex {

class DVMBasicBlock;

class ControlFlowGraph {
public:
    using nodes_t = std::vector<DVMBasicBlock *>;

    using edges_t = std::vector<std::pair<DVMBasicBlock*, DVMBasicBlock*>>;

    /// @brief Iterator for going through a list of basic blocks in order
    using nodesiterator_t = nodes_t::iterator;

    /// @brief Iterator for going throw a list of successors or predecessors
    using nodesetiterator_t = std::set<DVMBasicBlock *>::iterator;

    /// @brief Iterator for going through the edges
    using edgesiterator_t = edges_t::iterator;

    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    ControlFlowGraph(Impl*);

    ~ControlFlowGraph() = default;

    ControlFlowGraph(const ControlFlowGraph&) = delete;
    ControlFlowGraph &operator=(const ControlFlowGraph&) = delete;

    /// @return iterator to all the nodes
    iterator_range<nodesiterator_t> nodes();

    /// @return iterator to the edges
    iterator_range<edgesiterator_t> edges();

    /// @brief Check if successors exist for a node and return its successors
    /// @param node to get its successors
    /// @return successors from provided node
    iterator_range<nodesetiterator_t> successors(DVMBasicBlock *node);

    /// @brief Check if predecessors exist for a node and return its successors
    /// @param node to get its predecessors
    /// @return predecessors from provided node
    iterator_range<nodesetiterator_t> predecessors(DVMBasicBlock *node);

    /// @brief Return the number of basic blocks in the graph
    /// @return number of basic blocks
    size_t get_number_of_basic_blocks() const;

    /// @brief Get a basic block given an idx, the idx can be one
    /// address from the first to the last address of the block
    /// @param idx address of the block to retrieve
    /// @return block that contains an instruction in that address
    DVMBasicBlock *get_basic_block_by_idx(std::uint64_t idx);

    std::string toString();
};

}
}