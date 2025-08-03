//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include "shuriken/sdk/dex/control_flow_graph.hpp"
#include "shuriken/sdk/dex/basic_blocks.hpp"
#include "basic_blocks_impl.hpp"

#include <algorithm>

namespace shuriken::dex {

class ControlFlowGraph::Impl {
private:
    using connected_blocks_t = std::unordered_map<
            DVMBasicBlock*,
            std::set<DVMBasicBlock *>>;

    std::vector<std::unique_ptr<DVMBasicBlock>> node_ownership;

    nodes_t nodes_;

    connected_blocks_t predecessors_;

    connected_blocks_t successors_;

    edges_t edges_;

    std::string basic_blocks_string;
public:
    Impl() = default;
    ~Impl() = default;

    iterator_range<nodesiterator_t> nodes() {
        return make_range(nodes_.begin(), nodes_.end());
    }

    iterator_range<edgesiterator_t> edges() {
        return make_range(edges_.begin(), edges_.end());
    }

    iterator_range<nodesetiterator_t> successors(DVMBasicBlock *node) {
        if (successors_.contains(node))
            return make_range(successors_[node].begin(),
                              successors_[node].end());
        nodesetiterator_t empty_begin{};
        nodesetiterator_t empty_end{};
        return make_range(empty_begin, empty_end);
    }

    iterator_range<nodesetiterator_t> predecessors(DVMBasicBlock *node) {
        if (predecessors_.contains(node))
            return make_range(predecessors_[node].begin(),
                              predecessors_[node].end());
        nodesetiterator_t empty_begin{};
        nodesetiterator_t empty_end{};
        return make_range(empty_begin, empty_end);
    }

    size_t get_number_of_basic_blocks() const {
        return nodes_.size();
    }

    /// Add methods, do not expose it

    void add_predecessor(DVMBasicBlock * node, DVMBasicBlock * pred) {
        predecessors_[node].insert(pred);
    }

    void add_successor(DVMBasicBlock * node, DVMBasicBlock * suc) {
        successors_[node].insert(suc);
    }

    void add_node(DVMBasicBlock *node) {
        if (std::find(nodes_.begin(),
                      nodes_.end(),
                      node) == nodes_.end()) {
            nodes_.push_back(node);
            std::unique_ptr<DVMBasicBlock> ownership(node);
            node_ownership.push_back(std::move(ownership));
        }
    }

    void add_edge(DVMBasicBlock * src, DVMBasicBlock * dst) {
        add_node(src);
        add_node(dst);

        auto edge_pair = std::make_pair(src, dst);
        /// check if edge already exists
        auto it = std::find_if(edges_.begin(), edges_.end(), [&](std::pair<DVMBasicBlock *, DVMBasicBlock *> &edge) {
            return (edge_pair.first == edge.first) && (edge_pair.second == edge.second);
        });

        /// if not, add it
        if (it == edges_.end()) {
            edges_.push_back(edge_pair);
            /// now add the successors and predecessors
            add_successor(src, dst);
            add_predecessor(dst, src);
        }
    }

    DVMBasicBlock *get_basic_block_by_idx(std::uint64_t idx) {
        auto it = std::find_if(nodes_.begin(), nodes_.end(), [&](DVMBasicBlock *bb) -> bool {
            return idx >= bb->get_first_address() && idx < bb->get_last_address();
        });

        if (it == nodes_.end()) return nullptr;
        return *it;
    }

    std::string toString() {
        if (basic_blocks_string.empty()) {
            std::stringstream ss;
            for (DVMBasicBlock *dvmBasicBlock: nodes_) {
                ss << dvmBasicBlock->to_string();
                ss << "Predecessors: ";
                for (DVMBasicBlock *pred: predecessors_[dvmBasicBlock]) {
                    ss << pred->get_name() << " ";
                }
                ss << "\nSuccessors: ";
                for (DVMBasicBlock *succ: successors_[dvmBasicBlock]) {
                    ss << succ->get_name() << " ";
                }
                ss << "\n\n";
            }
            basic_blocks_string = ss.str();
        }
        return basic_blocks_string;
    }
};
}