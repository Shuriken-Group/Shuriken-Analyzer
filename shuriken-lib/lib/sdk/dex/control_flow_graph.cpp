//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include "shuriken/internal/sdk/dex/control_flow_graph_impl.hpp"


using namespace shuriken::dex;


ControlFlowGraph::ControlFlowGraph(ControlFlowGraph::Impl *impl) :
    impl(impl) {
}

shuriken::iterator_range<ControlFlowGraph::nodesiterator_t> ControlFlowGraph::nodes() {
    return impl->nodes();
}

shuriken::iterator_range<ControlFlowGraph::edgesiterator_t> ControlFlowGraph::edges() {
    return impl->edges();
}

shuriken::iterator_range<ControlFlowGraph::nodesetiterator_t> ControlFlowGraph::successors(DVMBasicBlock *node) {
    return impl->successors(node);
}

shuriken::iterator_range<ControlFlowGraph::nodesetiterator_t> ControlFlowGraph::predecessors(DVMBasicBlock *node) {
    return impl->predecessors(node);
}

size_t ControlFlowGraph::get_number_of_basic_blocks() const {
    return impl->get_number_of_basic_blocks();
}

DVMBasicBlock *ControlFlowGraph::get_basic_block_by_idx(std::uint64_t idx) {
    return impl->get_basic_block_by_idx(idx);
}

std::string ControlFlowGraph::toString() {
    return impl->toString();
}
