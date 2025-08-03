//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include "shuriken/internal/sdk/dex/basic_blocks_impl.hpp"

using namespace shuriken::dex;

DVMBasicBlock::DVMBasicBlock(shuriken::dex::DVMBasicBlock::Impl *impl) :
    impl(impl) {
}

size_t DVMBasicBlock::get_number_of_instructions() const {
    return impl->get_number_of_instructions();
}

std::list<std::reference_wrapper<Instruction>> & DVMBasicBlock::block_instructions() {
    return impl->block_instructions();
}

Instruction *DVMBasicBlock::get_terminator() {
    return impl->get_terminator();
}

std::uint64_t DVMBasicBlock::get_first_address() const {
    return impl->get_first_address();
}

std::uint64_t DVMBasicBlock::get_last_address() const {
    return impl->get_last_address();
}

std::string_view DVMBasicBlock::get_name() {
    return impl->get_name();
}

bool DVMBasicBlock::is_try_block() {
    return impl->is_try_block();
}

bool DVMBasicBlock::is_catch_block() {
    return impl->is_catch_block();
}

std::set<DVMBasicBlock *> &DVMBasicBlock::get_catch_blocks() {
    return impl->get_catch_blocks();
}

std::set<DVMType *> DVMBasicBlock::get_handlers() {
    return impl->get_handlers();
}

std::string_view DVMBasicBlock::to_string() {
    return impl->to_string();
}

Method *DVMBasicBlock::get_parent() {
    return impl->get_parent();
}
