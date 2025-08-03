//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include "shuriken/sdk/dex/basic_blocks.hpp"
#include "shuriken/sdk/dex/instruction.hpp"

#include <sstream>
#include <iomanip>

namespace shuriken::dex {

class DVMBasicBlock::Impl {
private:
    /// span of instructions from the disassembled instructions
    /// of a method
    std::list<std::reference_wrapper<Instruction>> instructions_;

    /// boolean indicating if the current block is a try block
    bool try_block = false;
    /// in case it is a try block, it can have catch blocks
    std::set<DVMBasicBlock *> catch_blocks;

    /// boolean indicating if the block is a catch block
    bool catch_block = false;

    /// some catch blocks at the end of the code will be empty
    bool is_empty_block = false;

    /// first and last address from the block
    std::uint64_t first_address = 0;
    std::uint64_t last_address = 0;

    /// a catch block can have multiple handler types
    std::set<DVMType *> handler_types;

    /// name of the block composed by first and last address
    std::string name;

    std::string block_string;

    /// Parent method where the basic block exists
    Method * parent;
public:
    Impl(std::list<std::reference_wrapper<Instruction>> instructions_) : instructions_(std::move(instructions_)) {

    }

    Impl(std::uint64_t first_address, std::uint64_t last_address) :
        is_empty_block(true), first_address(first_address), last_address(last_address) {
    }

    ~Impl() = default;

    void set_parent_method(Method * parent) {
        this->parent = parent;
    }

    Method * get_parent() {
        return parent;
    }

    size_t get_number_of_instructions() const {
        return instructions_.size();
    }

    std::list<std::reference_wrapper<Instruction>> & block_instructions() {
        return instructions_;
    }

    Instruction * get_terminator() {
        if (instructions_.empty()) return nullptr;
        return &(instructions_.back().get());
    }

    std::uint64_t get_first_address() const {
        if (is_empty_block) return first_address;
        if (instructions_.empty()) return static_cast<std::uint64_t>(-1);
        return instructions_.front().get().get_address();
    }

    std::uint64_t get_last_address() const {
        if (is_empty_block) return last_address;
        if (instructions_.empty()) return static_cast<std::uint64_t>(-1);
        return (instructions_.back().get().get_address() + instructions_.back().get().get_instruction_length());
    }

    std::string_view get_name() {
        if (!name.empty()) return name;
        name = "BB.";
        name += std::to_string(get_first_address()) + "-";
        name += std::to_string(get_last_address());
        return name;
    }

    bool is_try_block() {
        return try_block;
    }

    bool is_catch_block() {
        return catch_block;
    }

    std::set<DVMBasicBlock *> & get_catch_blocks() {
        return catch_blocks;
    }

    std::set<DVMType *> get_handlers() {
        return handler_types;
    }

    // Setters which are not present in the SDK exposed
    // to the user
    void set_is_try_block(bool try_block) {
        this->try_block = try_block;
    }

    void set_is_catch_block(bool catch_block) {
        this->catch_block = catch_block;
    }

    void add_catch_block(DVMBasicBlock * catch_block) {
        this->catch_blocks.insert(catch_block);
    }

    void add_handler(DVMType * handler) {
        this->handler_types.insert(handler);
    }

    std::string_view to_string() {
        if (block_string.empty()) {
            std::stringstream ss;
            ss << get_name().data() << '\n';
            if (is_try_block()) {
                ss << ".try_block ";

                for (DVMBasicBlock *basicBlock: catch_blocks) {
                    ss << " catch-block "
                       << basicBlock->get_name();

                    if (!basicBlock->get_handlers().empty()) {
                        ss << "(";
                        for (const auto *type: basicBlock->get_handlers())
                            ss << get_dalvik_format(*type) << "|";
                        ss.seekp(-1, std::ios::cur);
                        ss << ")";
                    }
                }

                ss << '\n';
            } else if (is_catch_block())
                ss << ".catch_block" << '\n';
            for (auto & insn: instructions_) {
                ss << std::hex << std::setw(8)
                   << std::setfill('0')
                   << insn.get().get_address()
                   << ' ' << insn.get().print_instruction() << '\n';
            }
            block_string = ss.str();
        }
        return block_string;
    }
};

}