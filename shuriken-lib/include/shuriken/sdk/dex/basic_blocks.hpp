//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include "shuriken/sdk/dex/custom_types.hpp"

#include <memory>

namespace shuriken {
namespace dex {

class Method;
class Instruction;

class DVMBasicBlock {
public:
    class Impl;
private:
    std::unique_ptr<Impl> impl;
public:
    DVMBasicBlock(Impl*);

    ~DVMBasicBlock() = default;

    /// avoid any kind of copy constructor
    DVMBasicBlock(const DVMBasicBlock &temp_obj) = delete;
    DVMBasicBlock &operator=(const DVMBasicBlock &temp_obj) = delete;

    Method * get_parent();

    size_t get_number_of_instructions() const;

    std::list<std::reference_wrapper<Instruction>> & block_instructions();

    Instruction * get_terminator();

    std::uint64_t get_first_address() const;

    std::uint64_t get_last_address() const;

    std::string_view get_name();

    bool is_try_block();

    bool is_catch_block();

    std::set<DVMBasicBlock *> & get_catch_blocks();

    std::set<DVMType *> get_handlers();

    std::string_view to_string();
};

}
}