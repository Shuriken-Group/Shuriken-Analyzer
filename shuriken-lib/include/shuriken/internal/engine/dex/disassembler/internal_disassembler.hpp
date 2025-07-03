//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include "shuriken/internal/providers/dex/dex_instructions.hpp"
#include "shuriken/sdk/dex/disassembly_constants.hpp"

namespace shuriken {
namespace dex {

// forward declaration
class DexEngine;
class EncodedMethod;

class InternalDisassembler {
private:
    DexEngine * dex_engine;
    InstructionProvider * last_instr;

    /// @brief If there's any switch in code, we will assign to some instructions
    /// the PackedSwitch or the SparswSwitch value
    /// @param instructions all the buffer with the instructions from a method.
    /// @param cache_instructions cache of instructions for avoiding searching
    /// always in the vector
    void assign_switch_if_any(
            std::list<std::unique_ptr<InstructionProvider>> &instructions,
            std::unordered_map<std::uint64_t, InstructionProvider *> &cache_instructions);
public:
    InternalDisassembler(DexEngine * dex_engine);

    InternalDisassembler() = default;
    ~InternalDisassembler() = default;

    /// @brief Get an instruction object from the op
    /// @param opcode op code of the instruction to return
    /// @param bytecode reference to the bytecode for disassembly
    /// @param index index of the current instruction to analyze
    /// @return unique pointer to the disassembled Instruction
    std::unique_ptr<InstructionProvider> disassemble_instruction(
            disassembler::opcodes opcode,
            std::span<uint8_t> bytecode,
            std::size_t index);

    /// @brief Determine given the last instruction the next instruction
    /// to run, the bytecode is retrieved from a :class:EncodedMethod.
    /// The offsets are calculated in number of bytes from the start of the
    /// method. Note, the offsets inside the bytecode are denoted in 16 bits
    /// units but method returns actual byte offsets.
    /// @param instruction instruction to obtain the next instructions
    /// @param curr_idx Current idx to calculate the newer one
    /// @return list of different offsets where code can go after the current
    /// instruction. Instructions like `if` or `switch` have more than one
    /// target, but `throw`, `return` and `goto` have just one. If entered
    /// opcode is not a branch instruction, next instruction is returned.
    std::vector<std::int64_t> determine_next(InstructionProvider *instruction,
                                             std::uint64_t curr_idx);

    /// @brief Same as the other `determine_next` but the instruction we give
    /// is the instruction `last_instr` that Disassembler stores.
    /// @param curr_idx Current idx to calculate the newer one
    /// @return list of different offsets where code can go after the current
    /// instruction. Instructions like `if` or `switch` have more than one
    /// target, but `throw`, `return` and `goto` have just one. If entered
    /// opcode is not a branch instruction, next instruction is returned.
    std::vector<std::int64_t> determine_next(std::uint64_t curr_idx);

    /// @brief Given an instruction check if it is a conditional jump
    /// and retrieve in that case the target of the jump
    /// @param instr instruction to retrieve the target of the jump
    /// @return target of a conditional jump
    std::int16_t get_conditional_jump_target(InstructionProvider *instr);

    /// @brief Given an instruction check if it is an unconditional jump
    /// and retrieve in that case the target of the jump
    /// @param instr instruction to retrieve the target of the jump
    /// @return target of an unconditional jump
    std::int32_t get_unconditional_jump_target(InstructionProvider *instr);

    /// @brief Retrieve information from possible exception code inside
    /// of a method
    /// @param method method to extract exception data
    /// @return exception data in a vector
    std::vector<disassembler::exception_data_t> determine_exception(EncodedMethod *method);

    std::list<std::unique_ptr<InstructionProvider>>
    disassemble(std::span<std::uint8_t> buffer_bytes);
};

}
}