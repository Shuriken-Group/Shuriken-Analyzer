//
// Created by fare9 on 10/03/25.
//


#ifndef SHURIKENPROJECT_DEX_INSTRUCTION_UTILS_H
#define SHURIKENPROJECT_DEX_INSTRUCTION_UTILS_H

#include "shuriken/common/Dex/dex_opcodes.h"
#include <iostream>

namespace shuriken::disassembler::dex {

    /// Declaration for using it in InstructionUtils
    class Instruction;

    class InstructionUtils {
    public:
        /// @brief Get the operation type from the given opcode
        /// @return operation type
        static dex_opcodes::operation_type get_operation_type_from_opcode(dex_opcodes::opcodes opcode);

        /// @brief Get operation type from a given instruction
        /// @param instr instruction to retrieve the operation type
        /// @return operation type
        static dex_opcodes::operation_type get_operation_type_from_instruction(Instruction *instr);

        /// @return if operation type is a jump of any type unconditional, conditional, switch
        static bool is_jump_instruction(Instruction *instr);
    };

}

#endif//SHURIKENPROJECT_DEX_INSTRUCTION_UTILS_H
