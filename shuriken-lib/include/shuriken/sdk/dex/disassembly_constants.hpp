//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

namespace shuriken {
namespace dex {

namespace disassembler {

enum class opcodes {
#define OPCODE(ID, VAL) \
ID = VAL,
#include "shuriken/sdk/dex/definitions/dvm_opcodes.def"
};

/// @brief Type of dex instruction, this can be used to check
/// what kind of instruction is the current one, in order to avoid
/// using dynamic casting
enum class dexinsttype {
    DEX_INSTRUCTION00X,
    DEX_INSTRUCTION10X,
    DEX_INSTRUCTION12X,
    DEX_INSTRUCTION11N,
    DEX_INSTRUCTION11X,
    DEX_INSTRUCTION10T,
    DEX_INSTRUCTION20T,
    DEX_INSTRUCTION20BC,
    DEX_INSTRUCTION22X,
    DEX_INSTRUCTION21T,
    DEX_INSTRUCTION21S,
    DEX_INSTRUCTION21H,
    DEX_INSTRUCTION21C,
    DEX_INSTRUCTION23X,
    DEX_INSTRUCTION22B,
    DEX_INSTRUCTION22T,
    DEX_INSTRUCTION22S,
    DEX_INSTRUCTION22C,
    DEX_INSTRUCTION22CS,
    DEX_INSTRUCTION30T,
    DEX_INSTRUCTION32X,
    DEX_INSTRUCTION31I,
    DEX_INSTRUCTION31T,
    DEX_INSTRUCTION31C,
    DEX_INSTRUCTION35C,
    DEX_INSTRUCTION3RC,
    DEX_INSTRUCTION45CC,
    DEX_INSTRUCTION4RCC,
    DEX_INSTRUCTION51L,
    DEX_PACKEDSWITCH,
    DEX_SPARSESWITCH,
    DEX_FILLARRAYDATA,
    DEX_DALVIKINCORRECT,
    DEX_NONE_OP = 99,
};

/// @brief Type of operands for the opcodes
enum operand_type {
    REGISTER = 0,//! register operand
    LITERAL = 1, //! literal value
    RAW = 2,     //! raw value
    OFFSET = 3,  //! offset value
    KIND = 0x100,//! used together with others
};

/// @brief Identify different type of operations
/// from instructions like branching, break, write
/// or read.
enum operation_type {
    CONDITIONAL_BRANCH_DVM_OPCODE = 0,//! conditional branch instructions ["throw", "throw.", "if."]
    UNCONDITIONAL_BRANCH_DVM_OPCODE,  //! unconditional branch instructions ["goto", "goto."]
    RET_BRANCH_DVM_OPCODE,            //! return instructions ["return", "return."]
    MULTI_BRANCH_DVM_OPCODE,          //! multi branching (switch) ["packed-switch$", "sparse-switch$"]
    CALL_DVM_OPCODE,                  //! call an external or internal method ["invoke", "invoke."]
    DATA_MOVEMENT_DVM_OPCODE,         //! move data instruction ["move", "move."]
    FIELD_READ_DVM_OPCODE,            //! read a field instruction [".get"]
    FIELD_WRITE_DVM_OPCODE,           //! write a field instruction [".put"]
    NONE_OPCODE = 99                  //!
};

/// @brief Identify the kind of argument inside of a Dalvik instruction
enum kind {
    METH = 0,         //! method reference
    STRING = 1,       //! string index
    FIELD = 2,        //! field reference
    TYPE = 3,         //! type reference
    PROTO = 9,        //! prototype reference
    METH_PROTO = 10,  //! method reference and proto reference
    CALL_SITE = 11,   //! call site item
    VARIES = 4,       //!
    INLINE_METHOD = 5,//! inlined method
    VTABLE_OFFSET = 6,//! static linked
    FIELD_OFFSET = 7, //! offset of a field (not reference)
    RAW_STRING = 8,   //!
    NONE_KIND = 99,   //!
};

} //! namespace disassembler
} //! namespace dex
} //! namespace shuriken