//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#include <shuriken/internal/providers/dex/dex_instructions.hpp>
#include <shuriken/sdk/dex/method.hpp>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <sstream>

using namespace shuriken::dex;

namespace {
    static const std::unordered_map<disassembler::opcodes, disassembler::kind> opcodes_kind_map{
#define INST_KIND(OP, VAL) {OP, VAL},

#include "definitions/opcode_kind.def"
    };

    static const std::unordered_map<disassembler::opcodes, disassembler::operation_type> opcodes_operation_type{
#define INST_OP(OP, VAL) {OP, VAL},

#include "definitions/dvm_inst_operation.def"
    };

    static const std::unordered_map<disassembler::opcodes, std::string> opcode_names{
#define INST_NAME(OP, NAME) \
    {OP, NAME},

#include "definitions/dvm_ins_names.def"
    };

    static const std::vector<disassembler::opcodes> side_effects_opcodes{
            disassembler::opcodes::OP_RETURN_VOID,
            disassembler::opcodes::OP_RETURN,
            disassembler::opcodes::OP_RETURN_WIDE,
            disassembler::opcodes::OP_RETURN_OBJECT,
            disassembler::opcodes::OP_MONITOR_ENTER,
            disassembler::opcodes::OP_MONITOR_EXIT,
            disassembler::opcodes::OP_FILL_ARRAY_DATA,
            disassembler::opcodes::OP_THROW,
            disassembler::opcodes::OP_GOTO,
            disassembler::opcodes::OP_SPARSE_SWITCH,
            disassembler::opcodes::OP_PACKED_SWITCH,
            disassembler::opcodes::OP_IF_EQ,
            disassembler::opcodes::OP_IF_NE,
            disassembler::opcodes::OP_IF_LT,
            disassembler::opcodes::OP_IF_GE,
            disassembler::opcodes::OP_IF_GT,
            disassembler::opcodes::OP_IF_LE,
            disassembler::opcodes::OP_IF_EQZ,
            disassembler::opcodes::OP_IF_NEZ,
            disassembler::opcodes::OP_IF_LTZ,
            disassembler::opcodes::OP_IF_GEZ,
            disassembler::opcodes::OP_IF_GTZ,
            disassembler::opcodes::OP_IF_LEZ,
            disassembler::opcodes::OP_APUT,
            disassembler::opcodes::OP_APUT_WIDE,
            disassembler::opcodes::OP_APUT_OBJECT,
            disassembler::opcodes::OP_APUT_BOOLEAN,
            disassembler::opcodes::OP_APUT_BYTE,
            disassembler::opcodes::OP_APUT_CHAR,
            disassembler::opcodes::OP_APUT_SHORT,
            disassembler::opcodes::OP_IPUT,
            disassembler::opcodes::OP_IPUT_WIDE,
            disassembler::opcodes::OP_IPUT_OBJECT,
            disassembler::opcodes::OP_IPUT_BOOLEAN,
            disassembler::opcodes::OP_IPUT_BYTE,
            disassembler::opcodes::OP_IPUT_CHAR,
            disassembler::opcodes::OP_IPUT_SHORT,
            disassembler::opcodes::OP_SPUT,
            disassembler::opcodes::OP_SPUT_WIDE,
            disassembler::opcodes::OP_SPUT_OBJECT,
            disassembler::opcodes::OP_SPUT_BOOLEAN,
            disassembler::opcodes::OP_SPUT_BYTE,
            disassembler::opcodes::OP_SPUT_CHAR,
            disassembler::opcodes::OP_SPUT_SHORT,
            disassembler::opcodes::OP_INVOKE_VIRTUAL,
            disassembler::opcodes::OP_INVOKE_SUPER,
            disassembler::opcodes::OP_INVOKE_DIRECT,
            disassembler::opcodes::OP_INVOKE_STATIC,
            disassembler::opcodes::OP_INVOKE_INTERFACE,
    };

    static const std::vector<disassembler::opcodes> may_throw_opcodes{
            disassembler::opcodes::OP_CONST_STRING,
            disassembler::opcodes::OP_CONST_CLASS,
            disassembler::opcodes::OP_MONITOR_ENTER,
            disassembler::opcodes::OP_MONITOR_EXIT,
            disassembler::opcodes::OP_CHECK_CAST,
            disassembler::opcodes::OP_INSTANCE_OF,
            disassembler::opcodes::OP_ARRAY_LENGTH,
            disassembler::opcodes::OP_NEW_INSTANCE,
            disassembler::opcodes::OP_NEW_ARRAY,
            disassembler::opcodes::OP_FILLED_NEW_ARRAY,
            disassembler::opcodes::OP_AGET,
            disassembler::opcodes::OP_AGET_WIDE,
            disassembler::opcodes::OP_AGET_OBJECT,
            disassembler::opcodes::OP_AGET_BOOLEAN,
            disassembler::opcodes::OP_AGET_BYTE,
            disassembler::opcodes::OP_AGET_CHAR,
            disassembler::opcodes::OP_AGET_SHORT,
            disassembler::opcodes::OP_APUT,
            disassembler::opcodes::OP_APUT_WIDE,
            disassembler::opcodes::OP_APUT_OBJECT,
            disassembler::opcodes::OP_APUT_BOOLEAN,
            disassembler::opcodes::OP_APUT_BYTE,
            disassembler::opcodes::OP_APUT_CHAR,
            disassembler::opcodes::OP_APUT_SHORT,
            disassembler::opcodes::OP_IGET,
            disassembler::opcodes::OP_IGET_WIDE,
            disassembler::opcodes::OP_IGET_OBJECT,
            disassembler::opcodes::OP_IGET_BOOLEAN,
            disassembler::opcodes::OP_IGET_BYTE,
            disassembler::opcodes::OP_IGET_CHAR,
            disassembler::opcodes::OP_IGET_SHORT,
            disassembler::opcodes::OP_IPUT,
            disassembler::opcodes::OP_IPUT_WIDE,
            disassembler::opcodes::OP_IPUT_OBJECT,
            disassembler::opcodes::OP_IPUT_BOOLEAN,
            disassembler::opcodes::OP_IPUT_BYTE,
            disassembler::opcodes::OP_IPUT_CHAR,
            disassembler::opcodes::OP_IPUT_SHORT,
            disassembler::opcodes::OP_SGET,
            disassembler::opcodes::OP_SGET_WIDE,
            disassembler::opcodes::OP_SGET_OBJECT,
            disassembler::opcodes::OP_SGET_BOOLEAN,
            disassembler::opcodes::OP_SGET_BYTE,
            disassembler::opcodes::OP_SGET_CHAR,
            disassembler::opcodes::OP_SGET_SHORT,
            disassembler::opcodes::OP_SPUT,
            disassembler::opcodes::OP_SPUT_WIDE,
            disassembler::opcodes::OP_SPUT_OBJECT,
            disassembler::opcodes::OP_SPUT_BOOLEAN,
            disassembler::opcodes::OP_SPUT_BYTE,
            disassembler::opcodes::OP_SPUT_CHAR,
            disassembler::opcodes::OP_SPUT_SHORT,
            disassembler::opcodes::OP_INVOKE_VIRTUAL,
            disassembler::opcodes::OP_INVOKE_SUPER,
            disassembler::opcodes::OP_INVOKE_DIRECT,
            disassembler::opcodes::OP_INVOKE_STATIC,
            disassembler::opcodes::OP_INVOKE_INTERFACE,
            disassembler::opcodes::OP_DIV_INT,
            disassembler::opcodes::OP_REM_INT,
            disassembler::opcodes::OP_DIV_LONG,
            disassembler::opcodes::OP_REM_LONG,
            disassembler::opcodes::OP_DIV_INT_LIT16,
            disassembler::opcodes::OP_REM_INT_LIT16,
            disassembler::opcodes::OP_DIV_INT_LIT8,
            disassembler::opcodes::OP_REM_INT_LIT8,
    };

    std::string get_kind_type_as_string(const kind_type_t &source_id, std::uint16_t iBBBB) {
        std::stringstream instruction_str;

        if (std::holds_alternative<std::monostate>(source_id)) {
            instruction_str << " // UNKNOWN@" << iBBBB;
        } else if (std::holds_alternative<DVMType *>(source_id)) {
            auto *type = std::get<DVMType *>(source_id);
            instruction_str << shuriken::dex::get_dalvik_format(*type);
            instruction_str << " // type@" << iBBBB;
        } else if (std::holds_alternative<Field *>(source_id)) {
            auto *field = std::get<Field *>(source_id);
            instruction_str << field->get_descriptor();
            instruction_str << " // field@" << iBBBB;
        } else if (std::holds_alternative<Method *>(source_id)) {
            auto *method = std::get<Method *>(source_id);
            instruction_str << method->get_descriptor();
            instruction_str << " // method@" << iBBBB;
        } else if (std::holds_alternative<DVMPrototype *>(source_id)) {
            auto *proto = std::get<DVMPrototype *>(source_id);
            instruction_str << proto->get_shorty_idx();
            instruction_str << " // proto@" << iBBBB;
        } else if (std::holds_alternative<std::string_view>(source_id)) {
            auto str = std::get<std::string_view>(source_id);
            instruction_str << "\"" << str << "\"";
            instruction_str << " // string@" << iBBBB;
        }

        return instruction_str.str();
    }
}


InstructionProvider::InstructionProvider([[maybe_unused]] std::span<std::uint8_t> bytecode,
                                         [[maybe_unused]] std::size_t index,
                                         disassembler::dexinsttype instruction_type) :
        instruction_type(instruction_type), op_codes({}), length(0), opcode(disassembler::opcodes::OP_NONE) {
}


InstructionProvider::InstructionProvider(std::span<std::uint8_t> bytecode, std::size_t index,
                                         disassembler::dexinsttype instruction_type, std::uint32_t length)
        : instruction_type(instruction_type), op_codes({bytecode.begin() + index, bytecode.begin() + index + length}),
          length(length), opcode(disassembler::opcodes::OP_NONE) {
}

InstructionProvider::~InstructionProvider() = default;

disassembler::kind InstructionProvider::get_kind() const {
    auto it = ::opcodes_kind_map.find(opcode);

    return it != opcodes_kind_map.end() ? it->second : disassembler::kind::NONE_KIND;
}


disassembler::dexinsttype InstructionProvider::get_instruction_type() const {
    return instruction_type;
}


std::uint32_t InstructionProvider::get_instruction_length() const {
    return length;
}


disassembler::opcodes InstructionProvider::get_instruction_opcode() const {
    return opcode;
}


void InstructionProvider::set_address(std::uint64_t address) {
    this->address = address;
}


std::uint64_t InstructionProvider::get_address() const {
    return address;
}


std::span<std::uint8_t> InstructionProvider::get_instruction_bytecode() const {
    return op_codes;
}

bool InstructionProvider::is_terminator() const {
    if (!::opcodes_operation_type.contains(opcode))
        return false;
    auto operation = opcodes_operation_type.at(opcode);
    if (operation == disassembler::operation_type::CONDITIONAL_BRANCH_DVM_OPCODE
        || operation == disassembler::operation_type::UNCONDITIONAL_BRANCH_DVM_OPCODE
        || operation == disassembler::operation_type::RET_BRANCH_DVM_OPCODE
        || operation == disassembler::operation_type::MULTI_BRANCH_DVM_OPCODE)
        return true;
    return false;
}


bool InstructionProvider::has_side_effects() const {
    if (std::find(::side_effects_opcodes.begin(), ::side_effects_opcodes.end(), opcode) != side_effects_opcodes.end())
        return true;
    return false;
}


bool InstructionProvider::may_throw() const {
    if (std::find(::may_throw_opcodes.begin(), ::may_throw_opcodes.end(), opcode) != may_throw_opcodes.end())
        return true;
    return false;
}

void InstructionProvider::invalidate_instruction(std::string_view err_msg) {
    error_message = err_msg;
    is_valid = false;
}

bool InstructionProvider::is_instruction_valid() const {
    return is_valid;
}

const std::string &InstructionProvider::get_error_message() const {
    return error_message;
}

DalvikIncorrectInstructionProvider::DalvikIncorrectInstructionProvider(std::span<uint8_t> bytecode, std::size_t index,
                                                                       std::string_view error_message,
                                                                       size_t size_instr,
                                                                       std::uint64_t address,
                                                                       disassembler::opcodes opcode) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_DALVIKINCORRECT, size_instr) {
    this->address = address;
    this->error_message = error_message;
    this->opcode = opcode;
}

std::string_view DalvikIncorrectInstructionProvider::format_instruction() {
    if (instruction.empty()) {
        instruction = "DalvikIncorrectInstructionProvider [error_message=";
        instruction += error_message + ", address=";
        instruction += std::to_string(address) + ", opcode=";
        instruction += opcode_names.at(opcode) + "]";
    }
    return instruction;
}

std::string_view DalvikIncorrectInstructionProvider::print_instruction() {
    return format_instruction();
}

std::string DalvikIncorrectInstructionProvider::print_instruction_string() {
    return std::string(format_instruction());
}

Instruction00xProvider::Instruction00xProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION00X) {
}


Instruction00xProvider::Instruction00xProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION00X) {
}

std::string_view Instruction00xProvider::format_instruction() {
    if (instruction.empty())
        instruction = opcode_names.at(opcode);
    return instruction;
}

std::string_view Instruction00xProvider::print_instruction() {
    return format_instruction();
}


std::string Instruction00xProvider::print_instruction_string() {
    return std::string(format_instruction());
}

Instruction10xProvider::Instruction10xProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION10X, 2) {

    if (this->op_codes[1] != 0) {
        invalidate_instruction("Instruction10x high byte should be 0");
        return;
    }

    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
}


Instruction10xProvider::Instruction10xProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION10X, 2) {

    if (this->op_codes[1] != 0) {
        invalidate_instruction("Instruction10x high byte should be 0");
        return;
    }

    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
}

std::string_view Instruction10xProvider::format_instruction() {
    if (instruction.empty())
        instruction = opcode_names.at(opcode);
    return instruction;
}

std::string_view Instruction10xProvider::print_instruction() {
    return format_instruction();
}

std::string Instruction10xProvider::print_instruction_string() {
    return std::string(format_instruction());
}

Instruction12xProvider::Instruction12xProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION12X, 2) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vA = (this->op_codes[1] & 0x0F);
    vB = (this->op_codes[1] & 0xF0) >> 4;
}

Instruction12xProvider::Instruction12xProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION12X, 2) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vA = (this->op_codes[1] & 0x0F);
    vB = (this->op_codes[1] & 0xF0) >> 4;
}


std::uint8_t Instruction12xProvider::getVA() const {
    return vA;
}


disassembler::operand_type Instruction12xProvider::get_vA_type() const {
    return disassembler::operand_type::REGISTER;
}


std::uint8_t Instruction12xProvider::getVB() const {
    return vB;
}


disassembler::operand_type Instruction12xProvider::get_vB_types() const {
    return disassembler::operand_type::REGISTER;
}

std::string_view Instruction12xProvider::format_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vA);
        instruction += ", ";
        instruction += "v" + std::to_string(vB);
    }
    return instruction;
}

std::string_view Instruction12xProvider::print_instruction() {
    return format_instruction();
}


std::string Instruction12xProvider::print_instruction_string() {
    return std::string(format_instruction());
}

Instruction11nProvider::Instruction11nProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION11N, 2) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vA = (this->op_codes[1] & 0x0F);
    nB = static_cast<std::int8_t>((this->op_codes[1] & 0xF0) >> 4);
}

Instruction11nProvider::Instruction11nProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION11N, 2) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vA = (this->op_codes[1] & 0x0F);
    nB = static_cast<std::int8_t>((this->op_codes[1] & 0xF0) >> 4);
}

std::uint8_t Instruction11nProvider::getVA() const {
    return vA;
}


disassembler::operand_type Instruction11nProvider::get_vA_type() const {
    return disassembler::operand_type::REGISTER;
}

std::int8_t Instruction11nProvider::getNB() const {
    return nB;
}


disassembler::operand_type Instruction11nProvider::get_nB_types() const {
    return disassembler::operand_type::LITERAL;
}

std::string_view Instruction11nProvider::format_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vA);
        instruction += ", ";
        instruction += std::to_string(nB);
    }
    return instruction;
}

std::string_view Instruction11nProvider::print_instruction() {
    return format_instruction();
}


std::string Instruction11nProvider::print_instruction_string() {
    return std::string(format_instruction());
}

Instruction11xProvider::Instruction11xProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION11X, 2) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vAA = this->op_codes[1];
}


Instruction11xProvider::Instruction11xProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION11X, 2) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vAA = this->op_codes[1];
}


std::uint8_t Instruction11xProvider::getVAA() const {
    return vAA;
}


disassembler::operand_type Instruction11xProvider::get_vAA_type() const {
    return disassembler::operand_type::REGISTER;
}

std::string_view Instruction11xProvider::format_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vAA);
    }
    return instruction;
}

std::string_view Instruction11xProvider::print_instruction() {
    return format_instruction();
}


std::string Instruction11xProvider::print_instruction_string() {
    return std::string(format_instruction());
}

Instruction10tProvider::Instruction10tProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION10T, 2) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    nAA = static_cast<std::int8_t>(this->op_codes[1]);
}


Instruction10tProvider::Instruction10tProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION10T, 2) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    nAA = static_cast<std::int8_t>(this->op_codes[1]);
}

std::int8_t Instruction10tProvider::getNAA() const {
    return nAA;
}


disassembler::operand_type Instruction10tProvider::get_nAA_type() const {
    return disassembler::operand_type::OFFSET;
}

std::string_view Instruction10tProvider::format_instruction() {
    if (instruction.empty()) {
        std::stringstream str;
        str << opcode_names.at(opcode);
        str << "0x " << std::hex << ((nAA * 2) + static_cast<std::int64_t>(address));
        str << " // ";
        if (nAA > 0)
            str << "+";
        else if (nAA < 0)
            str << "-";
        str << "0x" << std::hex << std::to_string(nAA);
        instruction = str.str();
    }
    return instruction;
}

std::string_view Instruction10tProvider::print_instruction() {
    return format_instruction();
}


std::string Instruction10tProvider::print_instruction_string() {
    return std::string(format_instruction());
}

Instruction20tProvider::Instruction20tProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION20T, 4) {
    if (op_codes[1] != 0) {
        invalidate_instruction("Error reading Instruction20t padding must be 0");
        return;
    }
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    nAAAA = *(reinterpret_cast<std::int16_t *>(&this->op_codes[2]));
}


Instruction20tProvider::Instruction20tProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION20T, 4) {
    if (op_codes[1] != 0) {
        invalidate_instruction("Error reading Instruction20t padding must be 0");
        return;
    }
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    nAAAA = *(reinterpret_cast<std::int16_t *>(&this->op_codes[2]));
}

std::int16_t Instruction20tProvider::getNAAAA() const {
    return nAAAA;
}


disassembler::operand_type Instruction20tProvider::get_nAAAA_type() const {
    return disassembler::operand_type::OFFSET;
}

std::string_view Instruction20tProvider::format_instruction() {
    if (instruction.empty()) {
        std::stringstream str;
        str << opcode_names.at(opcode);
        str << "0x " << std::hex << ((nAAAA * 2) + static_cast<std::int64_t>(address));
        str << " // ";
        if (nAAAA > 0)
            str << "+";
        else if (nAAAA < 0)
            str << "-";
        str << "0x" << std::hex << std::to_string(nAAAA);
        instruction = str.str();
    }
    return instruction;
}

std::string_view Instruction20tProvider::print_instruction() {
    return format_instruction();
}

std::string Instruction20tProvider::print_instruction_string() {
    return std::string(format_instruction());
}

Instruction20bcProvider::Instruction20bcProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION20BC, 4) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    nAA = this->op_codes[1];
    nBBBB = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[2]));
}


Instruction20bcProvider::Instruction20bcProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION20BC, 4) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    nAA = this->op_codes[1];
    nBBBB = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[2]));
}


std::uint8_t Instruction20bcProvider::getNAA() const {
    return nAA;
}


disassembler::operand_type Instruction20bcProvider::get_nAA_type() const {
    return disassembler::operand_type::LITERAL;
}

std::uint16_t Instruction20bcProvider::getNBBBB() const {
    return nBBBB;
}


disassembler::operand_type Instruction20bcProvider::get_nBBBB_type() const {
    return disassembler::operand_type::KIND;
}

std::string_view Instruction20bcProvider::format_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += std::to_string(nAA);
        instruction += ", kind@" + std::to_string(nBBBB);
    }
    return instruction;
}

std::string_view Instruction20bcProvider::print_instruction() {
    return format_instruction();
}


std::string Instruction20bcProvider::print_instruction_string() {
    return std::string(format_instruction());
}


Instruction22xProvider::Instruction22xProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22X, 4) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vAA = this->op_codes[1];
    vBBBB = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[2]));
}

Instruction22xProvider::Instruction22xProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22X, 4) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vAA = this->op_codes[1];
    vBBBB = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[2]));
}


std::uint8_t Instruction22xProvider::getVAA() const {
    return vAA;
}


disassembler::operand_type Instruction22xProvider::get_vAA_type() const {
    return disassembler::operand_type::REGISTER;
}


std::uint16_t Instruction22xProvider::getVBBBB() const {
    return vBBBB;
}


disassembler::operand_type Instruction22xProvider::get_vBBBB_type() const {
    return disassembler::operand_type::REGISTER;
}

std::string_view Instruction22xProvider::format_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vAA);
        instruction += ", v" + std::to_string(vBBBB);
    }
    return instruction;
}

std::string_view Instruction22xProvider::print_instruction() {
    return format_instruction();
}


std::string Instruction22xProvider::print_instruction_string() {
    return std::string(format_instruction());
}


Instruction21tProvider::Instruction21tProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION21T, 4) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vAA = this->op_codes[1];
    nBBBB = *(reinterpret_cast<std::int16_t *>(&this->op_codes[2]));

    if (nBBBB == 0) {
        invalidate_instruction("Error reading Instruction21t offset cannot be 0");
    }
}

Instruction21tProvider::Instruction21tProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION21T, 4) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vAA = this->op_codes[1];
    nBBBB = *(reinterpret_cast<std::int16_t *>(&this->op_codes[2]));
    if (nBBBB == 0) {
        invalidate_instruction("Error reading Instruction21t offset cannot be 0");
    }
}

std::uint8_t Instruction21tProvider::getVAA() const {
    return vAA;
}

disassembler::operand_type Instruction21tProvider::get_vAA_type() const {
    return disassembler::REGISTER;
}

std::int16_t Instruction21tProvider::getNBBBB() const {
    return nBBBB;
}

disassembler::operand_type Instruction21tProvider::get_nBBBB_type() const {
    return disassembler::OFFSET;
}

std::string_view Instruction21tProvider::format_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vAA);
        instruction += ", " + std::to_string(nBBBB);
    }
    return instruction;
}

std::string_view Instruction21tProvider::print_instruction() {
    return format_instruction();
}

std::string Instruction21tProvider::print_instruction_string() {
    return std::string(format_instruction());
}


Instruction21sProvider::Instruction21sProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION21S, 4) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vAA = this->op_codes[1];
    nBBBB = *(reinterpret_cast<std::int16_t *>(&this->op_codes[2]));
}

Instruction21sProvider::Instruction21sProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION21S, 4) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vAA = this->op_codes[1];
    nBBBB = *(reinterpret_cast<std::int16_t *>(&this->op_codes[2]));
}

std::uint8_t Instruction21sProvider::getVAA() const {
    return vAA;
}

disassembler::operand_type Instruction21sProvider::get_vAA_type() const {
    return disassembler::REGISTER;
}

std::int16_t Instruction21sProvider::getNBBBB() const {
    return nBBBB;
}

disassembler::operand_type Instruction21sProvider::get_nBBBB_type() const {
    return disassembler::OFFSET;
}

std::string_view Instruction21sProvider::format_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vAA);
        instruction += ", " + std::to_string(nBBBB);
    }
    return instruction;
}

std::string_view Instruction21sProvider::print_instruction() {
    return format_instruction();
}

std::string Instruction21sProvider::print_instruction_string() {
    return std::string(format_instruction());
}

Instruction21hProvider::Instruction21hProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION21S, 4) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vAA = this->op_codes[1];
    nBBBB = *(reinterpret_cast<std::int16_t *>(&this->op_codes[2]));
    switch (opcode) {
        case disassembler::opcodes::OP_CONST_HIGH16:
            nBBBB = nBBBB << 16;
            break;
        case disassembler::opcodes::OP_CONST_WIDE_HIGH16:
            nBBBB = nBBBB << 48;
            break;
        default:
            invalidate_instruction("Instruction21h: Error, not supported opcode");
            break;
    }
}

Instruction21hProvider::Instruction21hProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION21S, 4) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vAA = this->op_codes[1];
    nBBBB = *(reinterpret_cast<std::int16_t *>(&this->op_codes[2]));
    switch (opcode) {
        case disassembler::opcodes::OP_CONST_HIGH16:
            nBBBB = nBBBB << 16;
            break;
        case disassembler::opcodes::OP_CONST_WIDE_HIGH16:
            nBBBB = nBBBB << 48;
            break;
        default:
            invalidate_instruction("Instruction21h: Error, not supported opcode");
            break;
    }
}

std::uint8_t Instruction21hProvider::getVAA() const {
    return vAA;
}

disassembler::operand_type Instruction21hProvider::get_vAA_type() const {
    return disassembler::REGISTER;
}

std::int64_t Instruction21hProvider::getnBBBB() const {
    return nBBBB;
}

disassembler::operand_type Instruction21hProvider::get_nBBBB_type() const {
    return disassembler::LITERAL;
}

std::string_view Instruction21hProvider::format_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vAA);
        instruction += ", " + std::to_string(nBBBB);
    }
    return instruction;
}

std::string_view Instruction21hProvider::print_instruction() {
    return format_instruction();
}

std::string Instruction21hProvider::print_instruction_string() {
    return std::string(format_instruction());
}

Instruction21cProvider::Instruction21cProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION21C, 4) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vAA = this->op_codes[1];
    iBBBB = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[2]));
    source_id = std::monostate{};
}

Instruction21cProvider::Instruction21cProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION21C, 4) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vAA = this->op_codes[1];
    iBBBB = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[2]));
    source_id = std::monostate{};

    switch (get_kind()) {
        case disassembler::kind::STRING:
            source_id = dex.get_string_by_id(iBBBB);
            break;
        case disassembler::kind::TYPE:
            source_id = dex.get_type_by_id(iBBBB);
            break;
        case disassembler::kind::FIELD:
            source_id = dex.get_field_by_id(iBBBB);
            break;
        case disassembler::kind::METH:
            source_id = dex.get_method_by_id(iBBBB);
            break;
        case disassembler::kind::PROTO:
            source_id = dex.get_prototype_by_id(iBBBB);
            break;
        default:
            invalidate_instruction("Instruction21c: error, kind instruction not supported");
            break;
    }
}

std::uint8_t Instruction21cProvider::getVAA() const {
    return vAA;
}

disassembler::operand_type Instruction21cProvider::get_vAA_type() const {
    return disassembler::REGISTER;
}

std::uint16_t Instruction21cProvider::getIBBBB() const {
    return iBBBB;
}

disassembler::operand_type Instruction21cProvider::get_iBBBB_type() const {
    return disassembler::KIND;
}

kind_type_t Instruction21cProvider::get_iBBBB_kind() {
    return source_id;
}

std::string_view Instruction21cProvider::format_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vAA);
        instruction += ", " + get_kind_type_as_string(source_id, iBBBB);
    }
    return instruction;
}

std::string_view Instruction21cProvider::print_instruction() {
    return format_instruction();
}

std::string Instruction21cProvider::print_instruction_string() {
    return std::string(format_instruction());
}

Instruction23xProvider::Instruction23xProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION23X, 4) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vAA = this->op_codes[1];
    vBB = this->op_codes[2];
    vCC = this->op_codes[3];
}

Instruction23xProvider::Instruction23xProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION23X, 4) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vAA = this->op_codes[1];
    vBB = this->op_codes[2];
    vCC = this->op_codes[3];
}

std::uint8_t Instruction23xProvider::getVAA() const {
    return vAA;
}

disassembler::operand_type Instruction23xProvider::get_vAA_type() const {
    return disassembler::REGISTER;
}

std::uint8_t Instruction23xProvider::getVBB() const {
    return vBB;
}

disassembler::operand_type Instruction23xProvider::get_vBB_type() const {
    return disassembler::REGISTER;
}

std::uint8_t Instruction23xProvider::getVCC() const {
    return vCC;
}

disassembler::operand_type Instruction23xProvider::get_vCC_type() const {
    return disassembler::REGISTER;
}

std::string_view Instruction23xProvider::format_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vAA);
        instruction += ", v" + std::to_string(vBB);
        instruction += ", v" + std::to_string(vCC);
    }
    return instruction;
}

std::string_view Instruction23xProvider::print_instruction() {
    return format_instruction();
}

std::string Instruction23xProvider::print_instruction_string() {
    return std::string(format_instruction());
}

Instruction22bProvider::Instruction22bProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22B, 4) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vAA = this->op_codes[1];
    vBB = this->op_codes[2];
    nCC = static_cast<std::int8_t>(this->op_codes[3]);
}

Instruction22bProvider::Instruction22bProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22B, 4) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vAA = this->op_codes[1];
    vBB = this->op_codes[2];
    nCC = static_cast<std::int8_t>(this->op_codes[3]);
}

std::uint8_t Instruction22bProvider::getVAA() const {
    return vAA;
}

disassembler::operand_type Instruction22bProvider::get_vAA_type() const {
    return disassembler::REGISTER;
}

std::uint8_t Instruction22bProvider::getVBB() const {
    return vBB;
}

disassembler::operand_type Instruction22bProvider::get_vBB_type() const {
    return disassembler::REGISTER;
}

std::int8_t Instruction22bProvider::getNCC() const {
    return nCC;
}

disassembler::operand_type Instruction22bProvider::get_nCC_type() const {
    return disassembler::LITERAL;
}

std::string_view Instruction22bProvider::format_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vAA);
        instruction += ", v" + std::to_string(vBB);
        instruction += ", " + std::to_string(nCC);
    }
    return instruction;
}

std::string_view Instruction22bProvider::print_instruction() {
    return format_instruction();
}

std::string Instruction22bProvider::print_instruction_string() {
    return std::string(format_instruction());
}


Instruction22tProvider::Instruction22tProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22T, 4) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vA = this->op_codes[1] & 0x0F;
    vB = (this->op_codes[1] & 0xF0) >> 4;
    nCCCC = static_cast<std::int16_t>(this->op_codes[2]);

    if (nCCCC == 0)
        invalidate_instruction("Error reading Instruction22t offset cannot be 0");
}

Instruction22tProvider::Instruction22tProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22T, 4) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vA = this->op_codes[1] & 0x0F;
    vB = (this->op_codes[1] & 0xF0) >> 4;
    nCCCC = static_cast<std::int16_t>(this->op_codes[2]);

    if (nCCCC == 0)
        invalidate_instruction("Error reading Instruction22t offset cannot be 0");
}

std::uint8_t Instruction22tProvider::getVA() const {
    return vA;
}

disassembler::operand_type Instruction22tProvider::get_vA_type() const {
    return disassembler::REGISTER;
}

std::uint8_t Instruction22tProvider::getVB() const {
    return vB;
}

disassembler::operand_type Instruction22tProvider::get_vB_type() const {
    return disassembler::REGISTER;
}

std::int16_t Instruction22tProvider::getNCCCC() const {
    return nCCCC;
}

disassembler::operand_type Instruction22tProvider::get_nCCCC_type() const {
    return disassembler::OFFSET;
}

std::string_view Instruction22tProvider::format_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vA);
        instruction += ", v" + std::to_string(vB);
        instruction += ", " + std::to_string(nCCCC);
    }
    return instruction;
}

std::string_view Instruction22tProvider::print_instruction() {
    return format_instruction();
}

std::string Instruction22tProvider::print_instruction_string() {
    return std::string(format_instruction());
}

Instruction22sProvider::Instruction22sProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22S, 4) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vA = this->op_codes[1] & 0x0F;
    vB = (this->op_codes[1] & 0xF0) >> 4;
    nCCCC = static_cast<std::int16_t>(this->op_codes[2]);
}

Instruction22sProvider::Instruction22sProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22S, 4) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vA = this->op_codes[1] & 0x0F;
    vB = (this->op_codes[1] & 0xF0) >> 4;
    nCCCC = static_cast<std::int16_t>(this->op_codes[2]);
}

std::uint8_t Instruction22sProvider::getVA() const {
    return vA;
}

disassembler::operand_type Instruction22sProvider::get_vA_type() const {
    return disassembler::REGISTER;
}

std::uint8_t Instruction22sProvider::getVB() const {
    return vB;
}

disassembler::operand_type Instruction22sProvider::get_vB_type() const {
    return disassembler::REGISTER;
}

std::int16_t Instruction22sProvider::getNCCCC() const {
    return nCCCC;
}

disassembler::operand_type Instruction22sProvider::get_nCCCC_type() const {
    return disassembler::LITERAL;
}

std::string_view Instruction22sProvider::format_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vA);
        instruction += ", v" + std::to_string(vB);
        instruction += ", " + std::to_string(nCCCC);
    }
    return instruction;
}

std::string_view Instruction22sProvider::print_instruction() {
    return format_instruction();
}

std::string Instruction22sProvider::print_instruction_string() {
    return std::string(format_instruction());
}

Instruction22cProvider::Instruction22cProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22C, 4) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vA = this->op_codes[1] & 0x0F;
    vB = (this->op_codes[1] & 0xF0) >> 4;
    iCCCC = static_cast<std::uint16_t>(this->op_codes[2]);
    checked_id = std::monostate{};
}

Instruction22cProvider::Instruction22cProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22C, 4) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vA = this->op_codes[1] & 0x0F;
    vB = (this->op_codes[1] & 0xF0) >> 4;
    iCCCC = static_cast<std::uint16_t>(this->op_codes[2]);
    checked_id = std::monostate{};

    switch (get_kind()) {
        case disassembler::kind::TYPE:
            checked_id = dex.get_type_by_id(iCCCC);
            break;
        case disassembler::kind::FIELD:
            checked_id = dex.get_field_by_id(iCCCC);
            break;
        default:
            invalidate_instruction("Instruction22c: error, kind instruction not supported");
            break;
    }
}

std::uint8_t Instruction22cProvider::getVA() const {
    return vA;
}

disassembler::operand_type Instruction22cProvider::get_vA_type() const {
    return disassembler::REGISTER;
}

std::uint8_t Instruction22cProvider::getVB() const {
    return vB;
}

disassembler::operand_type Instruction22cProvider::get_vB_type() const {
    return disassembler::REGISTER;
}

std::uint16_t Instruction22cProvider::getICCCC() const {
    return iCCCC;
}

disassembler::operand_type Instruction22cProvider::get_iCCCC_type() const {
    return disassembler::KIND;
}

kind_type_t Instruction22cProvider::get_checked_id_as_kind() const {
    return checked_id;
}

std::string_view Instruction22cProvider::format_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vA);
        instruction += ", v" + std::to_string(vB);
        instruction += ", " + get_kind_type_as_string(checked_id, iCCCC);
    }
    return instruction;
}

std::string_view Instruction22cProvider::print_instruction() {
    return format_instruction();
}

std::string Instruction22cProvider::print_instruction_string() {
    return std::string(format_instruction());
}

Instruction22csProvider::Instruction22csProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22CS, 4) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vA = this->op_codes[1] & 0x0F;
    vB = (this->op_codes[1] & 0xF0) >> 4;
    iCCCC = static_cast<std::uint16_t>(this->op_codes[2]);
    field = std::monostate{};
}

Instruction22csProvider::Instruction22csProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22CS, 4) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vA = this->op_codes[1] & 0x0F;
    vB = (this->op_codes[1] & 0xF0) >> 4;
    iCCCC = static_cast<std::uint16_t>(this->op_codes[2]);

    switch (get_kind()) {
        case disassembler::kind::FIELD:
            field = dex.get_field_by_id(iCCCC);
            break;
        default:
            invalidate_instruction("Instruction22cs: error, kind instruction not supported");
            break;
    }
}

std::uint8_t Instruction22csProvider::getVA() const {
    return vA;
}

disassembler::operand_type Instruction22csProvider::get_vA_type() const {
    return disassembler::REGISTER;
}

std::uint8_t Instruction22csProvider::getVB() const {
    return vB;
}

disassembler::operand_type Instruction22csProvider::get_vB_type() const {
    return disassembler::REGISTER;
}

std::uint16_t Instruction22csProvider::getICCCC() const {
    return iCCCC;
}

disassembler::operand_type Instruction22csProvider::get_iCCCC_type() const {
    return disassembler::KIND;
}

kind_type_t Instruction22csProvider::get_field() const {
    return field;
}

std::string_view Instruction22csProvider::format_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vA);
        instruction += ", v" + std::to_string(vB);
        instruction += ", " + get_kind_type_as_string(field, iCCCC);
    }
    return instruction;
}

std::string_view Instruction22csProvider::print_instruction() {
    return format_instruction();
}

std::string Instruction22csProvider::print_instruction_string() {
    return std::string(format_instruction());
}


Instruction30tProvider::Instruction30tProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION30T, 6) {
    if (op_codes[1]) {
        invalidate_instruction("Error reading Instruction30t padding must be 0");
        return;
    }
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    nAAAAAAAA = static_cast<std::int32_t>(this->op_codes[2]);
    if (nAAAAAAAA == 0)
        invalidate_instruction("Error reading Instruction30t offset cannot be 0");
}

Instruction30tProvider::Instruction30tProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION30T, 6) {
    if (op_codes[1]) {
        invalidate_instruction("Error reading Instruction30t padding must be 0");
        return;
    }
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    nAAAAAAAA = static_cast<std::int32_t>(this->op_codes[2]);
    if (nAAAAAAAA == 0)
        invalidate_instruction("Error reading Instruction30t offset cannot be 0");
}

std::int32_t Instruction30tProvider::getNAAAAAAAA() const {
    return nAAAAAAAA;
}

disassembler::operand_type Instruction30tProvider::get_nAAAAAAAA_type() const {
    return disassembler::OFFSET;
}

std::string_view Instruction30tProvider::format_instruction() {
    if (instruction.empty()) {
        std::stringstream str;
        str << opcode_names.at(opcode) << " ";
        str << "0x" << std::hex << ((nAAAAAAAA * 2) + static_cast<std::int64_t>(address));
        str << " // ";
        if (nAAAAAAAA > 0)
            str << "+";
        else if (nAAAAAAAA < 0)
            str << "-";
        str << "0x" << std::hex << nAAAAAAAA;
        instruction = str.str();
    }
    return instruction;
}

std::string_view Instruction30tProvider::print_instruction() {
    return format_instruction();
}

std::string Instruction30tProvider::print_instruction_string() {
    return std::string(format_instruction());
}


Instruction32xProvider::Instruction32xProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION32X, 6) {
    if (op_codes[1] != 0) {
        invalidate_instruction("Error reading Instruction32x padding must be 0");
        return;
    }
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vAAAA = static_cast<std::uint16_t>(this->op_codes[2]);
    vBBBB = static_cast<std::uint16_t>(this->op_codes[4]);
}

Instruction32xProvider::Instruction32xProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION32X, 6) {
    if (op_codes[1] != 0) {
        invalidate_instruction("Error reading Instruction32x padding must be 0");
        return;
    }
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vAAAA = static_cast<std::uint16_t>(this->op_codes[2]);
    vBBBB = static_cast<std::uint16_t>(this->op_codes[4]);
}

std::uint16_t Instruction32xProvider::getVAAAA() const {
    return vAAAA;
}

disassembler::operand_type Instruction32xProvider::get_vAAAA_type() const {
    return disassembler::REGISTER;
}

std::uint16_t Instruction32xProvider::getVBBBB() const {
    return vBBBB;
}

disassembler::operand_type Instruction32xProvider::get_vBBBB_type() const {
    return disassembler::REGISTER;
}

std::string_view Instruction32xProvider::format_instruction() {
    if (instruction.empty()) {
        std::stringstream str;
        str << opcode_names.at(opcode) << " ";
        str << "v" << vAAAA << ", v" << vBBBB;
        instruction = str.str();
    }
    return instruction;
}

std::string_view Instruction32xProvider::print_instruction() {
    return format_instruction();
}

std::string Instruction32xProvider::print_instruction_string() {
    return std::string(format_instruction());
}

Instruction31iProvider::Instruction31iProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION31I, 6) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vAA = static_cast<std::uint16_t>(this->op_codes[1]);
    nBBBBBBBB = static_cast<std::uint32_t>(this->op_codes[2]);
}

Instruction31iProvider::Instruction31iProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION31I, 6) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vAA = static_cast<std::uint16_t>(this->op_codes[1]);
    nBBBBBBBB = static_cast<std::uint32_t>(this->op_codes[2]);
}

std::uint8_t Instruction31iProvider::getVAA() const {
    return vAA;
}

disassembler::operand_type Instruction31iProvider::get_vAA_type() const {
    return disassembler::REGISTER;
}

std::uint32_t Instruction31iProvider::getNBBBBBBBB() const {
    return nBBBBBBBB;
}

float Instruction31iProvider::getNBBBBBBBB_Float() const {
    union {
        float f;
        std::uint32_t i;
    } conv;

    conv.i = nBBBBBBBB;
    return conv.f;
}

disassembler::operand_type Instruction31iProvider::get_nBBBBBBBB_type() const {
    return disassembler::LITERAL;
}

std::string_view Instruction31iProvider::format_instruction() {
    if (instruction.empty()) {
        std::stringstream str;
        str << opcode_names.at(opcode) << " ";
        str << "v" << vAA;
        str << ", " << getNBBBBBBBB_Float() << " // " << nBBBBBBBB;
        instruction = str.str();
    }
    return instruction;
}

std::string_view Instruction31iProvider::print_instruction() {
    return format_instruction();
}

std::string Instruction31iProvider::print_instruction_string() {
    return std::string(format_instruction());
}

Instruction31tProvider::Instruction31tProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION31T, 6) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vAA = static_cast<std::uint16_t>(this->op_codes[1]);
    nBBBBBBBB = static_cast<std::int32_t>(this->op_codes[2]);
    switch_instruction = std::monostate{};

    switch (opcode) {
        case disassembler::opcodes::OP_PACKED_SWITCH:
            type_of_switch = disassembler::PACKED_SWITCH;
            break;
        case disassembler::opcodes::OP_SPARSE_SWITCH:
            type_of_switch = disassembler::SPARSE_SWITCH;
            break;
        default:
            type_of_switch = disassembler::NONE_SWITCH;
            break;
    }
}

Instruction31tProvider::Instruction31tProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION31T, 6) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vAA = static_cast<std::uint16_t>(this->op_codes[1]);
    nBBBBBBBB = static_cast<std::int32_t>(this->op_codes[2]);
    switch_instruction = std::monostate{};

    switch (opcode) {
        case disassembler::opcodes::OP_PACKED_SWITCH:
            type_of_switch = disassembler::PACKED_SWITCH;
            break;
        case disassembler::opcodes::OP_SPARSE_SWITCH:
            type_of_switch = disassembler::SPARSE_SWITCH;
            break;
        default:
            type_of_switch = disassembler::NONE_SWITCH;
            break;
    }
}

std::uint8_t Instruction31tProvider::getVAA() const {
    return vAA;
}

disassembler::operand_type Instruction31tProvider::get_vAA_type() const {
    return disassembler::REGISTER;
}

std::int32_t Instruction31tProvider::getNBBBBBBBB() const {
    return nBBBBBBBB;
}

disassembler::operand_type Instruction31tProvider::get_nBBBBBBBB_type() const {
    return disassembler::OFFSET;
}

disassembler::type_of_switch_t Instruction31tProvider::get_type_of_switch() const {
    return type_of_switch;
}

switch_type_t Instruction31tProvider::get_switch() const {
    return switch_instruction;
}

void Instruction31tProvider::set_packed_switch(PackedSwitchProvider *packed_switch) {
    this->switch_instruction = packed_switch;
}

void Instruction31tProvider::set_sparse_switch(SparseSwitchProvider *sparse_switch) {
    this->switch_instruction = sparse_switch;
}

std::string_view Instruction31tProvider::format_instruction() {
    if (instruction.empty()) {
        std::stringstream str;
        str << opcode_names.at(opcode) << " ";
        str << "v" << vAA;
        str << ", " << nBBBBBBBB;
        instruction = str.str();
    }
    return instruction;
}

std::string_view Instruction31tProvider::print_instruction() {
    return format_instruction();
}

std::string Instruction31tProvider::print_instruction_string() {
    return std::string(format_instruction());
}

Instruction31cProvider::Instruction31cProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION31C, 6) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vAA = static_cast<std::uint16_t>(this->op_codes[1]);
    iBBBBBBBB = static_cast<std::uint32_t>(this->op_codes[2]);
}

Instruction31cProvider::Instruction31cProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION31C, 6) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vAA = static_cast<std::uint16_t>(this->op_codes[1]);
    iBBBBBBBB = static_cast<std::uint32_t>(this->op_codes[2]);
    pointed_string = dex.get_string_by_id(iBBBBBBBB);
}

std::uint8_t Instruction31cProvider::getVAA() const {
    return vAA;
}

disassembler::operand_type Instruction31cProvider::get_vAA_type() const {
    return disassembler::REGISTER;
}

std::uint32_t Instruction31cProvider::getIBBBBBBBB() const {
    return iBBBBBBBB;
}

disassembler::operand_type Instruction31cProvider::get_IBBBBBBBB_type() const {
    return disassembler::OFFSET;
}

std::string_view Instruction31cProvider::get_string_value() const {
    return pointed_string;
}

std::string Instruction31cProvider::get_string_value_string() const {
    return pointed_string;
}

std::string_view Instruction31cProvider::format_instruction() {
    if (instruction.empty()) {
        std::stringstream str;
        str << opcode_names.at(opcode) << " ";
        str << "v" << vAA;
        str << ", " << iBBBBBBBB;
        if (!pointed_string.empty()) {
            str << " // " << pointed_string;
        }
        instruction = str.str();
    }
    return instruction;
}

std::string_view Instruction31cProvider::print_instruction() {
    return format_instruction();
}

std::string Instruction31cProvider::print_instruction_string() {
    return std::string(format_instruction());
}

Instruction35cProvider::Instruction35cProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION35C, 6) {
    std::uint8_t regs[5];

    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    array_size = (this->op_codes[1] & 0xF0) >> 4;
    type_index = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[2]));
    type_value = std::monostate{};

    regs[4] = this->op_codes[1] & 0x0F;
    regs[0] = this->op_codes[4] & 0x0F;
    regs[1] = (this->op_codes[4] & 0xF0) >> 4;
    regs[2] = this->op_codes[5] & 0x0F;
    regs[3] = (this->op_codes[5] & 0xF0) >> 4;

    if (array_size > 5) {
        invalidate_instruction("Error in array size of Instruction35c, cannot be greater than 5");
        return;
    }

    for (size_t I = 0; I < array_size; ++I) {
        registers.push_back(regs[I]);
    }
}

Instruction35cProvider::Instruction35cProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION35C, 6) {
    std::uint8_t regs[5];

    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    array_size = (this->op_codes[1] & 0xF0) >> 4;
    type_index = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[2]));
    type_value = std::monostate{};

    regs[4] = this->op_codes[1] & 0x0F;
    regs[0] = this->op_codes[4] & 0x0F;
    regs[1] = (this->op_codes[4] & 0xF0) >> 4;
    regs[2] = this->op_codes[5] & 0x0F;
    regs[3] = (this->op_codes[5] & 0xF0) >> 4;

    if (array_size > 5) {
        invalidate_instruction("Instruction35c: Error in array size of Instruction35c, cannot be greater than 5");
        return;
    }

    for (size_t I = 0; I < array_size; ++I) {
        registers.push_back(regs[I]);
    }

    switch (get_kind()) {
        case disassembler::TYPE:
            type_value = dex.get_type_by_id(type_index);
            break;
        case disassembler::METH:
            type_value = dex.get_method_by_id(type_index);
            break;
        default:
            invalidate_instruction("Instruction35c: error, kind instruction not supported");
            break;
    };
}

std::uint8_t Instruction35cProvider::get_number_of_registers() const {
    return array_size;
}

std::span<std::uint8_t> Instruction35cProvider::get_registers() {
    static std::span regs{registers};
    return regs;
}

disassembler::operand_type Instruction35cProvider::get_registers_type() const {
    return disassembler::REGISTER;
}

std::uint16_t Instruction35cProvider::get_type_idx() const {
    return type_index;
}

disassembler::operand_type Instruction35cProvider::get_value_type() const {
    return disassembler::KIND;
}

disassembler::kind Instruction35cProvider::get_value_kind() const {
    return get_kind();
}

kind_type_t Instruction35cProvider::get_value() const {
    return type_value;
}

std::string_view Instruction35cProvider::format_instruction() {
    if (instruction.empty()) {
        std::stringstream str;
        str << opcode_names.at(opcode) << " ";
        str << " {";
        for (size_t i = 0, e = registers.size(); i < e; i++) {
            auto reg = registers[i];
            str << "v" << reg;
            if (i < registers.size() - 1)
                str << ", ";
        }
        str << "}, " + get_kind_type_as_string(type_value, type_index);
        instruction = str.str();
    }
    return instruction;
}

std::string_view Instruction35cProvider::print_instruction() {
    return format_instruction();
}

std::string Instruction35cProvider::print_instruction_string() {
    return std::string(format_instruction());
}

Instruction3rcProvider::Instruction3rcProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION3RC, 6) {
    std::uint16_t vCCCC;
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    array_size = this->op_codes[1];
    index = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[2]));
    vCCCC = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[4]));
    index_value = std::monostate{};

    for (std::uint16_t I = vCCCC, E = vCCCC + array_size; I < E; I++)
        registers.push_back(I);
}

Instruction3rcProvider::Instruction3rcProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION3RC, 6) {
    std::uint16_t vCCCC;
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    array_size = this->op_codes[1];
    index = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[2]));
    vCCCC = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[4]));
    index_value = std::monostate{};

    for (std::uint16_t I = vCCCC, E = vCCCC + array_size; I < E; I++)
        registers.push_back(I);

    switch (get_kind()) {
        case disassembler::TYPE:
            index_value = dex.get_type_by_id(static_cast<std::uint32_t>(index));
            break;
        case disassembler::METH:
            index_value = dex.get_method_by_id(static_cast<std::uint32_t>(index));
            break;
        default:
            break;
    }

}

std::uint8_t Instruction3rcProvider::get_registers_size() const {
    return array_size;
}

std::uint16_t Instruction3rcProvider::get_index() const {
    return index;
}

kind_type_t Instruction3rcProvider::get_index_value() const {
    return index_value;
}

disassembler::operand_type Instruction3rcProvider::get_index_type() const {
    return disassembler::KIND;
}

std::span<std::uint16_t> Instruction3rcProvider::get_registers() {
    static std::span<std::uint16_t> regs{registers};
    return regs;
}

std::string_view Instruction3rcProvider::format_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " {";
        for (auto reg: registers)
            instruction += "v" + std::to_string(reg) + ", ";
        if (!registers.empty())
            instruction = instruction.substr(0, instruction.size() - 2);
        instruction += "}, " + get_kind_type_as_string(index_value, index);
    }
    return instruction;
}

std::string_view Instruction3rcProvider::print_instruction() {
    return format_instruction();
}

std::string Instruction3rcProvider::print_instruction_string() {
    return std::string(format_instruction());
}

Instruction45ccProvider::Instruction45ccProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION45CC, 8) {
    std::uint8_t regC, regD, regE, regF, regG;

    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    reg_count = (this->op_codes[1] & 0xF0) >> 4;
    regG = this->op_codes[1] & 0x0F;
    method_reference = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[2]));
    regD = (this->op_codes[4] & 0xF0) >> 4;
    regC = this->op_codes[4] & 0x0F;
    regF = (this->op_codes[5] & 0xF0) >> 4;
    regE = this->op_codes[5] & 0x0F;
    prototype_reference = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[8]));
    method_value = std::monostate{};
    prototype_value = std::monostate{};

    if (reg_count > 5) {
        invalidate_instruction("Error in reg_count from Instruction45cc cannot be greater than 5");
        return;
    }

    if (reg_count > 0)
        registers.push_back(regC);
    if (reg_count > 1)
        registers.push_back(regD);
    if (reg_count > 2)
        registers.push_back(regE);
    if (reg_count > 3)
        registers.push_back(regF);
    if (reg_count > 4)
        registers.push_back(regG);
}

Instruction45ccProvider::Instruction45ccProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION45CC, 8) {
    std::uint8_t regC, regD, regE, regF, regG;

    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    reg_count = (this->op_codes[1] & 0xF0) >> 4;
    regG = this->op_codes[1] & 0x0F;
    method_reference = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[2]));
    regD = (this->op_codes[4] & 0xF0) >> 4;
    regC = this->op_codes[4] & 0x0F;
    regF = (this->op_codes[5] & 0xF0) >> 4;
    regE = this->op_codes[5] & 0x0F;
    prototype_reference = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[8]));
    method_value = std::monostate{};
    prototype_value = std::monostate{};

    if (reg_count > 5) {
        invalidate_instruction("Error in reg_count from Instruction45cc cannot be greater than 5");
        return;
    }

    if (reg_count > 0)
        registers.push_back(regC);
    if (reg_count > 1)
        registers.push_back(regD);
    if (reg_count > 2)
        registers.push_back(regE);
    if (reg_count > 3)
        registers.push_back(regF);
    if (reg_count > 4)
        registers.push_back(regG);

    if (method_reference >= dex.get_number_of_methods()) {
        invalidate_instruction("Error method reference out of bound in Instruction45cc");
        return;
    }

    if (prototype_reference >= dex.get_number_of_prototypes()) {
        invalidate_instruction("Error prototype reference out of bound in Instruction45cc");
        return;
    }

    method_value = dex.get_method_by_id(method_reference);
    prototype_value = dex.get_prototype_by_id(prototype_reference);
}

std::uint8_t Instruction45ccProvider::get_number_of_registers() const {
    return reg_count;
}

std::span<std::uint8_t> Instruction45ccProvider::get_registers() {
    static std::span<std::uint8_t> regs{registers};
    return regs;
}

std::uint16_t Instruction45ccProvider::get_method_reference() const {
    return method_reference;
}

kind_type_t Instruction45ccProvider::get_method_value() const {
    return method_value;
}

std::uint16_t Instruction45ccProvider::get_prototype_reference() const {
    return prototype_reference;
}

kind_type_t Instruction45ccProvider::get_prototype_value() const {
    return prototype_value;
}

std::string_view Instruction45ccProvider::format_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " {";
        for (auto reg: registers)
            instruction += "v" + std::to_string(reg) + ", ";
        if (!registers.empty())
            instruction = instruction.substr(0, instruction.size() - 2);
        instruction += "meth@" + std::to_string(method_reference) + ", ";
        instruction += "proto@" + std::to_string(prototype_reference);
    }
    return instruction;
}

std::string_view Instruction45ccProvider::print_instruction() {
    return format_instruction();
}

std::string Instruction45ccProvider::print_instruction_string() {
    return std::string(format_instruction());
}

Instruction4rccProvider::Instruction4rccProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION4RCC, 8) {
    std::uint16_t vCCCC;

    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    reg_count = this->op_codes[1];
    method_reference = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[2]));
    vCCCC = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[4]));
    prototype_reference = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[6]));
    method_value = std::monostate{};
    prototype_value = std::monostate{};

    for (std::uint16_t I = vCCCC, E = vCCCC + reg_count; I < E; ++I)
        registers.push_back(I);
}

Instruction4rccProvider::Instruction4rccProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION4RCC, 8) {
    std::uint16_t vCCCC;

    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    reg_count = this->op_codes[1];
    method_reference = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[2]));
    vCCCC = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[4]));
    prototype_reference = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[6]));
    method_value = std::monostate{};
    prototype_value = std::monostate{};

    for (std::uint16_t I = vCCCC, E = vCCCC + reg_count; I < E; ++I)
        registers.push_back(I);

    if (method_reference >= dex.get_number_of_methods()) {
        invalidate_instruction("Error method reference out of bound in Instruction4rcc");
        return;
    }

    if (prototype_reference >= dex.get_number_of_prototypes()) {
        invalidate_instruction("Error prototype reference out of bound in Instruction4rcc");
        return;
    }

    method_value = dex.get_method_by_id(method_reference);
    prototype_value = dex.get_prototype_by_id(prototype_reference);
}

std::uint8_t Instruction4rccProvider::get_number_of_registers() const {
    return reg_count;
}

std::span<std::uint16_t> Instruction4rccProvider::get_registers() {
    static std::span<std::uint16_t> regs(registers);
    return regs;
}

std::uint16_t Instruction4rccProvider::get_method_reference() const {
    return method_reference;
}

kind_type_t Instruction4rccProvider::get_method_value() const {
    return method_value;
}

std::uint16_t Instruction4rccProvider::get_prototype_reference() const {
    return prototype_reference;
}

kind_type_t Instruction4rccProvider::get_prototype_value() const {
    return prototype_value;
}

std::string_view Instruction4rccProvider::format_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction = " {";
        for (const auto reg: registers)
            instruction += "v" + std::to_string(reg) + ", ";
        if (!registers.empty())
            instruction = instruction.substr(0, instruction.size() - 2);
        instruction += "}, ";
        instruction += "meth@" + std::to_string(method_reference) + ", ";
        instruction += "proto@" + std::to_string(prototype_reference);
    }
    return instruction;
}

std::string_view Instruction4rccProvider::print_instruction() {
    return format_instruction();
}

std::string Instruction4rccProvider::print_instruction_string() {
    return std::string(format_instruction());
}


Instruction51lProvider::Instruction51lProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION51L, 10) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vAA = this->op_codes[1];
    nBBBBBBBBBBBBBBBB = *(reinterpret_cast<std::int64_t *>(&this->op_codes[2]));
}

Instruction51lProvider::Instruction51lProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION51L, 10) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    vAA = this->op_codes[1];
    nBBBBBBBBBBBBBBBB = *(reinterpret_cast<std::int64_t *>(&this->op_codes[2]));
}

std::uint8_t Instruction51lProvider::getVAA() const {
    return vAA;
}

disassembler::operand_type Instruction51lProvider::get_vAA_type() const {
    return disassembler::REGISTER;
}

std::int64_t Instruction51lProvider::getNBBBBBBBBBBBBBBBB() const {
    return nBBBBBBBBBBBBBBBB;
}

double Instruction51lProvider::get_nBBBBBBBBBBBBBBBB_double() const {
    union {
        double d;
        std::int64_t j;
    } conv;

    conv.j = nBBBBBBBBBBBBBBBB;

    return conv.d;
}

disassembler::operand_type Instruction51lProvider::get_nBBBBBBBBBBBBBBBB_type() const {
    return disassembler::LITERAL;
}

std::string_view Instruction51lProvider::format_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " v" + std::to_string(vAA);
        instruction += ", #" + std::to_string(get_nBBBBBBBBBBBBBBBB_double());
        instruction += " // " + std::to_string(nBBBBBBBBBBBBBBBB);
    }
    return instruction;
}

std::string_view Instruction51lProvider::print_instruction() {
    return format_instruction();
}

std::string Instruction51lProvider::print_instruction_string() {
    return std::string(format_instruction());
}

PackedSwitchProvider::PackedSwitchProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_PACKEDSWITCH, 8) {
    std::int32_t aux;
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    size = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[2]));
    first_key = *(reinterpret_cast<std::uint32_t *>(&this->op_codes[4]));

    // because the instruction is larger, we have to
    // re-accommodate the op_codes span and the length
    // we have to increment it
    length += (size * 4);

    op_codes = {bytecode.begin() + index, bytecode.begin() + index + length};

    // now read the targets
    auto multiplier = sizeof(std::int32_t);
    for (size_t I = 0; I < size; ++I) {
        aux = *(reinterpret_cast<std::int32_t *>(&op_codes[8 + (I * multiplier)]));
        targets.push_back(aux);
    }
}

PackedSwitchProvider::PackedSwitchProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_PACKEDSWITCH, 8) {
    std::int32_t aux;
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    size = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[2]));
    first_key = *(reinterpret_cast<std::uint32_t *>(&this->op_codes[4]));

    // because the instruction is larger, we have to
    // re-accommodate the op_codes span and the length
    // we have to increment it
    length += (size * 4);

    op_codes = {bytecode.begin() + index, bytecode.begin() + index + length};

    // now read the targets
    auto multiplier = sizeof(std::int32_t);
    for (size_t I = 0; I < size; ++I) {
        aux = *(reinterpret_cast<std::int32_t *>(&op_codes[8 + (I * multiplier)]));
        targets.push_back(aux);
    }
}

std::uint16_t PackedSwitchProvider::get_number_of_targets() const {
    return size;
}

std::int32_t PackedSwitchProvider::get_first_key() const {
    return first_key;
}

std::span<std::int32_t> PackedSwitchProvider::get_targets() {
    static std::span<std::int32_t> tgts{targets};
    return tgts;
}

std::string_view PackedSwitchProvider::format_instruction() {
    if (instruction.empty()) {
        std::stringstream data;
        data << opcode_names.at(opcode) << " (size)" << size << " (first/last key)" << first_key << "[";
        for (const auto target: targets)
            data << "0x" << std::hex << target << ",";
        if (size > 0)
            data.seekp(-1, data.cur);
        data << "]";
        instruction = data.str();
    }
    return instruction;
}

std::string_view PackedSwitchProvider::print_instruction() {
    return format_instruction();
}

std::string PackedSwitchProvider::print_instruction_string() {
    return std::string(format_instruction());
}

SparseSwitchProvider::SparseSwitchProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_SPARSESWITCH, 4) {
    std::int32_t aux_key, aux_target;

    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);

    size = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[2]));

    length += (sizeof(std::int32_t) * size) * 2;
    op_codes = {bytecode.begin() + index, bytecode.begin() + index + length};

    auto base_targets = 4 + sizeof(std::int32_t) * size;
    auto multiplier = sizeof(std::int32_t);

    for (size_t I = 0; I < size; ++I) {
        aux_key = *(reinterpret_cast<std::int32_t *>(&op_codes[4 + I * multiplier]));
        aux_target = *(reinterpret_cast<std::int32_t *>(&op_codes[base_targets + I * multiplier]));

        keys_targets.emplace_back(aux_key, aux_target);
    }
}

SparseSwitchProvider::SparseSwitchProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_SPARSESWITCH, 4) {
    std::int32_t aux_key, aux_target;

    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);

    size = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[2]));

    length += (sizeof(std::int32_t) * size) * 2;
    op_codes = {bytecode.begin() + index, bytecode.begin() + index + length};

    auto base_targets = 4 + sizeof(std::int32_t) * size;
    auto multiplier = sizeof(std::int32_t);

    for (size_t I = 0; I < size; ++I) {
        aux_key = *(reinterpret_cast<std::int32_t *>(&op_codes[4 + I * multiplier]));
        aux_target = *(reinterpret_cast<std::int32_t *>(&op_codes[base_targets + I * multiplier]));

        keys_targets.emplace_back(aux_key, aux_target);
    }
}

std::uint16_t SparseSwitchProvider::get_size_of_targets() const {
    return size;
}

std::span<std::pair<std::int32_t, std::int32_t>> SparseSwitchProvider::get_keys_targets() {
    static std::span<std::pair<std::int32_t, std::int32_t>> targets{keys_targets};
    return targets;
}

std::string_view SparseSwitchProvider::format_instruction() {
    if (instruction.empty()) {
        std::stringstream output;
        output << opcode_names.at(opcode) << " (size)" << size << "[";
        for (const auto &key_target: keys_targets) {
            auto key = std::get<0>(key_target);
            auto target = std::get<1>(key_target);
            if (key < 0)
                output << "-0x" << std::hex << key << ":";
            else
                output << "0x" << std::hex << key << ":";
            if (target < 0)
                output << "-0x" << std::hex << target << ":";
            else
                output << "0x" << std::hex << target << ":";
            output << ",";
        }
        if (size > 0)
            output.seekp(-1, std::stringstream::cur);
        output << "]";
        instruction = output.str();
    }
    return instruction;
}

std::string_view SparseSwitchProvider::print_instruction() {
    return format_instruction();
}

std::string SparseSwitchProvider::print_instruction_string() {
    return std::string(format_instruction());
}

FillArrayDataProvivder::FillArrayDataProvivder(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_FILLARRAYDATA, 8) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    element_width = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[2]));
    size = *(reinterpret_cast<std::uint32_t *>(&this->op_codes[4]));

    // again we have to fix the length of the instruction
    // and also the opcodes
    auto buff_size = (size * element_width);
    length += buff_size;
    if (buff_size % 2 != 0)
        length += 1;
    op_codes = {bytecode.begin() + index, bytecode.begin() + index + length};

    for (size_t I = 0; I < buff_size; ++I)
        data.push_back(op_codes[8 + I]);
}

FillArrayDataProvivder::FillArrayDataProvivder(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_FILLARRAYDATA, 8) {
    opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    element_width = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[2]));
    size = *(reinterpret_cast<std::uint32_t *>(&this->op_codes[4]));

    // again we have to fix the length of the instruction
    // and also the opcodes
    auto buff_size = (size * element_width);
    length += buff_size;
    if (buff_size % 2 != 0)
        length += 1;
    op_codes = {bytecode.begin() + index, bytecode.begin() + index + length};

    for (size_t I = 0; I < buff_size; ++I)
        data.push_back(op_codes[8 + I]);
}

std::uint16_t FillArrayDataProvivder::get_element_width() const {
    return element_width;
}

std::uint32_t FillArrayDataProvivder::get_size_of_data() const {
    return size;
}

std::span<std::uint8_t> FillArrayDataProvivder::get_data() {
    static std::span<std::uint8_t> data_read{data};
    return data_read;
}

std::string_view FillArrayDataProvivder::format_instruction() {
    if (instruction.empty()) {
        std::stringstream output;
        output << "(width)" << element_width << " (size)" << size << " [";
        for (auto byte: data)
            output << "0x" << std::hex << static_cast<std::uint32_t>(byte) << ",";
        if (size > 0)
            output.seekp(-1, std::stringstream::cur);
        output << "]";
        instruction = output.str();
    }
    return instruction;
}

std::string_view FillArrayDataProvivder::print_instruction() {
    return format_instruction();
}

std::string FillArrayDataProvivder::print_instruction_string() {
    return std::string(format_instruction());
}
