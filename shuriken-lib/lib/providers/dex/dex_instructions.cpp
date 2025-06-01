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
        instruction_type(instruction_type), bytecode({}), length(0), opcode(disassembler::opcodes::OP_NONE) {
}


InstructionProvider::InstructionProvider(std::span<std::uint8_t> bytecode, std::size_t index,
                                         disassembler::dexinsttype instruction_type, std::uint32_t length)
        : instruction_type(instruction_type), bytecode({bytecode.begin() + index, bytecode.begin() + index + length}),
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
    return bytecode;
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

Instruction00xProvider::Instruction00xProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION00X) {
}


Instruction00xProvider::Instruction00xProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION00X) {
}


std::string_view Instruction00xProvider::print_instruction() {
    if (instruction.empty())
        instruction = opcode_names.at(opcode);
    return instruction;
}


std::string Instruction00xProvider::print_instruction_string() {
    if (instruction.empty())
        instruction = opcode_names.at(opcode);
    return instruction;
}

Instruction10xProvider::Instruction10xProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION10X, 2) {
}


Instruction10xProvider::Instruction10xProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION10X, 2) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
}

std::string_view Instruction10xProvider::print_instruction() {
    if (instruction.empty())
        instruction = opcode_names.at(opcode);
    return instruction;
}

std::string Instruction10xProvider::print_instruction_string() {
    if (instruction.empty())
        instruction = opcode_names.at(opcode);
    return instruction;
}

Instruction12xProvider::Instruction12xProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION12X, 2) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    vA = (this->bytecode[1] & 0x0F);
    vB = (this->bytecode[1] & 0xF0) >> 4;
}

Instruction12xProvider::Instruction12xProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION12X, 2) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    vA = (this->bytecode[1] & 0x0F);
    vB = (this->bytecode[1] & 0xF0) >> 4;
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


std::string_view Instruction12xProvider::print_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vA);
        instruction += ", ";
        instruction += "v" + std::to_string(vB);
    }
    return instruction;
}


std::string Instruction12xProvider::print_instruction_string() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vA);
        instruction += ", ";
        instruction += "v" + std::to_string(vB);
    }
    return instruction;
}

Instruction11nProvider::Instruction11nProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION11N, 2) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    vA = (this->bytecode[1] & 0x0F);
    nB = static_cast<std::int8_t>((this->bytecode[1] & 0xF0) >> 4);
}

Instruction11nProvider::Instruction11nProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION11N, 2) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    vA = (this->bytecode[1] & 0x0F);
    nB = static_cast<std::int8_t>((this->bytecode[1] & 0xF0) >> 4);
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

std::string_view Instruction11nProvider::print_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vA);
        instruction += ", ";
        instruction += std::to_string(nB);
    }
    return instruction;
}


std::string Instruction11nProvider::print_instruction_string() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vA);
        instruction += ", ";
        instruction += std::to_string(nB);
    }
    return instruction;
}

Instruction11xProvider::Instruction11xProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION11X, 2) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    vAA = this->bytecode[1];
}


Instruction11xProvider::Instruction11xProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION11X, 2) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    vAA = this->bytecode[1];
}


std::uint8_t Instruction11xProvider::getVAA() const {
    return vAA;
}


disassembler::operand_type Instruction11xProvider::get_vAA_type() const {
    return disassembler::operand_type::REGISTER;
}


std::string_view Instruction11xProvider::print_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vAA);
    }
    return instruction;
}


std::string Instruction11xProvider::print_instruction_string() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vAA);
    }
    return instruction;
}

Instruction10tProvider::Instruction10tProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION10T, 2) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    nAA = static_cast<std::int8_t>(this->bytecode[1]);
}


Instruction10tProvider::Instruction10tProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION10T, 2) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    nAA = static_cast<std::int8_t>(this->bytecode[1]);
}

std::int8_t Instruction10tProvider::getNAA() const {
    return nAA;
}


disassembler::operand_type Instruction10tProvider::get_nAA_type() const {
    return disassembler::operand_type::OFFSET;
}


std::string_view Instruction10tProvider::print_instruction() {
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


std::string Instruction10tProvider::print_instruction_string() {
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

Instruction20tProvider::Instruction20tProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION20T, 4) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    nAAAA = *(reinterpret_cast<std::int16_t *>(&this->bytecode[2]));
}


Instruction20tProvider::Instruction20tProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION20T, 4) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    nAAAA = *(reinterpret_cast<std::int16_t *>(&this->bytecode[2]));
}

std::int16_t Instruction20tProvider::getNAAAA() const {
    return nAAAA;
}


disassembler::operand_type Instruction20tProvider::get_nAAAA_type() const {
    return disassembler::operand_type::OFFSET;
}

std::string_view Instruction20tProvider::print_instruction() {
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

std::string Instruction20tProvider::print_instruction_string() {
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

Instruction20bcProvider::Instruction20bcProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION20BC, 4) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    nAA = this->bytecode[1];
    nBBBB = *(reinterpret_cast<std::uint16_t *>(&this->bytecode[2]));
}


Instruction20bcProvider::Instruction20bcProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION20BC, 4) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    nAA = this->bytecode[1];
    nBBBB = *(reinterpret_cast<std::uint16_t *>(&this->bytecode[2]));
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

std::string_view Instruction20bcProvider::print_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += std::to_string(nAA);
        instruction += ", kind@" + std::to_string(nBBBB);
    }
    return instruction;
}


std::string Instruction20bcProvider::print_instruction_string() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += std::to_string(nAA);
        instruction += ", kind@" + std::to_string(nBBBB);
    }
    return instruction;
}


Instruction22xProvider::Instruction22xProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22X, 4) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    vAA = this->bytecode[1];
    vBBBB = *(reinterpret_cast<std::uint16_t *>(&this->bytecode[2]));
}

Instruction22xProvider::Instruction22xProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22X, 4) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    vAA = this->bytecode[1];
    vBBBB = *(reinterpret_cast<std::uint16_t *>(&this->bytecode[2]));
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

std::string_view Instruction22xProvider::print_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vAA);
        instruction += ", v" + std::to_string(vBBBB);
    }
    return instruction;
}


std::string Instruction22xProvider::print_instruction_string() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vAA);
        instruction += ", v" + std::to_string(vBBBB);
    }
    return instruction;
}


Instruction21tProvider::Instruction21tProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION21T, 4) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    vAA = this->bytecode[1];
    nBBBB = *(reinterpret_cast<std::int16_t *>(&this->bytecode[2]));
}

Instruction21tProvider::Instruction21tProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION21T, 4) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    vAA = this->bytecode[1];
    nBBBB = *(reinterpret_cast<std::int16_t *>(&this->bytecode[2]));
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

std::string_view Instruction21tProvider::print_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vAA);
        instruction += ", " + std::to_string(nBBBB);
    }
    return instruction;
}

std::string Instruction21tProvider::print_instruction_string() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vAA);
        instruction += ", " + std::to_string(nBBBB);
    }
    return instruction;
}


Instruction21sProvider::Instruction21sProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION21S, 4) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    vAA = this->bytecode[1];
    nBBBB = *(reinterpret_cast<std::int16_t *>(&this->bytecode[2]));
}

Instruction21sProvider::Instruction21sProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION21S, 4) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    vAA = this->bytecode[1];
    nBBBB = *(reinterpret_cast<std::int16_t *>(&this->bytecode[2]));
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

std::string_view Instruction21sProvider::print_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vAA);
        instruction += ", " + std::to_string(nBBBB);
    }
    return instruction;
}

std::string Instruction21sProvider::print_instruction_string() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vAA);
        instruction += ", " + std::to_string(nBBBB);
    }
    return instruction;
}

Instruction21hProvider::Instruction21hProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION21S, 4) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    vAA = this->bytecode[1];
    nBBBB = *(reinterpret_cast<std::int16_t *>(&this->bytecode[2]));
    switch (opcode) {
        case disassembler::opcodes::OP_CONST_HIGH16:
            nBBBB = nBBBB << 16;
            break;
        case disassembler::opcodes::OP_CONST_WIDE_HIGH16:
            nBBBB = nBBBB << 48;
            break;
        default:
            break;
    }
}

Instruction21hProvider::Instruction21hProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION21S, 4) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    vAA = this->bytecode[1];
    nBBBB = *(reinterpret_cast<std::int16_t *>(&this->bytecode[2]));
    switch (opcode) {
        case disassembler::opcodes::OP_CONST_HIGH16:
            nBBBB = nBBBB << 16;
            break;
        case disassembler::opcodes::OP_CONST_WIDE_HIGH16:
            nBBBB = nBBBB << 48;
            break;
        default:
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

std::string_view Instruction21hProvider::print_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vAA);
        instruction += ", " + std::to_string(nBBBB);
    }
    return instruction;
}

std::string Instruction21hProvider::print_instruction_string() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vAA);
        instruction += ", " + std::to_string(nBBBB);
    }
    return instruction;
}

Instruction21cProvider::Instruction21cProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION21C, 4) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    vAA = this->bytecode[1];
    iBBBB = *(reinterpret_cast<std::uint16_t *>(&this->bytecode[2]));
    source_id = std::monostate{};
}

Instruction21cProvider::Instruction21cProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION21C, 4) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    vAA = this->bytecode[1];
    iBBBB = *(reinterpret_cast<std::uint16_t *>(&this->bytecode[2]));
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

std::string_view Instruction21cProvider::print_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vAA);
        instruction += ", " + get_kind_type_as_string(source_id, iBBBB);
    }
    return instruction;
}

std::string Instruction21cProvider::print_instruction_string() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vAA);
        instruction += ", " + get_kind_type_as_string(source_id, iBBBB);
    }
    return instruction;
}

Instruction23xProvider::Instruction23xProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION23X, 4) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    vAA = this->bytecode[1];
    vBB = this->bytecode[2];
    vCC = this->bytecode[3];
}

Instruction23xProvider::Instruction23xProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION23X, 4) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    vAA = this->bytecode[1];
    vBB = this->bytecode[2];
    vCC = this->bytecode[3];
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

std::string_view Instruction23xProvider::print_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vAA);
        instruction += ", v" + std::to_string(vBB);
        instruction += ", v" + std::to_string(vCC);
    }
    return instruction;
}

std::string Instruction23xProvider::print_instruction_string() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vAA);
        instruction += ", v" + std::to_string(vBB);
        instruction += ", v" + std::to_string(vCC);
    }
    return instruction;
}

Instruction22bProvider::Instruction22bProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22B, 4) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    vAA = this->bytecode[1];
    vBB = this->bytecode[2];
    nCC = static_cast<std::int8_t>(this->bytecode[3]);
}

Instruction22bProvider::Instruction22bProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22B, 4) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    vAA = this->bytecode[1];
    vBB = this->bytecode[2];
    nCC = static_cast<std::int8_t>(this->bytecode[3]);
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

std::string_view Instruction22bProvider::print_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vAA);
        instruction += ", v" + std::to_string(vBB);
        instruction += ", " + std::to_string(nCC);
    }
    return instruction;
}

std::string Instruction22bProvider::print_instruction_string() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vAA);
        instruction += ", v" + std::to_string(vBB);
        instruction += ", " + std::to_string(nCC);
    }
    return instruction;
}


Instruction22tProvider::Instruction22tProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22T, 4) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    vA = this->bytecode[1] & 0x0F;
    vB = (this->bytecode[1] & 0xF0) >> 4;
    nCCCC = static_cast<std::int16_t>(this->bytecode[2]);
}

Instruction22tProvider::Instruction22tProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22T, 4) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    vA = this->bytecode[1] & 0x0F;
    vB = (this->bytecode[1] & 0xF0) >> 4;
    nCCCC = static_cast<std::int16_t>(this->bytecode[2]);
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

std::string_view Instruction22tProvider::print_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vA);
        instruction += ", v" + std::to_string(vB);
        instruction += ", " + std::to_string(nCCCC);
    }
    return instruction;
}

std::string Instruction22tProvider::print_instruction_string() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vA);
        instruction += ", v" + std::to_string(vB);
        instruction += ", " + std::to_string(nCCCC);
    }
    return instruction;
}

Instruction22sProvider::Instruction22sProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22S, 4) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    vA = this->bytecode[1] & 0x0F;
    vB = (this->bytecode[1] & 0xF0) >> 4;
    nCCCC = static_cast<std::int16_t>(this->bytecode[2]);
}

Instruction22sProvider::Instruction22sProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22S, 4) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    vA = this->bytecode[1] & 0x0F;
    vB = (this->bytecode[1] & 0xF0) >> 4;
    nCCCC = static_cast<std::int16_t>(this->bytecode[2]);
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

std::string_view Instruction22sProvider::print_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vA);
        instruction += ", v" + std::to_string(vB);
        instruction += ", " + std::to_string(nCCCC);
    }
    return instruction;
}

std::string Instruction22sProvider::print_instruction_string() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vA);
        instruction += ", v" + std::to_string(vB);
        instruction += ", " + std::to_string(nCCCC);
    }
    return instruction;
}

Instruction22cProvider::Instruction22cProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22C, 4) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    vA = this->bytecode[1] & 0x0F;
    vB = (this->bytecode[1] & 0xF0) >> 4;
    iCCCC = static_cast<std::uint16_t>(this->bytecode[2]);
    checked_id = std::monostate{};
}

Instruction22cProvider::Instruction22cProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22C, 4) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    vA = this->bytecode[1] & 0x0F;
    vB = (this->bytecode[1] & 0xF0) >> 4;
    iCCCC = static_cast<std::uint16_t>(this->bytecode[2]);
    checked_id = std::monostate{};

    switch (get_kind()) {
        case disassembler::kind::TYPE:
            checked_id = dex.get_type_by_id(iCCCC);
            break;
        case disassembler::kind::FIELD:
            checked_id = dex.get_field_by_id(iCCCC);
            break;
        default:
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

std::string_view Instruction22cProvider::print_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vA);
        instruction += ", v" + std::to_string(vB);
        instruction += ", " + get_kind_type_as_string(checked_id, iCCCC);
    }
    return instruction;
}

std::string Instruction22cProvider::print_instruction_string() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vA);
        instruction += ", v" + std::to_string(vB);
        instruction += ", " + get_kind_type_as_string(checked_id, iCCCC);
    }
    return instruction;
}

Instruction22csProvider::Instruction22csProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22CS, 4) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    vA = this->bytecode[1] & 0x0F;
    vB = (this->bytecode[1] & 0xF0) >> 4;
    iCCCC = static_cast<std::uint16_t>(this->bytecode[2]);
    field = std::monostate{};
}

Instruction22csProvider::Instruction22csProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22CS, 4) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    vA = this->bytecode[1] & 0x0F;
    vB = (this->bytecode[1] & 0xF0) >> 4;
    iCCCC = static_cast<std::uint16_t>(this->bytecode[2]);

    switch (get_kind()) {
        case disassembler::kind::FIELD:
            field = dex.get_field_by_id(iCCCC);
            break;
        default:
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

std::string_view Instruction22csProvider::print_instruction() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vA);
        instruction += ", v" + std::to_string(vB);
        instruction += ", " + get_kind_type_as_string(field, iCCCC);
    }
    return instruction;
}

std::string Instruction22csProvider::print_instruction_string() {
    if (instruction.empty()) {
        instruction = opcode_names.at(opcode);
        instruction += " ";
        instruction += "v" + std::to_string(vA);
        instruction += ", v" + std::to_string(vB);
        instruction += ", " + get_kind_type_as_string(field, iCCCC);
    }
    return instruction;
}


Instruction30tProvider::Instruction30tProvider(std::span<uint8_t> bytecode, std::size_t index) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION30T, 6) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    nAAAAAAAA = static_cast<std::int32_t>(this->bytecode[2]);
}

Instruction30tProvider::Instruction30tProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
        InstructionProvider(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION30T, 6) {
    opcode = static_cast<disassembler::opcodes>(this->bytecode[0]);
    nAAAAAAAA = static_cast<std::int32_t>(this->bytecode[2]);
}

std::int32_t Instruction30tProvider::getNAAAAAAAA() const {
    return nAAAAAAAA;
}

disassembler::operand_type Instruction30tProvider::get_nAAAAAAAA_type() const {
    return disassembler::OFFSET;
}

std::string_view Instruction30tProvider::print_instruction() {
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

std::string Instruction30tProvider::print_instruction_string() {
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



