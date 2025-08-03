//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

// sdk headers
#include "shuriken/sdk/dex/instruction.hpp"
#include "shuriken/sdk/dex/disassembly_constants.hpp"
#include "shuriken/sdk/dex/method.hpp"
// internal headers
#include "shuriken/internal/engine/dex/dex_engine.hpp"
// Standard headers
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace shuriken::dex {

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
#define SIDE_EFFECT(OP) \
{OP},

#include "definitions/side_effect_opcodes.def"
    };

    static const std::vector<disassembler::opcodes> may_throw_opcodes{
#define MAY_THROW(OP) \
{OP},

#include "definitions/may_throw_opcodes.def"
    };

    std::string get_kind_type_as_string(const kind_type_t &source_id, std::uint16_t iBBBB) {
        std::stringstream instruction_str;

        if (std::holds_alternative<std::monostate>(source_id)) {
            instruction_str << " // UNKNOWN@" << iBBBB;
        } else if (std::holds_alternative<DVMType *>(source_id)) {
            auto *type = std::get<DVMType *>(source_id);
            instruction_str << shuriken::dex::get_dalvik_format(*type);
            instruction_str << " // type@" << std::setfill('0') << std::setw(4) << iBBBB;
        } else if (std::holds_alternative<FieldID *>(source_id)) {
            auto *field = std::get<FieldID *>(source_id);
            instruction_str << get_dalvik_format(field->get_class()) << "->";
            instruction_str << field->get_name();
            instruction_str << " // field@" << std::setfill('0') << std::setw(4) << iBBBB;
        } else if (std::holds_alternative<MethodID *>(source_id)) {
            auto *method = std::get<MethodID *>(source_id);
            instruction_str << get_dalvik_format(method->get_class()) << "->";
            instruction_str << method->get_name();
            instruction_str << method->get_prototype().get_descriptor();
            instruction_str << " // method@" << std::setfill('0') << std::setw(4) << iBBBB;
        } else if (std::holds_alternative<DVMPrototype *>(source_id)) {
            auto *proto = std::get<DVMPrototype *>(source_id);
            instruction_str << proto->get_shorty_idx();
            instruction_str << " // proto@" << std::setfill('0') << std::setw(4) << iBBBB;
        } else if (std::holds_alternative<std::string_view>(source_id)) {
            auto str = std::get<std::string_view>(source_id);
            instruction_str << "\"" << str << "\"";
            instruction_str << " // string@" << std::setfill('0') << std::setw(4) << iBBBB;
        }

        return instruction_str.str();
    }
}

class Instruction::Impl {
private:
    disassembler::dexinsttype instruction_type;
protected:
    std::span<std::uint8_t> op_codes;
    std::uint32_t length;
    disassembler::opcodes opcode;
    std::uint64_t address;
    std::string instruction;

    virtual std::string_view format_instruction() = 0;

    virtual void invalidate_instruction(std::string_view err_msg) {
        error_message = err_msg;
        is_valid = false;
    }

    bool is_valid = true;
    std::string error_message;
public:

    Impl([[maybe_unused]] std::span<std::uint8_t> bytecode,
         [[maybe_unused]] std::size_t index,
         disassembler::dexinsttype instruction_type) :
            instruction_type(instruction_type), op_codes({}), length(0), opcode(disassembler::opcodes::OP_NONE) {
    }

    Impl(std::span<std::uint8_t> bytecode, std::size_t index,
         disassembler::dexinsttype instruction_type, std::uint32_t length)
            : instruction_type(instruction_type),
              op_codes({bytecode.begin() + index, bytecode.begin() + index + length}),
              length(length), opcode(disassembler::opcodes::OP_NONE) {
    }

    virtual ~Impl() = default;


    // Virtual methods that derived Impl classes can override
    virtual disassembler::kind get_kind() const {
        auto it = opcodes_kind_map.find(opcode);
        return it != opcodes_kind_map.end() ? it->second : disassembler::kind::NONE_KIND;
    }


    virtual disassembler::dexinsttype get_instruction_type() const {
        return instruction_type;
    }

    virtual disassembler::operation_type get_operation_type() const {
        if (opcodes_operation_type.find(opcode) == opcodes_operation_type.end())
            return disassembler::operation_type::NONE_TYPE;
        return opcodes_operation_type.at(opcode);
    }

    virtual bool is_jump_instruction() const {
        auto operation_type = get_operation_type();
        return (operation_type == disassembler::operation_type::CONDITIONAL_BRANCH_DVM_OPCODE
                || operation_type == disassembler::operation_type::UNCONDITIONAL_BRANCH_DVM_OPCODE
                || operation_type == disassembler::operation_type::MULTI_BRANCH_DVM_OPCODE);
    }

    virtual std::uint32_t get_instruction_length() const {
        return length;
    }

    virtual disassembler::opcodes get_instruction_opcode() const {
        return opcode;
    }

    virtual void set_address(std::uint64_t address) {
        this->address = address;
    }

    virtual std::uint64_t get_address() const {
        return address;
    }

    virtual std::span<std::uint8_t> get_instruction_bytecode() const {
        return op_codes;
    }

    virtual bool is_terminator() const {
        if (!opcodes_operation_type.contains(opcode))
            return false;
        auto operation = opcodes_operation_type.at(opcode);
        if (operation == disassembler::operation_type::CONDITIONAL_BRANCH_DVM_OPCODE
            || operation == disassembler::operation_type::UNCONDITIONAL_BRANCH_DVM_OPCODE
            || operation == disassembler::operation_type::RET_BRANCH_DVM_OPCODE
            || operation == disassembler::operation_type::MULTI_BRANCH_DVM_OPCODE)
            return true;
        return false;
    }

    virtual bool has_side_effects() const {
        if (std::find(side_effects_opcodes.begin(), side_effects_opcodes.end(), opcode) != side_effects_opcodes.end())
            return true;
        return false;
    }

    virtual bool may_throw() const {
        if (std::find(may_throw_opcodes.begin(), may_throw_opcodes.end(), opcode) != may_throw_opcodes.end())
            return true;
        return false;
    }


    virtual bool is_instruction_valid() const {
        return is_valid;
    }

    virtual const std::string &get_error_message() const {
        return error_message;
    }

    virtual std::string_view print_instruction() = 0;

    virtual std::string print_instruction_string() = 0;
};

class DalvikIncorrectInstruction::Impl : public Instruction::Impl {
protected:
    std::string_view format_instruction() {
        if (instruction.empty()) {
            instruction = "DalvikIncorrectInstructionProvider [error_message=";
            instruction += error_message + ", address=";
            instruction += std::to_string(address) + ", opcode=";
            instruction += opcode_names.at(opcode) + "]";
        }
        return instruction;
    }
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index,
         std::string_view error_message,
         size_t size_instr,
         std::uint64_t address,
         disassembler::opcodes opcode) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_DALVIKINCORRECT, size_instr) {
        this->address = address;
        this->error_message = error_message;
        this->opcode = opcode;
    }

    ~Impl() = default;

    std::string_view print_instruction() {
        return format_instruction();
    }

    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class Instruction00x::Impl : public Instruction::Impl {
protected:
    std::string_view format_instruction() {
        if (instruction.empty())
            instruction = opcode_names.at(opcode);
        return instruction;
    }
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION00X) {
    }

    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION00X) {
    }

    ~Impl() = default;

    std::string_view print_instruction() {
        return format_instruction();
    }


    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class Instruction10x::Impl : public Instruction::Impl {
protected:
    std::string_view format_instruction() {
        if (instruction.empty())
            instruction = opcode_names.at(opcode);
        return instruction;
    }
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION10X, 2) {

        if (this->op_codes[1] != 0) {
            invalidate_instruction("Instruction10x high byte should be 0");
            return;
        }

        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    }

    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION10X, 2) {

        if (this->op_codes[1] != 0) {
            invalidate_instruction("Instruction10x high byte should be 0");
            return;
        }

        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
    }

    ~Impl() = default;

    std::string_view print_instruction() {
        return format_instruction();
    }

    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class Instruction12x::Impl : public Instruction::Impl {
private:
    std::uint8_t vA;
    std::uint8_t vB;
protected:
    std::string_view format_instruction() {
        if (instruction.empty()) {
            instruction = opcode_names.at(opcode);
            instruction += " ";
            instruction += "v" + std::to_string(vA);
            instruction += ", ";
            instruction += "v" + std::to_string(vB);
        }
        return instruction;
    }
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION12X, 2) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vA = (this->op_codes[1] & 0x0F);
        vB = (this->op_codes[1] & 0xF0) >> 4;
    }
    
    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION12X, 2) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vA = (this->op_codes[1] & 0x0F);
        vB = (this->op_codes[1] & 0xF0) >> 4;
    }

    ~Impl() = default;

    std::uint8_t getVA() const {
        return vA;
    }


    disassembler::operand_type get_vA_type() const {
        return disassembler::operand_type::REGISTER;
    }


    std::uint8_t getVB() const {
        return vB;
    }


    disassembler::operand_type get_vB_types() const {
        return disassembler::operand_type::REGISTER;
    }

    std::string_view print_instruction() {
        return format_instruction();
    }


    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class Instruction11n::Impl : public Instruction::Impl {
private:
    std::uint8_t vA;
    std::int8_t nB;
protected:
    std::string_view format_instruction() {
        if (instruction.empty()) {
            instruction = opcode_names.at(opcode);
            instruction += " ";
            instruction += "v" + std::to_string(vA);
            instruction += ", ";
            instruction += std::to_string(nB);
        }
        return instruction;
    }
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION11N, 2) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vA = (this->op_codes[1] & 0x0F);
        nB = static_cast<std::int8_t>((this->op_codes[1] & 0xF0) >> 4);
    }

    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION11N, 2) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vA = (this->op_codes[1] & 0x0F);
        nB = static_cast<std::int8_t>((this->op_codes[1] & 0xF0) >> 4);
    }

    ~Impl() = default;

    std::uint8_t getVA() const {
        return vA;
    }


    disassembler::operand_type get_vA_type() const {
        return disassembler::operand_type::REGISTER;
    }

    std::int8_t getNB() const {
        return nB;
    }


    disassembler::operand_type get_nB_types() const {
        return disassembler::operand_type::LITERAL;
    }

    std::string_view print_instruction() {
        return format_instruction();
    }


    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class Instruction11x::Impl : public Instruction::Impl {
private:
    std::uint8_t vAA;
protected:
    std::string_view format_instruction() {
        if (instruction.empty()) {
            instruction = opcode_names.at(opcode);
            instruction += " ";
            instruction += "v" + std::to_string(vAA);
        }
        return instruction;
    }
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION11X, 2) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vAA = this->op_codes[1];
    }


    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION11X, 2) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vAA = this->op_codes[1];
    }

    ~Impl() = default;

    std::uint8_t getVAA() const {
        return vAA;
    }


    disassembler::operand_type get_vAA_type() const {
        return disassembler::operand_type::REGISTER;
    }


    std::string_view print_instruction() {
        return format_instruction();
    }


    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class Instruction10t::Impl : public Instruction::Impl {
private:
    std::int8_t nAA;
protected:
    std::string_view format_instruction() {
        if (instruction.empty()) {
            std::stringstream str;
            str << opcode_names.at(opcode) << " ";
            str << "0x" << std::hex << ((nAA * 2) + static_cast<std::int64_t>(address));
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
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION10T, 2) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        nAA = static_cast<std::int8_t>(this->op_codes[1]);
    }

    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION10T, 2) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        nAA = static_cast<std::int8_t>(this->op_codes[1]);
    }

    ~Impl() = default;

    std::int8_t getNAA() const {
        return nAA;
    }

    disassembler::operand_type get_nAA_type() const {
        return disassembler::operand_type::OFFSET;
    }

    std::string_view print_instruction() {
        return format_instruction();
    }

    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class Instruction20t::Impl : public Instruction::Impl {
private:
    std::int16_t nAAAA;
protected:
    std::string_view format_instruction() {
        if (instruction.empty()) {
            std::stringstream str;
            str << opcode_names.at(opcode) << " ";
            str << "0x" << std::hex << ((nAAAA * 2) + static_cast<std::int64_t>(address));
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
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION20T, 4) {
        if (op_codes[1] != 0) {
            invalidate_instruction("Error reading Instruction20t padding must be 0");
            return;
        }
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        nAAAA = *(reinterpret_cast<std::int16_t *>(&this->op_codes[2]));
    }

    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION20T, 4) {
        if (op_codes[1] != 0) {
            invalidate_instruction("Error reading Instruction20t padding must be 0");
            return;
        }
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        nAAAA = *(reinterpret_cast<std::int16_t *>(&this->op_codes[2]));
    }

    ~Impl() = default;

    std::int16_t getNAAAA() const {
        return nAAAA;
    }

    disassembler::operand_type get_nAAAA_type() const {
        return disassembler::operand_type::OFFSET;
    }

    std::string_view print_instruction() {
        return format_instruction();
    }

    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class Instruction20bc::Impl : public Instruction::Impl {
private:
    std::uint8_t nAA;
    std::uint16_t nBBBB;
protected:
    std::string_view format_instruction() {
        if (instruction.empty()) {
            instruction = opcode_names.at(opcode);
            instruction += " ";
            instruction += std::to_string(nAA);
            instruction += ", kind@" + std::to_string(nBBBB);
        }
        return instruction;
    }
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION20BC, 4) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        nAA = this->op_codes[1];
        nBBBB = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[2]));
    }

    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION20BC, 4) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        nAA = this->op_codes[1];
        nBBBB = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[2]));
    }

    ~Impl() = default;

    std::uint8_t getNAA() const {
        return nAA;
    }

    disassembler::operand_type get_nAA_type() const {
        return disassembler::operand_type::LITERAL;
    }

    std::uint16_t getNBBBB() const {
        return nBBBB;
    }

    disassembler::operand_type get_nBBBB_type() const {
        return disassembler::operand_type::KIND;
    }

    std::string_view print_instruction() {
        return format_instruction();
    }


    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class Instruction22x::Impl : public Instruction::Impl {
private:
    std::uint8_t vAA;
    std::uint16_t vBBBB;
protected:
    std::string_view format_instruction() {
        if (instruction.empty()) {
            instruction = opcode_names.at(opcode);
            instruction += " ";
            instruction += "v" + std::to_string(vAA);
            instruction += ", v" + std::to_string(vBBBB);
        }
        return instruction;
    }
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22X, 4) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vAA = this->op_codes[1];
        vBBBB = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[2]));
    }

    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22X, 4) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vAA = this->op_codes[1];
        vBBBB = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[2]));
    }

    ~Impl() = default;

    std::uint8_t getVAA() const {
        return vAA;
    }


    disassembler::operand_type get_vAA_type() const {
        return disassembler::operand_type::REGISTER;
    }


    std::uint16_t getVBBBB() const {
        return vBBBB;
    }

    disassembler::operand_type get_vBBBB_type() const {
        return disassembler::operand_type::REGISTER;
    }

    std::string_view print_instruction() {
        return format_instruction();
    }

    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class Instruction21t::Impl : public Instruction::Impl {
private:
    std::uint8_t vAA;
    std::int16_t nBBBB;
protected:
    std::string_view format_instruction() {
        if (instruction.empty()) {
            instruction = opcode_names.at(opcode);
            instruction += " ";
            instruction += "v" + std::to_string(vAA);
            instruction += ", " + std::to_string(nBBBB);
        }
        return instruction;
    }
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION21T, 4) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vAA = this->op_codes[1];
        nBBBB = *(reinterpret_cast<std::int16_t *>(&this->op_codes[2]));

        if (nBBBB == 0) {
            invalidate_instruction("Error reading Instruction21t offset cannot be 0");
        }
    }

    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION21T, 4) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vAA = this->op_codes[1];
        nBBBB = *(reinterpret_cast<std::int16_t *>(&this->op_codes[2]));
        if (nBBBB == 0) {
            invalidate_instruction("Error reading Instruction21t offset cannot be 0");
        }
    }

    ~Impl() = default;

    std::uint8_t getVAA() const {
        return vAA;
    }

    disassembler::operand_type get_vAA_type() const {
        return disassembler::REGISTER;
    }

    std::int16_t getNBBBB() const {
        return nBBBB;
    }

    disassembler::operand_type get_nBBBB_type() const {
        return disassembler::OFFSET;
    }

    std::string_view print_instruction() {
        return format_instruction();
    }

    std::string print_instruction_string() {
        return std::string(format_instruction());
    }

};

class Instruction21s::Impl : public Instruction::Impl {
private:
    std::uint8_t vAA;
    std::int16_t nBBBB;
protected:
    std::string_view format_instruction() {
        if (instruction.empty()) {
            instruction = opcode_names.at(opcode);
            instruction += " ";
            instruction += "v" + std::to_string(vAA);
            instruction += ", " + std::to_string(nBBBB);
        }
        return instruction;
    }
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION21S, 4) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vAA = this->op_codes[1];
        nBBBB = *(reinterpret_cast<std::int16_t *>(&this->op_codes[2]));
    }

    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION21S, 4) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vAA = this->op_codes[1];
        nBBBB = *(reinterpret_cast<std::int16_t *>(&this->op_codes[2]));
    }

    ~Impl() = default;

    std::uint8_t getVAA() const {
        return vAA;
    }

    disassembler::operand_type get_vAA_type() const {
        return disassembler::REGISTER;
    }

    std::int16_t getNBBBB() const {
        return nBBBB;
    }

    disassembler::operand_type get_nBBBB_type() const {
        return disassembler::OFFSET;
    }

    std::string_view print_instruction() {
        return format_instruction();
    }

    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class Instruction21h::Impl : public Instruction::Impl {
private:
    std::uint8_t vAA;
    std::int64_t nBBBB;
protected:
    std::string_view format_instruction() {
        if (instruction.empty()) {
            instruction = opcode_names.at(opcode);
            instruction += " ";
            instruction += "v" + std::to_string(vAA);
            instruction += ", " + std::to_string(nBBBB);
        }
        return instruction;
    }
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION21S, 4) {
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

    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION21S, 4) {
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

    ~Impl() = default;

    std::uint8_t getVAA() const {
        return vAA;
    }

    disassembler::operand_type get_vAA_type() const {
        return disassembler::REGISTER;
    }

    std::int64_t getnBBBB() const {
        return nBBBB;
    }

    disassembler::operand_type get_nBBBB_type() const {
        return disassembler::LITERAL;
    }

    std::string_view print_instruction() {
        return format_instruction();
    }

    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class Instruction21c::Impl : public Instruction::Impl {
private:
    std::uint8_t vAA;
    std::uint16_t iBBBB;
    kind_type_t source_id;
protected:
    std::string_view format_instruction() {
        if (instruction.empty()) {
            instruction = opcode_names.at(opcode);
            instruction += " ";
            instruction += "v" + std::to_string(vAA);
            instruction += ", " + get_kind_type_as_string(source_id, iBBBB);
        }
        return instruction;
    }
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION21C, 4) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vAA = this->op_codes[1];
        iBBBB = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[2]));
        source_id = std::monostate{};
    }

    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION21C, 4) {
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

    ~Impl() = default;

    std::uint8_t getVAA() const {
        return vAA;
    }

    disassembler::operand_type get_vAA_type() const {
        return disassembler::REGISTER;
    }

    std::uint16_t getIBBBB() const {
        return iBBBB;
    }

    disassembler::operand_type get_iBBBB_type() const {
        return disassembler::KIND;
    }

    kind_type_t get_iBBBB_kind() {
        return source_id;
    }

    std::string_view print_instruction() {
        return format_instruction();
    }

    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class Instruction23x::Impl : public Instruction::Impl {
private:
    std::uint8_t vAA;
    std::uint8_t vBB;
    std::uint8_t vCC;
protected:
    std::string_view format_instruction() {
        if (instruction.empty()) {
            instruction = opcode_names.at(opcode);
            instruction += " ";
            instruction += "v" + std::to_string(vAA);
            instruction += ", v" + std::to_string(vBB);
            instruction += ", v" + std::to_string(vCC);
        }
        return instruction;
    }
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION23X, 4) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vAA = this->op_codes[1];
        vBB = this->op_codes[2];
        vCC = this->op_codes[3];
    }

    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION23X, 4) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vAA = this->op_codes[1];
        vBB = this->op_codes[2];
        vCC = this->op_codes[3];
    }

    ~Impl() = default;

    std::uint8_t getVAA() const {
        return vAA;
    }

    disassembler::operand_type get_vAA_type() const {
        return disassembler::REGISTER;
    }

    std::uint8_t getVBB() const {
        return vBB;
    }

    disassembler::operand_type get_vBB_type() const {
        return disassembler::REGISTER;
    }

    std::uint8_t getVCC() const {
        return vCC;
    }

    disassembler::operand_type get_vCC_type() const {
        return disassembler::REGISTER;
    }

    std::string_view print_instruction() {
        return format_instruction();
    }

    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class Instruction22b::Impl : public Instruction::Impl {
private:
    std::uint8_t vAA;
    std::uint8_t vBB;
    std::int8_t nCC;
protected:
    std::string_view format_instruction() {
        if (instruction.empty()) {
            instruction = opcode_names.at(opcode);
            instruction += " ";
            instruction += "v" + std::to_string(vAA);
            instruction += ", v" + std::to_string(vBB);
            instruction += ", " + std::to_string(nCC);
        }
        return instruction;
    }
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22B, 4) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vAA = this->op_codes[1];
        vBB = this->op_codes[2];
        nCC = static_cast<std::int8_t>(this->op_codes[3]);
    }

    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22B, 4) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vAA = this->op_codes[1];
        vBB = this->op_codes[2];
        nCC = static_cast<std::int8_t>(this->op_codes[3]);
    }

    ~Impl() = default;

    std::uint8_t getVAA() const {
        return vAA;
    }

    disassembler::operand_type get_vAA_type() const {
        return disassembler::REGISTER;
    }

    std::uint8_t getVBB() const {
        return vBB;
    }

    disassembler::operand_type get_vBB_type() const {
        return disassembler::REGISTER;
    }

    std::int8_t getNCC() const {
        return nCC;
    }

    disassembler::operand_type get_nCC_type() const {
        return disassembler::LITERAL;
    }

    std::string_view print_instruction() {
        return format_instruction();
    }

    std::string print_instruction_string() {
        return std::string(format_instruction());
    }

};

class Instruction22t::Impl : public Instruction::Impl {
private:
    std::uint8_t vA;
    std::uint8_t vB;
    std::int16_t nCCCC;
protected:
    std::string_view format_instruction() {
        if (instruction.empty()) {
            instruction = opcode_names.at(opcode);
            instruction += " ";
            instruction += "v" + std::to_string(vA);
            instruction += ", v" + std::to_string(vB);
            instruction += ", " + std::to_string(nCCCC);
        }
        return instruction;
    }
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22T, 4) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vA = this->op_codes[1] & 0x0F;
        vB = (this->op_codes[1] & 0xF0) >> 4;
        nCCCC = static_cast<std::int16_t>(this->op_codes[2]);

        if (nCCCC == 0)
            invalidate_instruction("Error reading Instruction22t offset cannot be 0");
    }

    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22T, 4) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vA = this->op_codes[1] & 0x0F;
        vB = (this->op_codes[1] & 0xF0) >> 4;
        nCCCC = static_cast<std::int16_t>(this->op_codes[2]);

        if (nCCCC == 0)
            invalidate_instruction("Error reading Instruction22t offset cannot be 0");
    }

    ~Impl() = default;
    
    std::uint8_t getVA() const {
        return vA;
    }

    disassembler::operand_type get_vA_type() const {
        return disassembler::REGISTER;
    }

    std::uint8_t getVB() const {
        return vB;
    }

    disassembler::operand_type get_vB_type() const {
        return disassembler::REGISTER;
    }

    std::int16_t getNCCCC() const {
        return nCCCC;
    }

    disassembler::operand_type get_nCCCC_type() const {
        return disassembler::OFFSET;
    }

    std::string_view print_instruction() {
        return format_instruction();
    }

    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class Instruction22s::Impl : public Instruction::Impl {
private:
    std::uint8_t vA;
    std::uint8_t vB;
    std::int16_t nCCCC;
protected:
    std::string_view format_instruction() {
        if (instruction.empty()) {
            instruction = opcode_names.at(opcode);
            instruction += " ";
            instruction += "v" + std::to_string(vA);
            instruction += ", v" + std::to_string(vB);
            instruction += ", " + std::to_string(nCCCC);
        }
        return instruction;
    }
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22S, 4) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vA = this->op_codes[1] & 0x0F;
        vB = (this->op_codes[1] & 0xF0) >> 4;
        nCCCC = static_cast<std::int16_t>(this->op_codes[2]);
    }
    
    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22S, 4) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vA = this->op_codes[1] & 0x0F;
        vB = (this->op_codes[1] & 0xF0) >> 4;
        nCCCC = static_cast<std::int16_t>(this->op_codes[2]);
    }

    ~Impl() = default;

    std::uint8_t getVA() const {
        return vA;
    }

    disassembler::operand_type get_vA_type() const {
        return disassembler::REGISTER;
    }

    std::uint8_t getVB() const {
        return vB;
    }

    disassembler::operand_type get_vB_type() const {
        return disassembler::REGISTER;
    }

    std::int16_t getNCCCC() const {
        return nCCCC;
    }

    disassembler::operand_type get_nCCCC_type() const {
        return disassembler::LITERAL;
    }

    std::string_view print_instruction() {
        return format_instruction();
    }

    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class Instruction22c::Impl : public Instruction::Impl {
private:
    std::uint8_t vA;
    std::uint8_t vB;
    std::uint16_t iCCCC;
    kind_type_t checked_id;
protected:
    std::string_view format_instruction() {
        if (instruction.empty()) {
            instruction = opcode_names.at(opcode);
            instruction += " ";
            instruction += "v" + std::to_string(vA);
            instruction += ", v" + std::to_string(vB);
            instruction += ", " + get_kind_type_as_string(checked_id, iCCCC);
        }
        return instruction;
    }
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22C, 4) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vA = this->op_codes[1] & 0x0F;
        vB = (this->op_codes[1] & 0xF0) >> 4;
        iCCCC = static_cast<std::uint16_t>(this->op_codes[2]);
        checked_id = std::monostate{};
    }

    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22C, 4) {
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

    ~Impl() = default;

    std::uint8_t getVA() const {
        return vA;
    }

    disassembler::operand_type get_vA_type() const {
        return disassembler::REGISTER;
    }

    std::uint8_t getVB() const {
        return vB;
    }

    disassembler::operand_type get_vB_type() const {
        return disassembler::REGISTER;
    }

    std::uint16_t getICCCC() const {
        return iCCCC;
    }

    disassembler::operand_type get_iCCCC_type() const {
        return disassembler::KIND;
    }

    kind_type_t get_checked_id_as_kind() const {
        return checked_id;
    }

    std::string_view print_instruction() {
        return format_instruction();
    }

    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class Instruction22cs::Impl : public Instruction::Impl {
private:
    std::uint8_t vA;
    std::uint8_t vB;
    std::uint16_t iCCCC;
    kind_type_t field;
protected:
    std::string_view format_instruction() {
        if (instruction.empty()) {
            instruction = opcode_names.at(opcode);
            instruction += " ";
            instruction += "v" + std::to_string(vA);
            instruction += ", v" + std::to_string(vB);
            instruction += ", " + get_kind_type_as_string(field, iCCCC);
        }
        return instruction;
    }
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22CS, 4) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vA = this->op_codes[1] & 0x0F;
        vB = (this->op_codes[1] & 0xF0) >> 4;
        iCCCC = static_cast<std::uint16_t>(this->op_codes[2]);
        field = std::monostate{};
    }

    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION22CS, 4) {
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

    ~Impl() = default;

    std::uint8_t getVA() const {
        return vA;
    }

    disassembler::operand_type get_vA_type() const {
        return disassembler::REGISTER;
    }

    std::uint8_t getVB() const {
        return vB;
    }

    disassembler::operand_type get_vB_type() const {
        return disassembler::REGISTER;
    }

    std::uint16_t getICCCC() const {
        return iCCCC;
    }

    disassembler::operand_type get_iCCCC_type() const {
        return disassembler::KIND;
    }

    kind_type_t get_field() const {
        return field;
    }

    std::string_view print_instruction() {
        return format_instruction();
    }

    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class Instruction30t::Impl : public Instruction::Impl {
private:
    std::int32_t nAAAAAAAA;
protected:
    std::string_view format_instruction() {
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
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION30T, 6) {
        if (op_codes[1]) {
            invalidate_instruction("Error reading Instruction30t padding must be 0");
            return;
        }
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        nAAAAAAAA = static_cast<std::int32_t>(this->op_codes[2]);
        if (nAAAAAAAA == 0)
            invalidate_instruction("Error reading Instruction30t offset cannot be 0");
    }

    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION30T, 6) {
        if (op_codes[1]) {
            invalidate_instruction("Error reading Instruction30t padding must be 0");
            return;
        }
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        nAAAAAAAA = static_cast<std::int32_t>(this->op_codes[2]);
        if (nAAAAAAAA == 0)
            invalidate_instruction("Error reading Instruction30t offset cannot be 0");
    }

    ~Impl() = default;

    std::int32_t getNAAAAAAAA() const {
        return nAAAAAAAA;
    }

    disassembler::operand_type get_nAAAAAAAA_type() const {
        return disassembler::OFFSET;
    }

    std::string_view print_instruction() {
        return format_instruction();
    }

    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class Instruction32x::Impl : public Instruction::Impl {
private:
    std::uint16_t vAAAA;
    std::uint16_t vBBBB;
protected:
    std::string_view format_instruction() {
        if (instruction.empty()) {
            std::stringstream str;
            str << opcode_names.at(opcode) << " ";
            str << "v" << std::to_string(vAAAA) << ", v" << std::to_string(vBBBB);
            instruction = str.str();
        }
        return instruction;
    }
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION32X, 6) {
        if (op_codes[1] != 0) {
            invalidate_instruction("Error reading Instruction32x padding must be 0");
            return;
        }
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vAAAA = static_cast<std::uint16_t>(this->op_codes[2]);
        vBBBB = static_cast<std::uint16_t>(this->op_codes[4]);
    }

    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION32X, 6) {
        if (op_codes[1] != 0) {
            invalidate_instruction("Error reading Instruction32x padding must be 0");
            return;
        }
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vAAAA = static_cast<std::uint16_t>(this->op_codes[2]);
        vBBBB = static_cast<std::uint16_t>(this->op_codes[4]);
    }

    ~Impl() = default;

    std::uint16_t getVAAAA() const {
        return vAAAA;
    }

    disassembler::operand_type get_vAAAA_type() const {
        return disassembler::REGISTER;
    }

    std::uint16_t getVBBBB() const {
        return vBBBB;
    }

    disassembler::operand_type get_vBBBB_type() const {
        return disassembler::REGISTER;
    }

    std::string_view print_instruction() {
        return format_instruction();
    }

    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class Instruction31i::Impl : public Instruction::Impl {
private:
    std::uint8_t vAA;
    std::uint32_t nBBBBBBBB;
protected:
    std::string_view format_instruction() {
        if (instruction.empty()) {
            std::stringstream str;
            str << opcode_names.at(opcode) << " ";
            str << "v" << std::to_string(vAA);
            str << ", " << getNBBBBBBBB_Float() << " // " << std::to_string(nBBBBBBBB);
            instruction = str.str();
        }
        return instruction;
    }
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION31I, 6) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vAA = static_cast<std::uint16_t>(this->op_codes[1]);
        nBBBBBBBB = static_cast<std::uint32_t>(this->op_codes[2]);
    }

    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION31I, 6) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vAA = static_cast<std::uint16_t>(this->op_codes[1]);
        nBBBBBBBB = static_cast<std::uint32_t>(this->op_codes[2]);
    }

    ~Impl() = default;

    std::uint8_t getVAA() const {
        return vAA;
    }

    disassembler::operand_type get_vAA_type() const {
        return disassembler::REGISTER;
    }

    std::uint32_t getNBBBBBBBB() const {
        return nBBBBBBBB;
    }

    float getNBBBBBBBB_Float() const {
        union {
            float f;
            std::uint32_t i;
        } conv;

        conv.i = nBBBBBBBB;
        return conv.f;
    }

    disassembler::operand_type get_nBBBBBBBB_type() const {
        return disassembler::LITERAL;
    }

    std::string_view print_instruction() {
        return format_instruction();
    }

    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class Instruction31t::Impl : public Instruction::Impl {
private:
    std::uint8_t vAA;
    std::int32_t nBBBBBBBB;
    disassembler::type_of_switch_t type_of_switch;
    switch_instr_t switch_instruction;
protected:
    std::string_view format_instruction() {
        if (instruction.empty()) {
            std::stringstream str;
            str << opcode_names.at(opcode) << " ";
            str << "v" << std::to_string(vAA);
            str << ", " << std::to_string(nBBBBBBBB);
            instruction = str.str();
        }
        return instruction;
    }
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION31T, 6) {
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

    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION31T, 6) {
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

    ~Impl() = default;

    std::uint8_t getVAA() const {
        return vAA;
    }

    disassembler::operand_type get_vAA_type() const {
        return disassembler::REGISTER;
    }

    std::int32_t getNBBBBBBBB() const {
        return nBBBBBBBB;
    }

    disassembler::operand_type get_nBBBBBBBB_type() const {
        return disassembler::OFFSET;
    }

    disassembler::type_of_switch_t get_type_of_switch() const {
        return type_of_switch;
    }

    switch_instr_t get_switch() const {
        return switch_instruction;
    }


    void set_packed_switch(PackedSwitch *packed_switch) {
        this->switch_instruction = packed_switch;
    }

    void set_sparse_switch(SparseSwitch *sparse_switch) {
        this->switch_instruction = sparse_switch;
    }

    std::string_view print_instruction() {
        return format_instruction();
    }

    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class Instruction31c::Impl : public Instruction::Impl {
private:
    std::uint8_t vAA;
    std::uint32_t iBBBBBBBB;
    std::string_view pointed_string;
protected:
    std::string_view format_instruction() {
        if (instruction.empty()) {
            std::stringstream str;
            str << opcode_names.at(opcode) << " ";
            str << "v" << std::to_string(vAA);
            str << ", " << std::to_string(iBBBBBBBB);
            if (!pointed_string.empty()) {
                str << " // " << pointed_string;
            }
            instruction = str.str();
        }
        return instruction;
    }
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION31C, 6) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vAA = static_cast<std::uint16_t>(this->op_codes[1]);
        iBBBBBBBB = static_cast<std::uint32_t>(this->op_codes[2]);
    }

    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION31C, 6) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vAA = static_cast<std::uint16_t>(this->op_codes[1]);
        iBBBBBBBB = static_cast<std::uint32_t>(this->op_codes[2]);
        pointed_string = dex.get_string_by_id(iBBBBBBBB);
    }

    ~Impl() = default;

    std::uint8_t getVAA() const {
        return vAA;
    }

    disassembler::operand_type get_vAA_type() const {
        return disassembler::REGISTER;
    }

    std::uint32_t getIBBBBBBBB() const {
        return iBBBBBBBB;
    }

    disassembler::operand_type get_IBBBBBBBB_type() const {
        return disassembler::OFFSET;
    }

    std::string_view get_string_value() const {
        return pointed_string;
    }

    std::string get_string_value_string() const {
        return std::string(pointed_string);
    }

    std::string_view print_instruction() {
        return format_instruction();
    }

    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class Instruction35c::Impl : public Instruction::Impl {
private:
    std::uint8_t array_size;
    std::uint16_t type_index;
    kind_type_t type_value;
    std::vector<std::uint8_t> registers;
protected:
    std::string_view format_instruction() {
        if (instruction.empty()) {
            std::stringstream str;
            str << opcode_names.at(opcode) << " ";
            str << " {";
            for (size_t i = 0, e = registers.size(); i < e; i++) {
                auto reg = registers[i];
                str << "v" << std::to_string(reg);
                if (i < registers.size() - 1)
                    str << ", ";
            }
            str << "}, " + get_kind_type_as_string(type_value, type_index);
            instruction = str.str();
        }
        return instruction;
    }
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION35C, 6) {
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

    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION35C, 6) {
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

    ~Impl() = default;

    std::uint8_t get_number_of_registers() const {
        return array_size;
    }

    std::span<std::uint8_t> get_registers() {
        return std::span{registers};
    }

    disassembler::operand_type get_registers_type() const {
        return disassembler::REGISTER;
    }

    std::uint16_t get_type_idx() const {
        return type_index;
    }

    disassembler::operand_type get_value_type() const {
        return disassembler::KIND;
    }

    disassembler::kind get_value_kind() const {
        return get_kind();
    }

    kind_type_t get_value() const {
        return type_value;
    }

    std::string_view print_instruction() {
        return format_instruction();
    }

    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class Instruction3rc::Impl : public Instruction::Impl {
private:
    std::uint8_t array_size;
    std::uint16_t index;
    kind_type_t index_value;
    std::vector<std::uint16_t> registers;
protected:
    std::string_view format_instruction() {
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
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION3RC, 6) {
        std::uint16_t vCCCC;
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        array_size = this->op_codes[1];
        index = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[2]));
        vCCCC = *(reinterpret_cast<std::uint16_t *>(&this->op_codes[4]));
        index_value = std::monostate{};

        for (std::uint16_t I = vCCCC, E = vCCCC + array_size; I < E; I++)
            registers.push_back(I);
    }

    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION3RC, 6) {
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

    ~Impl() = default;

    std::uint8_t get_registers_size() const {
        return array_size;
    }

    std::uint16_t get_index() const {
        return index;
    }

    kind_type_t get_index_value() const {
        return index_value;
    }

    disassembler::operand_type get_index_type() const {
        return disassembler::KIND;
    }

    std::span<std::uint16_t> get_registers() {
        return std::span<std::uint16_t>{registers};
    }

    std::string_view print_instruction() {
        return format_instruction();
    }

    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class Instruction45cc::Impl : public Instruction::Impl {
private:
    std::uint8_t reg_count;
    std::vector<std::uint8_t> registers;
    std::uint16_t method_reference;
    kind_type_t method_value;
    std::uint16_t prototype_reference;
    kind_type_t prototype_value;
protected:
    std::string_view format_instruction() {
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
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION45CC, 8) {
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

    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION45CC, 8) {
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

    ~Impl() = default;

    std::uint8_t get_number_of_registers() const {
        return reg_count;
    }

    std::span<std::uint8_t> get_registers() {
        return std::span<std::uint8_t>{registers};
    }

    std::uint16_t get_method_reference() const {
        return method_reference;
    }

    kind_type_t get_method_value() const {
        return method_value;
    }

    std::uint16_t get_prototype_reference() const {
        return prototype_reference;
    }

    kind_type_t get_prototype_value() const {
        return prototype_value;
    }

    std::string_view print_instruction() {
        return format_instruction();
    }

    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class Instruction4rcc::Impl : public Instruction::Impl {
private:
    std::uint8_t reg_count;
    std::vector<std::uint16_t> registers;
    std::uint16_t method_reference;
    kind_type_t method_value;
    std::uint16_t prototype_reference;
    kind_type_t prototype_value;
protected:
    std::string_view format_instruction() {
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
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION4RCC, 8) {
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

    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION4RCC, 8) {
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

    ~Impl() = default;

    std::uint8_t get_number_of_registers() const {
        return reg_count;
    }

    std::span<std::uint16_t> get_registers() {
        return std::span<std::uint16_t>{registers};
    }

    std::uint16_t get_method_reference() const {
        return method_reference;
    }

    kind_type_t get_method_value() const {
        return method_value;
    }

    std::uint16_t get_prototype_reference() const {
        return prototype_reference;
    }

    kind_type_t get_prototype_value() const {
        return prototype_value;
    }

    std::string_view print_instruction() {
        return format_instruction();
    }

    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class Instruction51l::Impl : public Instruction::Impl {
private:
    std::uint8_t vAA;
    std::int64_t nBBBBBBBBBBBBBBBB;
protected:
    std::string_view format_instruction() {
        if (instruction.empty()) {
            instruction = opcode_names.at(opcode);
            instruction += " v" + std::to_string(vAA);
            instruction += ", #" + std::to_string(get_nBBBBBBBBBBBBBBBB_double());
            instruction += " // " + std::to_string(nBBBBBBBBBBBBBBBB);
        }
        return instruction;
    }
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION51L, 10) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vAA = this->op_codes[1];
        nBBBBBBBBBBBBBBBB = *(reinterpret_cast<std::int64_t *>(&this->op_codes[2]));
    }

    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_INSTRUCTION51L, 10) {
        opcode = static_cast<disassembler::opcodes>(this->op_codes[0]);
        vAA = this->op_codes[1];
        nBBBBBBBBBBBBBBBB = *(reinterpret_cast<std::int64_t *>(&this->op_codes[2]));
    }

    ~Impl() = default;

    std::uint8_t getVAA() const {
        return vAA;
    }

    disassembler::operand_type get_vAA_type() const {
        return disassembler::REGISTER;
    }

    std::int64_t getNBBBBBBBBBBBBBBBB() const {
        return nBBBBBBBBBBBBBBBB;
    }

    double get_nBBBBBBBBBBBBBBBB_double() const {
        union {
            double d;
            std::int64_t j;
        } conv;

        conv.j = nBBBBBBBBBBBBBBBB;

        return conv.d;
    }

    disassembler::operand_type get_nBBBBBBBBBBBBBBBB_type() const {
        return disassembler::LITERAL;
    }

    std::string_view print_instruction() {
        return format_instruction();
    }

    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class PackedSwitch::Impl : public Instruction::Impl {
private:
    std::uint16_t size;
    std::int32_t first_key;
    std::vector<std::int32_t> targets;
protected:
    std::string_view format_instruction() {
        if (instruction.empty()) {
            std::stringstream data;
            data << "packed-switch-data" << " (size) " << size << " (first/last key) " << first_key << "[";
            for (const auto target: targets)
                data << "0x" << std::hex << target << ",";
            if (size > 0)
                data.seekp(-1, data.cur);
            data << "]";
            instruction = data.str();
        }
        return instruction;
    }
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_PACKEDSWITCH, 8) {
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

    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_PACKEDSWITCH, 8) {
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

    ~Impl() = default;

    std::uint16_t get_number_of_targets() const {
        return size;
    }

    std::int32_t get_first_key() const {
        return first_key;
    }

    std::span<std::int32_t> get_targets() {
        return std::span<std::int32_t>{targets};
    }

    std::string_view print_instruction() {
        return format_instruction();
    }

    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class SparseSwitch::Impl : public Instruction::Impl {
private:
    std::uint16_t size;
    std::vector<std::pair<std::int32_t, std::int32_t>> keys_targets;
protected:
    std::string_view format_instruction() {
        if (instruction.empty()) {
            std::stringstream output;
            output << "sparse-switch-data" << " (size) " << size << "[";
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
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_SPARSESWITCH, 4) {
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

    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_SPARSESWITCH, 4) {
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

    ~Impl() = default;

    std::uint16_t get_size_of_targets() const {
        return size;
    }

    std::span<std::pair<std::int32_t, std::int32_t>> get_keys_targets() {
        return std::span<std::pair<std::int32_t, std::int32_t>>{keys_targets};
    }

    std::string_view print_instruction() {
        return format_instruction();
    }

    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

class FillArrayData::Impl : public Instruction::Impl {
private:
    std::uint16_t element_width;
    std::uint32_t size;
    std::vector<std::uint8_t> data;
protected:
    std::string_view format_instruction() {
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
public:
    Impl(std::span<uint8_t> bytecode, std::size_t index) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_FILLARRAYDATA, 8) {
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

    Impl(std::span<uint8_t> bytecode, std::size_t index, DexEngine &dex) :
            Instruction::Impl(bytecode, index, disassembler::dexinsttype::DEX_FILLARRAYDATA, 8) {
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

    ~Impl() = default;

    std::uint16_t get_element_width() const {
        return element_width;
    }

    std::uint32_t get_size_of_data() const {
        return size;
    }

    std::span<std::uint8_t> get_data() {
        return std::span<std::uint8_t>{data};
    }

    std::string_view print_instruction() {
        return format_instruction();
    }

    std::string print_instruction_string() {
        return std::string(format_instruction());
    }
};

}