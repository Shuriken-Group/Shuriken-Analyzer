//--------------------------------------------------------------------*- C++ -*-
// Shuriken-Analyzer: library for bytecode analysis.
// @author Farenain <kunai.static.analysis@gmail.com>

#pragma once

#include "shuriken/sdk/dex/disassembly_constants.hpp"
#include "shuriken/internal/engine/dex/dex_engine.hpp"
#include "shuriken/sdk/dex/dvm_types.hpp"
#include "shuriken/sdk/dex/dvm_prototypes.hpp"
#include "shuriken/sdk/dex/field.hpp"

#include <span>
#include <cstdint>
#include <string>
#include <variant>


namespace shuriken {
namespace dex {

/// @brief Some instructions that depending on its kind
/// will make use of a type or another.
using kind_type_t = std::variant<
        std::monostate,
        DVMType *,
        DVMPrototype *,
        Field*,
        Method*,
        std::string_view>;

class InstructionProvider {
private:
    disassembler::dexinsttype instruction_type;
protected:
    std::span<std::uint8_t> bytecode;
    std::uint32_t length;
    disassembler::opcodes opcode;
    std::uint64_t address;
    std::string instruction;
public:
    InstructionProvider(std::span<std::uint8_t> bytecode, std::size_t index, disassembler::dexinsttype instruction_type);
    InstructionProvider(std::span<std::uint8_t> bytecode, std::size_t index, disassembler::dexinsttype instruction_type, std::uint32_t length);

    virtual ~InstructionProvider();

    virtual disassembler::kind get_kind() const;

    virtual disassembler::dexinsttype get_instruction_type() const;

    virtual std::uint32_t get_instruction_length() const;

    virtual disassembler::opcodes get_instruction_opcode() const;

    virtual void set_address(std::uint64_t address);

    virtual std::uint64_t get_address() const;

    virtual std::span<std::uint8_t> get_instruction_bytecode() const;

    virtual std::string_view print_instruction() = 0;

    virtual std::string print_instruction_string() = 0;

    virtual bool is_terminator() const;

    virtual bool has_side_effects() const;

    virtual bool may_throw() const;
};

class Instruction00xProvider : public InstructionProvider {
public:
    Instruction00xProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction00xProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction10xProvider : public InstructionProvider {
public:
    Instruction10xProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction10xProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction12xProvider : public InstructionProvider {
private:
    std::uint8_t vA;
    std::uint8_t vB;
public:
    Instruction12xProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction12xProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVA() const;
    disassembler::operand_type get_vA_type() const;

    std::uint8_t getVB() const;
    disassembler::operand_type get_vB_types() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction11nProvider : public InstructionProvider {
private:
    std::uint8_t vA;
    std::int8_t nB;
public:
    Instruction11nProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction11nProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVA() const;
    disassembler::operand_type get_vA_type() const;

    std::int8_t getNB() const;
    disassembler::operand_type get_nB_types() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction11xProvider : public InstructionProvider {
private:
    std::uint8_t vAA;
public:
    Instruction11xProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction11xProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction10tProvider : public InstructionProvider {
private:
    std::int8_t nAA;
public:
    Instruction10tProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction10tProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::int8_t getNAA() const;
    disassembler::operand_type get_nAA_type() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction20tProvider : public InstructionProvider {
private:
    std::int16_t nAAAA;
public:
    Instruction20tProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction20tProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::int16_t getNAAAA() const;
    disassembler::operand_type get_nAAAA_type() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction20bcProvider : public InstructionProvider {
private:
    std::uint8_t nAA;
    std::uint16_t nBBBB;
public:
    Instruction20bcProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction20bcProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getNAA() const;
    disassembler::operand_type get_nAA_type() const;

    std::uint16_t getNBBBB() const;
    disassembler::operand_type get_nBBBB_type() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction22xProvider : public InstructionProvider {
private:
    std::uint8_t vAA;
    std::uint16_t vBBBB;
public:
    Instruction22xProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction22xProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;

    std::uint16_t getVBBBB() const;
    disassembler::operand_type get_vBBBB_type() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction21tProvider : public InstructionProvider {
private:
    std::uint8_t vAA;
    std::int16_t nBBBB;
public:
    Instruction21tProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction21tProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;

    std::int16_t getNBBBB() const;
    disassembler::operand_type get_nBBBB_type() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction21sProvider : public InstructionProvider {
private:
    std::uint8_t vAA;
    std::int16_t nBBBB;
public:
    Instruction21sProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction21sProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;

    std::int16_t getNBBBB() const;
    disassembler::operand_type get_nBBBB_type() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction21hProvider : public InstructionProvider {
private:
    std::uint8_t vAA;
    std::int64_t nBBBB;
public:
    Instruction21hProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction21hProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;

    std::int64_t getnBBBB() const;
    disassembler::operand_type get_nBBBB_type() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction21cProvider : public InstructionProvider {
private:
    std::uint8_t vAA;
    std::uint16_t iBBBB;
    kind_type_t source_id;
public:
    Instruction21cProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction21cProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;

    std::uint16_t getIBBBB() const;
    disassembler::operand_type get_iBBBB_type() const;

    kind_type_t get_iBBBB_kind();

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction23xProvider : public InstructionProvider {
private:
    std::uint8_t vAA;
    std::uint8_t vBB;
    std::uint8_t vCC;
public:
    Instruction23xProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction23xProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;

    std::uint8_t getVBB() const;
    disassembler::operand_type get_vBB_type() const;

    std::uint8_t getVCC() const;
    disassembler::operand_type get_vCC_type() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction22bProvider : public InstructionProvider {
private:
    std::uint8_t vAA;
    std::uint8_t vBB;
    std::int8_t nCC;
public:
    Instruction22bProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction22bProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVAA() const;
    disassembler::operand_type get_vAA_type() const;

    std::uint8_t getVBB() const;
    disassembler::operand_type get_vBB_type() const;

    std::int8_t getNCC() const;
    disassembler::operand_type get_nCC_type() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction22tProvider : public InstructionProvider {
private:
    std::uint8_t vA;
    std::uint8_t vB;
    std::int16_t nCCCC;
public:
    Instruction22tProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction22tProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVA() const;
    disassembler::operand_type get_vA_type() const;

    std::uint8_t getVB() const;
    disassembler::operand_type get_vB_type() const;

    std::int16_t getNCCCC() const;
    disassembler::operand_type get_nCCCC_type() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction22sProvider : public InstructionProvider {
private:
    std::uint8_t vA;
    std::uint8_t vB;
    std::int16_t nCCCC;
public:
    Instruction22sProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction22sProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVA() const;
    disassembler::operand_type get_vA_type() const;

    std::uint8_t getVB() const;
    disassembler::operand_type get_vB_type() const;

    std::int16_t getNCCCC() const;
    disassembler::operand_type get_nCCCC_type() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction22cProvider : public InstructionProvider {
private:
    std::uint8_t vA;
    std::uint8_t vB;
    std::uint16_t iCCCC;
    kind_type_t checked_id;
public:
    Instruction22cProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction22cProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVA() const;
    disassembler::operand_type get_vA_type() const;

    std::uint8_t getVB() const;
    disassembler::operand_type get_vB_type() const;

    std::uint16_t getICCCC() const;
    disassembler::operand_type get_iCCCC_type() const;

    kind_type_t get_checked_id_as_kind() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction22csProvider : public InstructionProvider {
private:
    std::uint8_t vA;
    std::uint8_t vB;
    std::uint16_t iCCCC;
    kind_type_t field;
public:
    Instruction22csProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction22csProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::uint8_t getVA() const;
    disassembler::operand_type get_vA_type() const;

    std::uint8_t getVB() const;
    disassembler::operand_type get_vB_type() const;

    std::uint16_t getICCCC() const;
    disassembler::operand_type get_iCCCC_type() const;

    kind_type_t get_field() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

class Instruction30tProvider : public InstructionProvider {
private:
    std::int32_t nAAAAAAAA;
public:
    Instruction30tProvider(std::span<uint8_t> bytecode, std::size_t index);
    Instruction30tProvider(std::span<uint8_t> bytecode, std::size_t index, DexEngine & dex);

    std::int32_t getNAAAAAAAA() const;
    disassembler::operand_type get_nAAAAAAAA_type() const;

    std::string_view print_instruction() override;
    std::string print_instruction_string() override;
};

} //! namespace dex
} //! namespace shuriken